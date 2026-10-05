"""Evaluate one package of the furniture track (FurnitureBench, DROID pick-and-place, Push-T).

Steps, every artifact kept under the output directory:
  1. frame alignment   the package is delivered in the camera frame; re-express it in the robot base frame with the
                       hidden camera extrinsics                                   -> pkg_baseframe/
  2. shape repair      expand `cylinder_container` props into the boxes video2sim builds and drop OBJ faces that
                       index missing vertices (the simulator's loader ignores them; trimesh would raise)
  3. scene             validate the package, build its scene in the simulator, match objects   -> scene_pkg.json
  4. rollout           the package's own actions in its own scene                 -> rollout_pkg.json
The metric record is computed from these artifacts by v2w.metrics.record.rigid_record.

Usage: python -m v2w.tracks.furniture.evaluate --run PKG --sample SAMPLE_DIR --out EVAL_DIR/eval.json [--video-dir DIR]
"""
import argparse
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from v2w import paths
from v2w.metrics.frames import align_package

NAME = 'pkg'   # artifact prefix: pkg_baseframe/, scene_pkg.json, rollout_pkg.json


# ---------------------------------------------------------------- shape repair
def expand_cylinder_containers(pkg: Path) -> int:
    """Replace `cylinder_container` props by the box parts of video2sim.shapes.cylinder_container_parts (origin at the
    inner floor centre). Returns the number of expanded props."""
    from video2sim.shapes import cylinder_container_parts
    mf = pkg / 'protocol.json'
    m = json.loads(mf.read_text())
    props = (m.get('scene') or {}).get('props') or []
    out, n = [], 0
    for p in props:
        if p.get('kind') != 'cylinder_container':
            out.append(p); continue
        parts = cylinder_container_parts(float(p['inner_diameter']), float(p['height']), wall=float(p.get('wall', 0.003)), sections=int(p.get('sections', 20)))
        base = [float(v) for v in p['pos']]
        for i, (off, half, yaw) in enumerate(parts):
            out.append(dict(name=f"{p.get('name', 'vessel')}_part{i:02d}", kind='box', half_size=[float(v) for v in half],
                            pos=[base[0] + float(off[0]), base[1] + float(off[1]), base[2] + float(off[2])],
                            quat=[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], color=p.get('color'),
                            static_friction=p.get('static_friction', 1.0), dynamic_friction=p.get('dynamic_friction', 1.0)))
        n += 1
    if n:
        m['scene']['props'] = out
        mf.write_text(json.dumps(m, indent=1))
    return n


def _sanitize_obj(path: Path) -> int:
    """Drop faces referencing vertex indices beyond the vertex list (1-based; relative indices are left alone)."""
    lines = path.read_text().splitlines(keepends=True)
    nv = sum(1 for l in lines if l.startswith('v '))
    out, dropped = [], 0
    for l in lines:
        if l.startswith('f '):
            try:
                idx = [int(t.split('/')[0]) for t in l.split()[1:]]
            except ValueError:
                out.append(l); continue
            if any(i > nv or i == 0 for i in idx):
                dropped += 1; continue
        out.append(l)
    if dropped:
        path.write_text(''.join(out))
    return dropped


def sanitize_meshes(pkg: Path) -> dict:
    """{package-relative OBJ: faces dropped} over the mesh and collision files of every mesh entity."""
    m = json.loads((pkg / 'protocol.json').read_text())
    sc = m.get('scene') or {}
    seen, report = set(), {}
    for e in (sc.get('objects') or []) + (sc.get('props') or []):
        if e.get('kind') != 'mesh':
            continue
        for key in ('mesh_path', 'collision_path'):
            rel = e.get(key)
            if not rel or rel in seen:
                continue
            seen.add(rel)
            f = Path(rel) if Path(rel).is_absolute() else pkg / rel
            if f.suffix.lower() == '.obj' and f.exists():
                n = _sanitize_obj(f)
                if n:
                    report[rel] = n
    return report


# ---------------------------------------------------------------- scene
def scene_report(pkg: Path, sample: Path) -> dict:
    """Build: the package validates and its scene builds in the simulator. Also the initial-position object matching."""
    from video2sim.bench import metrics
    from video2sim.bench.evaluate import scene_point_cloud
    from video2sim.bench.protocol import load_manifest, validate_manifest, scene_in_base_frame
    rep = {'package': str(pkg), 'sample': sample.name, 'mode': 'scene_only', 'build': False}
    try:
        meta = json.loads((sample / 'meta.json').read_text())
        rep['sample'] = meta.get('sample_id', sample.name)
    except Exception as e:
        rep.update(build=None, error_kind='evaluator', error=f'{type(e).__name__}: {e}')
        return rep
    try:
        m = load_manifest(pkg); errs = validate_manifest(pkg, m)
        rep['validate_errors'] = errs; rep['status'] = m.get('status')
        if errs: raise RuntimeError('; '.join(errs))
        if m.get('status') == 'infeasible':
            rep['stage'] = 'infeasible'; return rep
        scene = scene_in_base_frame(m, pkg)
        g = json.loads((sample / 'hidden' / 'scene_gt.json').read_text()); gz = np.load(sample / 'hidden' / 'objects_6d.npz')
        gt_traj = {n: gz[n] for n in g['object_frame0']}
        g_scene = {'support': g.get('support'), 'props': g['props'], 'objects': [dict(o, pos=g['object_frame0'][o['name']][:3], quat=g['object_frame0'][o['name']][3:]) for o in g['objects']]}
        pa, pb = scene_point_cloud(scene), scene_point_cloud(g_scene)
        rep['chamfer'] = {**metrics.chamfer(pa, pb), 'n_a': len(pa), 'n_b': len(pb)}
        agent_init = {o['name']: np.asarray(o['pos'] + list(o.get('quat', (1, 0, 0, 0)))) for o in scene['objects']}
        rep['object_match'] = metrics.match_objects(agent_init, {n: np.asarray(p[0]) for n, p in gt_traj.items()})
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        from video2sim.bench import bridge_env  # noqa: F401  (registers V2SBridge-v1)
        env = gym.make('V2SBridge-v1', obs_mode='state', num_envs=1, sim_backend='physx_cpu', render_backend='none',
                       scene_spec=scene, camera=None, enable_cameras=False)
        try:
            env.reset(seed=0)
        finally:
            env.close()
        rep['build'] = True; rep['stage'] = 'done'
    except Exception as e:
        rep['error'] = f'{type(e).__name__}: {e}'; rep['traceback'] = traceback.format_exc()[-1500:]
    return rep


# ---------------------------------------------------------------- driver
def evaluate(run: Path, sample: Path, out: Path, video_dir: Path | None = None) -> dict:
    work = out.parent; work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    T_base_cam = np.asarray(json.loads((sample / 'hidden/camera.json').read_text())['extrinsics_base_cam'], float)
    aligned = work / f'{NAME}_baseframe'
    align_package(run, T_base_cam, aligned)
    expanded = expand_cylinder_containers(aligned)
    repaired = sanitize_meshes(aligned)

    scene_j = work / f'scene_{NAME}.json'
    scene = scene_report(aligned, sample)
    from video2sim.bench.evaluate import _json_default
    scene_j.write_text(json.dumps(scene, indent=2, default=_json_default))

    rollout_j = work / f'rollout_{NAME}.json'
    video = ['--video', str(video_dir / f'{sample.name}_rollout.mp4')] if video_dir else ['--no-render']
    subprocess.run([sys.executable, '-m', 'v2w.tracks.furniture.rollout', str(aligned), '--sample', str(sample), '--out', str(rollout_j)] + video,
                   cwd=str(paths.REPO), env=paths.env(), check=False, timeout=3600)

    res = dict(sample=sample.name, package=str(run), build=bool(scene.get('build')), shape_expanded=expanded, mesh_repairs=repaired)
    if scene.get('error'):
        res['scene_error'] = scene['error']
    ro = json.loads(rollout_j.read_text()) if rollout_j.exists() else None
    if ro is None:
        res.update(task_success=False, phi_detail=dict(success=False, reason='rollout crashed'))
    elif ro.get('ok') and ro.get('final_poses'):
        from v2w.metrics.task import context, verdict
        phi = json.loads((sample / 'hidden/phi.json').read_text())
        g = json.loads((sample / 'hidden/scene_gt.json').read_text()); nm = g['objects'][0]['name']
        key = nm if nm in ro['final_poses'] else list(ro['final_poses'])[0]
        r = verdict(context(sample, aligned, scene_j), phi, ro, key)
        res.update(task_success=bool(r['success']), phi_detail=r,
                   own_actions=dict(T=ro.get('T'), quiescent=ro.get('quiescent'), attach_events=len(ro.get('attach_events', [])), ik_pos_err_max_cm=ro.get('ik_pos_err_max_cm')))
    else:
        res.update(task_success=False, phi_detail=dict(success=False, reason=ro.get('error', 'own-actions rollout failed')))
    res['seconds'] = round(time.time() - t0, 1)
    out.write_text(json.dumps(res, indent=1, default=float))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', required=True); ap.add_argument('--sample', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--video-dir', default=None)
    a = ap.parse_args(argv)
    vd = Path(a.video_dir) if a.video_dir else None
    if vd: vd.mkdir(parents=True, exist_ok=True)
    res = evaluate(Path(a.run), Path(a.sample), Path(a.out), vd)
    print(json.dumps({k: res[k] for k in ('sample', 'build', 'task_success')}))


if __name__ == '__main__':
    main()
