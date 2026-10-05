"""Scene step of the hand-track evaluator: does the package build in `V2SEgoHand-v1`, the first-generation scene Chamfer
(`video2sim.bench.evaluate.scene_point_cloud`) and the object correspondence by name / initial position that the metric
record reads as `object_match`.

Usage: python -m v2w.tracks.hand.scene PKG --sample SAMPLE_DIR --out scene.json
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
from video2sim.bench import metrics
from video2sim.bench.evaluate import scene_point_cloud
from video2sim.bench.protocol import load_manifest, scene_in_base_frame, validate_manifest

# Bridge-profile rules that do not apply to a hand package: an egocentric camera-frame package may sit > 1 m away.
WAIVED_RULES = ('status success requires an action stream', 'actions positions must be in metres')


def build_scene(pkg, sd):
    rep = dict(package=str(pkg), sample=sd.name, mode='ego_scene', build=False)
    try:
        m = load_manifest(pkg); errs = validate_manifest(pkg, m)
        na = [e for e in errs if e.startswith(WAIVED_RULES)]; errs = [e for e in errs if e not in na]
        rep['validate_errors'] = errs; rep['not_applicable_rules'] = na; rep['status'] = m.get('status')
        if m.get('status') == 'infeasible':
            rep['stage'] = 'infeasible'; return rep
        scene = scene_in_base_frame(m, pkg)
        rep['n_objects'] = len(scene.get('objects') or []); rep['n_props'] = len(scene.get('props') or []); rep['joints_declared'] = [j.get('name') for j in (scene.get('joints') or [])]
        g = json.loads((sd / 'hidden/scene_gt.json').read_text()); gz = np.load(sd / 'hidden/objects_6d.npz')
        gt_traj = {n: gz[n] for n in g['object_frame0']}
        g_scene = {'support': g.get('support'), 'props': g['props'], 'objects': [dict(o, pos=g['object_frame0'][o['name']][:3], quat=g['object_frame0'][o['name']][3:]) for o in g['objects']]}
        try:
            pa, pb = scene_point_cloud(scene), scene_point_cloud(g_scene)
            rep['chamfer'] = {**metrics.chamfer(pa, pb), 'reference': 'video_model_fit', 'n_a': len(pa), 'n_b': len(pb)}
        except Exception as e:
            rep['chamfer_error'] = f'{type(e).__name__}: {e}'
        agent_init = {o['name']: np.asarray(list(o['pos']) + list(o.get('quat', (1, 0, 0, 0)))) for o in scene['objects']}
        from v2w.metrics.task import match_objects
        source = json.loads((sd / 'hidden/phi.json').read_text())['source']
        rep['object_match'], rep['object_match_evidence'] = match_objects(agent_init, {n: np.asarray(p[0]) for n, p in gt_traj.items()}, source=source, legacy_match=metrics.match_objects)
        import gymnasium as gym, mani_skill.envs  # noqa: F401
        from . import env as _env  # noqa: F401  registers V2SEgoHand-v1
        env = gym.make('V2SEgoHand-v1', obs_mode='none', num_envs=1, sim_backend='physx_cpu', render_backend='none', scene_spec=scene, camera=None, enable_cameras=False)
        try:
            env.reset(seed=0, options=dict(settle=0.0)); u = env.unwrapped
            rep['built_objects'] = list(u.objs); rep['built_props'] = list(u.props); rep['built_joints'] = list(getattr(u, 'joints', {}))
            for _ in range(30): u.step_physics(1)
            rep['build'] = True
        finally:
            try: env.close()
            except Exception: pass
    except Exception as e:
        rep['error'] = f'{type(e).__name__}: {e}'; rep['traceback'] = traceback.format_exc()[-3000:]
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('pkg'); ap.add_argument('--sample', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); t0 = time.time()
    rep = build_scene(Path(a.pkg), Path(a.sample))
    rep['seconds'] = round(time.time() - t0, 1); Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: v for k, v in rep.items() if k != 'traceback'}, default=str)[:600])


if __name__ == '__main__':
    main()
