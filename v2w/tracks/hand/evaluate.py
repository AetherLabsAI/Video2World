"""Evaluate one package on a hand-track sample (HOT3D / HOI4D / OakInk2 pick-place, DexYCB lift).

  1. frame alignment  the package is delivered in the frame-0 camera frame; the hidden camera pose maps it to the robot
                      base frame                                                         -> pkg_baseframe/
  2. shape fixes      container expansion and mesh sanitising (shared with the furniture track)
  3. scene            build in `V2SEgoHand-v1`, object correspondence                     -> scene_pkg.json
  4. rollout          the package's own actions on a Franka arm + gripper or Wuji hands   -> rollout_pkg.json
  5. summary          the task predicate on the rollout, gated on physical validity       -> eval.json

The metric record is built from these artifacts by `v2w.tracks.hand.metrics`.
Usage: python -m v2w.tracks.hand.evaluate --run PKG --sample SAMPLE_DIR --out EVAL_DIR/eval.json --embodiment arm|dexhand|dexhand_bimanual
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from v2w import paths

ICD = '/etc/vulkan/icd.d/nvidia_icd.json' if Path('/etc/vulkan/icd.d/nvidia_icd.json').exists() else '/usr/share/vulkan/icd.d/lvp_icd.x86_64.json'
EMBODIMENTS = ('arm', 'dexhand', 'dexhand_bimanual')


def jload(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None


def run(module, args, log, timeout):
    cmd = [sys.executable, '-m', module] + [str(a) for a in args]
    env = paths.env(VK_ICD_FILENAMES=__import__('os').environ.get('VK_ICD_FILENAMES', ICD))
    with open(log, 'a') as lf:
        lf.write(f'\n===== {time.strftime("%Y-%m-%d %H:%M:%S")} {" ".join(cmd)}\n'); lf.flush()
        try:
            return subprocess.run(cmd, cwd=str(paths.REPO), env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            lf.write('TIMEOUT\n'); return -9


def evaluate(pkg, sd, outj, embodiment, video_dir=None):
    work = outj.parent; work.mkdir(parents=True, exist_ok=True); log = work / 'evaluate.log'
    vid = (lambda tag: ['--video', str(video_dir / f'{sd.name}_{tag}.mp4')]) if video_dir else (lambda tag: ['--no-render'])
    t0 = time.time()
    res = dict(run=str(pkg), sample=sd.name, evaluator='v2w.tracks.hand', embodiment=embodiment, build=False, replay_success=False)
    phi = jload(sd / 'hidden/phi.json')
    # 1. frame alignment (hidden frame-0 camera pose)
    from v2w.metrics.frames import align_package
    cam = jload(sd / 'hidden/camera.json'); Tbc = np.asarray(cam['extrinsics_base_cam'], float)
    aligned = work / 'pkg_baseframe'
    try:
        res['frame_align'] = {k: v for k, v in align_package(pkg, Tbc, aligned).items() if k in ('agent_extrinsics_identity', 'actions_T', 'trajectory_frames')}
    except Exception as e:
        res.update(error=f'frame_align failed: {type(e).__name__}: {e}', error_kind='package', traceback=traceback.format_exc()[-2000:])
        return res
    # 2. shape fixes
    from v2w.tracks.furniture.evaluate import expand_cylinder_containers, sanitize_meshes
    try:
        res['shape_shim_expanded'] = expand_cylinder_containers(aligned); res['mesh_sanitized'] = sanitize_meshes(aligned)
    except Exception as e:
        res['shape_shim_error'] = f'{type(e).__name__}: {e}'
    # 3. scene
    scene_j = work / 'scene_pkg.json'
    run('v2w.tracks.hand.scene', [aligned, '--sample', sd, '--out', scene_j], log, 1800)
    sc = jload(scene_j) or {}
    res['build'] = bool(sc.get('build')); res['validate_errors'] = sc.get('validate_errors'); res['scene_error'] = sc.get('error')
    res['object_match'] = sc.get('object_match'); res['joints_declared'] = sc.get('joints_declared')
    if not res['build']:
        res['error'] = f"package does not build in V2SEgoHand-v1: {sc.get('error') or sc.get('validate_errors')}"; res['error_kind'] = 'package'
        return res
    # 4. own-actions rollout
    om = json.dumps({k: v for k, v in (sc.get('object_match') or {}).items() if v})
    rollout_j = work / 'rollout_pkg.json'
    run('v2w.tracks.hand.rollout', [aligned, '--sample', sd, '--out', rollout_j, '--embodiment', embodiment, '--object-match', om] + vid('rollout'), log, 5400)
    # 5. summary verdict (the metric record recomputes it from the same artifacts)
    from v2w.metrics import task as T
    from . import phi as hand_phi
    from .physics import gate_success
    hand_phi.patch()
    try:
        ctx = T.context(sd, aligned, scene_j)
    except Exception as e:
        ctx = None; res['canon_error'] = f'{type(e).__name__}: {e}'
    ro = jload(rollout_j)
    if ro and ro.get('ok') and ro.get('final_poses'):
        name = ro.get('target') or (phi['source'] if phi['source'] in ro['final_poses'] else list(ro['final_poses'])[0])
        try:
            verdict = T.verdict(ctx, phi, ro, name) if ctx is not None else dict(type=phi['type'], success=False, reason='canonical context unavailable')
            res['task_success_raw'] = bool(verdict['success']); res['phi_detail'] = verdict
            res['replay_success'], res['physical_validity'] = gate_success(verdict['success'], ro)
        except Exception as e:
            res['replay_success'] = False; res['phi_detail'] = dict(type='error', success=False, reason=f'{type(e).__name__}: {e}')
        res['own_actions'] = dict(N=ro.get('N'), dt=ro.get('dt'), actions_format=ro.get('actions_format'), quiescent=ro.get('quiescent'), attach_events=len(ro.get('attach_events', [])),
                                  release_events=len(ro.get('release_events', [])), grasp_misses=ro.get('grasp_misses'), held_at_end=bool((ro.get('grasping_final') or {}).get(name)),
                                  embodiment=ro.get('embodiment'), robot=ro.get('robot'), ik_pos_err_max_cm=ro.get('ik_pos_err_max_cm'), root_track_err_cm=ro.get('root_track_err_cm'))
    else:
        res['replay_success'] = False; res['phi_detail'] = dict(type='none', success=False, reason=(ro or {}).get('error', 'package delivers no actions / rollout crashed'))
    res['seconds'] = round(time.time() - t0)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--sample', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--video-dir', default=None, help='render the own-actions rollout into DIR/<sample>_rollout.mp4')
    ap.add_argument('--embodiment', default='arm', choices=EMBODIMENTS, help='arm = Franka arm + gripper; dexhand = one floating Wuji hand; dexhand_bimanual = two Wuji hands')
    a = ap.parse_args()
    outj = Path(a.out).resolve(); vd = Path(a.video_dir).resolve() if a.video_dir else None
    if vd: vd.mkdir(parents=True, exist_ok=True)
    res = evaluate(Path(a.run).resolve(), Path(a.sample).resolve(), outj, a.embodiment, vd)
    outj.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ('phi_detail', 'traceback')}, default=str)[:800])


if __name__ == '__main__':
    main()
