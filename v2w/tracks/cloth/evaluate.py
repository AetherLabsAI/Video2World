"""Evaluate a cloth-folding package: execute its own scene and actions in MuJoCo and score the cloth surface.

A candidate package (droid-cloth-camera/1) is expressed in the first video camera frame; the evaluator moves it into
the robot base frame with the hidden camera extrinsics, resamples the commands onto the 15 Hz control clock (time zero
is the first video frame, no duration normalisation) and scores the execution against the visible-surface GT, with
footprints measured in the GT table plane. The sample's own reference package (``gt_pkg``, base frame, recorded
clock) is executed as recorded, which reproduces the human-assisted reference.

python -m v2w.tracks.cloth.evaluate --run PKG --sample SAMPLE_DIR --out EVAL_DIR/eval.json [--video-dir DIR]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .contract import VERSION, load

FPS = 15


def finite(x):
    """JSON-safe copy: non-finite floats become null, numpy scalars become Python scalars."""
    if isinstance(x, dict):
        return {str(k): finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [finite(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return str(x) if isinstance(x, Path) else x


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite(value), indent=1, allow_nan=False) + '\n')


def align(scene, tcp, T):
    """Camera-frame scene and fingertip poses -> robot base frame; gravity along minus the table normal."""
    R, t = T[:3, :3], T[:3, 3]
    corners = np.asarray(scene['cloth']['corners']) @ R.T + t
    cloth = dict(scene['cloth'], corners=corners.tolist(), center=corners.mean(0).tolist(), yaw=0.,
                 size=[float(np.linalg.norm(corners[1] - corners[0])), float(np.linalg.norm(corners[3] - corners[0]))])
    point = R @ np.asarray(scene['table']['point']) + t
    normal = R @ np.asarray(scene['table']['normal'])
    table = dict(point=point.tolist(), normal=normal.tolist(), z=float(point[2]))
    return dict(table=table, cloth=cloth, gravity='table_normal'), T[None] @ tcp


def command_clock(tcp, grip, dt, horizon):
    """Commands on the 15 Hz control clock: poses interpolated (slerp), gripper held per action row, last pose held."""
    from scipy.spatial.transform import Rotation, Slerp
    times = np.arange(max(2, int(np.ceil(horizon * FPS)))) / FPS
    knots = np.arange(len(tcp)) * dt
    query = np.clip(times, 0, knots[-1])
    poses = np.tile(np.eye(4), (len(times), 1, 1))
    poses[:, :3, :3] = Slerp(knots, Rotation.from_matrix(tcp[:, :3, :3]))(query).as_matrix()
    for j in range(3):
        poses[:, j, 3] = np.interp(query, knots, tcp[:, j, 3])
    closures = grip[np.minimum(np.floor((times + 1e-9) / dt).astype(int), len(grip) - 1)]
    return poses, closures


def failure(sample, message, kind='agent', stage='validation'):
    record = dict(schema_version='eval2/4', sample=sample, build=False, task_success=False,
                  progress=dict(progress=0., source='own_actions'), error_kind=kind, error=message,
                  execution=dict(profile=VERSION, status='failed', stage=stage))
    return dict(sample=sample, build=False, task_success=False, error_kind=kind, error=message, eval2=record)


def record(sample, cloth, execution):
    rec = dict(schema_version='eval2/4', sample=sample, build=True, task_success=bool(cloth['task_success']), metric_status={},
               **{k: cloth[k] for k in ('progress', 'deformable', 'object', 'own_trajectory', 'relative')},
               cloth=cloth, metric_definitions=cloth['metric_definitions'], execution=execution)
    return dict(sample=sample, build=True, task_success=rec['task_success'], eval2=rec)


def video_path(video_dir, sample):
    if not video_dir:
        return None
    Path(video_dir).mkdir(parents=True, exist_ok=True)
    return str(Path(video_dir) / (sample + '_rollout.mp4'))


def evaluate_reference(pkg, sd, work, video_dir=None):
    """The sample's reference package: recorded fingertip poses and gripper closure on the recorded clock, base frame."""
    from .score import score
    from .simulate import execute
    scene = json.loads((pkg / 'scene.json').read_text())
    actions = np.load(pkg / 'actions.npz')
    surface = dict(np.load(sd / 'hidden/cloth_surface.npz', allow_pickle=True))
    visibility = dict(np.load(sd / 'hidden/visibility.npz'))
    ex = execute(scene, actions['tcp_pose'], actions['gripper_closure'], fps=float(actions['fps']), render=video_path(video_dir, sd.name))
    np.savez_compressed(work / 'executed.npz', vertices=ex['vertices'], final=ex['final'])
    cloth = score(surface, ex, scene['table'], visibility)
    run = dict(mode='reference_actions', profile='cloth/1', control_fps=float(actions['fps']), physics_dt=ex['dt'],
               grasp_frame=ex['grasp_frame'], release_frame=ex['release_frame'], captured_vertices=ex['captured'], finite=ex['finite'])
    return record(sd.name, cloth, run)


def evaluate_candidate(pkg, sd, work, video_dir=None):
    from .score import as_observed, densify, score
    from .simulate import execute
    try:
        m, scene, tcp, grip = load(pkg)
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as e:
        return failure(sd.name, str(e))
    # Evaluator data is read before the candidate runs, so missing or corrupt GT stays an evaluator error.
    T = np.asarray(json.loads((sd / 'hidden/camera.json').read_text())['extrinsics_base_cam'], float)
    surface = dict(np.load(sd / 'hidden/cloth_surface.npz', allow_pickle=False))
    visibility = dict(np.load(sd / 'hidden/visibility.npz', allow_pickle=False))
    times = np.asarray(np.load(sd / 'hidden/timestamps.npz', allow_pickle=False)['t'], float)
    if len(times) != len(surface['frames']) or len(times) < 2 or not np.all(np.diff(times) > 0) or abs(times[0]) > 1e-8:
        raise ValueError('invalid video timestamps')
    gt_table = json.loads((sd / 'hidden/scene_gt.json').read_text())['table']
    base, base_tcp = align(scene, tcp, T)
    dt = float(m['actions']['dt'])
    duration = len(tcp) * dt
    commands, closures = command_clock(base_tcp, grip, dt, max(duration, float(times[-1]) + 1 / FPS))
    try:
        ex = execute(base, commands, closures, FPS, render=video_path(video_dir, sd.name))
        if not ex['finite'] or not np.isfinite(ex['initial']).all() or not np.isfinite(ex['final']).all():
            raise ValueError('non-finite candidate simulation')
    except (ValueError, RuntimeError) as e:
        # A numerical reset is a candidate physics failure; renderer or driver errors propagate as evaluator errors.
        if isinstance(e, RuntimeError) and 'MuJoCo reset' not in str(e):
            raise
        return failure(sd.name, str(e), stage='simulation')
    # The executor samples after each control interval; prepend the settled initial state so index k is time k / FPS.
    V = np.concatenate([ex['initial'][None], ex['vertices']], axis=0)
    scored = dict(ex, vertices=V)
    for k in ('grasp_frame', 'release_frame'):
        if scored[k] is not None:
            scored[k] += 1
    frames = np.rint(times * FPS).astype(int)
    if np.max(np.abs(frames / FPS - times)) > 1e-6:
        raise ValueError('video timestamps are not on the 15 Hz clock')
    surface = dict(surface, frames=frames, time_s=times)
    np.savez_compressed(work / 'executed.npz', vertices=V, final=ex['final'], time_s=np.arange(len(V)) / FPS)
    write(work / 'scene_base.json', base)
    run = dict(mode='candidate_actions', profile=VERSION, control_fps=FPS, physics_dt=ex['dt'], candidate_duration_s=duration,
               executed_duration_s=len(commands) / FPS, time_origin='first video frame', time_scaling=False, finite=True,
               grasp_frame=scored['grasp_frame'], release_frame=scored['release_frame'], captured_vertices=ex['captured'])
    # A sheet the GT cameras cannot see at some frame cannot be scored: a method failure, not a scorer error.
    voxel = float(surface['voxel'])
    counts = [len(as_observed(densify(V[k], ex['grid']), visibility, i, voxel)) for i, k in enumerate(frames)]
    end = len(as_observed(densify(ex['final'], ex['grid']), visibility, len(times) - 1, voxel))
    if min(counts + [end]) < 4:
        result = failure(sd.name, 'candidate cloth has insufficient visible surface for scoring', stage='visibility')
        result['eval2']['execution'].update(run=run, visible_points=counts, terminal_visible_points=end)
        return result
    return record(sd.name, score(surface, scored, gt_table, visibility), run)


def evaluate(pkg, sample, out, video_dir=None):
    pkg, sd, out = Path(pkg), Path(sample), Path(out)
    work = out.parent
    work.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((pkg / 'protocol.json').read_text()) if (pkg / 'protocol.json').is_file() else {}
    if protocol.get('protocol_version') == 'cloth/1' and pkg.resolve() == (sd / 'gt_pkg').resolve():
        result = evaluate_reference(pkg, sd, work, video_dir)
    else:
        result = evaluate_candidate(pkg, sd, work, video_dir)
    write(out, result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--run', required=True)
    p.add_argument('--sample', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--video-dir')
    a = p.parse_args(argv)
    try:
        r = evaluate(a.run, a.sample, a.out, a.video_dir)
    except Exception as e:
        write(a.out, failure(Path(a.sample).name, type(e).__name__ + ': ' + str(e), kind='evaluator', stage='evaluator'))
        raise
    print(json.dumps({k: r[k] for k in ('sample', 'build', 'task_success')}))


if __name__ == '__main__':
    main()
