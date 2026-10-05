"""Evaluate an in-house candidate package: validate it, execute it on the G1 robot in Isaac Sim, score the execution.

    python -m v2w.tracks.inhouse.evaluate --kind camera|point|reviewed|wallet|fryer --run PKG --sample SAMPLE_DIR --out EVAL_DIR/eval.json

camera    provider samples, and episode samples with accepted pose GT
point     tasks scored by a complete 3D point trajectory
reviewed  episodes whose pose GT has explicitly unknown frames (pick-and-hold, bag insertion)
wallet    rigid or elastic wallet target scored by its complete surface
fryer     air-fryer basket extraction

A package that violates the public contract is a method failure (build = false); any other error propagates and is
recorded by the runner as an evaluator failure.
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np

from v2w import paths
from . import scoring, tasks
from .gt import registry
from .package import read, write, load, world_scene, native_package, validate_wallet, robot_dir, local

AGENT_ERRORS = (ValueError, KeyError, TypeError, OSError)
WALLET_DIVERGENCE = ('Particle simulation diverged', 'Nonfinite native physics state')


def public_task(sd):
    p = Path(sd) / 'native_task.json'
    return read(p) if p.is_file() else registry()['provider_tasks'][Path(sd).name]


def method_failure(sd, out, error):
    result = dict(schema_version='eval2/4', sample=Path(sd).name, build=False, task_success=False, progress=dict(progress=0.), error_kind='agent', delivery_failure=str(error))
    write(out, dict(build=False, error_kind='agent', error=str(error), eval2=result))
    return result


# ---------------------------------------------------------------- package preparation
def prepare(kind, pkg, sd, dest):
    public = public_task(sd)
    if kind == 'wallet':
        return prepare_wallet(pkg, sd, dest, public)
    contract = read(sd / 'task_contract.json') if (sd / 'task_contract.json').is_file() else None
    _, _, q, _ = load(pkg, public)
    if kind in ('point', 'reviewed', 'fryer') and len(q) != public['control_frames']:
        raise ValueError('Actions must cover the declared full 20 Hz task clock')
    if contract is not None and (public['hand'] != contract['hand'] or public['task_type'] != contract['kind']):
        raise ValueError('Public task and evaluator task differ')
    initial_arm = None
    if kind == 'fryer':
        tasks.validate_fryer(contract)
        settle = 40
        if (pkg / 'initial_arm.npy').exists():
            initial_arm = np.load(local(pkg, 'initial_arm.npy'), allow_pickle=False)
            if initial_arm.shape != (14,) or not np.isfinite(initial_arm).all():
                raise ValueError('Invalid initial robot arm state')
    elif kind == 'reviewed' and contract['kind'] == 'initially_held_insert':
        if contract['schema'] != 'inhouse-rigid-insert-task/1':
            raise ValueError('Unsupported reviewed task')
        settle = 40
    elif contract is not None:
        tasks.validate_episode(contract)
        settle = 40 if contract['kind'] == 'initially_held_place' else 2
    else:
        settle = 40 if public['task'] in ('book', 'box') else 2
    world, _ = native_package(pkg, sd, dest, public, settle, initial_arm)
    if kind == 'reviewed' and contract['kind'] == 'initially_held_insert':
        tasks.inside_cavity(np.zeros((1, 1, 3)), next(e for e in world['props'] if e['name'] == 'receiver'), pkg, contract['containment_tolerance_m'])
    return world


def prepare_wallet(pkg, sd, dest, public):
    import shutil
    pr, scene, q, t = load(pkg, public)
    validate_wallet(scene, pkg)
    elastic = bool(scene['objects'][0].get('deformable'))
    expected = 'complete_deformable_surface' if elastic else 'rigid_object_pose'
    if pr.get('state_kind') != expected:
        raise ValueError('state_kind must be ' + expected + ' for the submitted dynamics')
    if len(q) != int(public['control_frames']):
        raise ValueError('Actions must match the public control_frames at 20 Hz; a fixed 2 s settle is added')
    world = world_scene(scene, t)
    if public['task_type'] == 'initially_held_insert':
        tasks.inside_cavity(np.zeros((1, 1, 3)), next(e for e in world['props'] if e['name'] == 'receiver'), pkg, .003)
    shutil.copytree(pkg, dest)
    write(dest / 'scene.json', world)
    np.save(dest / 'actions.npy', q)
    write(dest / 'protocol.json', dict(sample=Path(sd).name, kind=public['task_type'], state_kind=expected, source_camera_transform=t.tolist()))
    return world


# ---------------------------------------------------------------- Isaac execution
def execute(kind, package, run, log, video_dir=None):
    env = paths.env(PYTHONNOUSERSITE='1', OMNI_KIT_ACCEPT_EULA='YES', ACCEPT_EULA='Y', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    libs = paths.tool('graphics_libs', required=False)
    if libs:
        env['LD_LIBRARY_PATH'] = libs + ':' + env.get('LD_LIBRARY_PATH', '')
    mode = 'wallet' if kind == 'wallet' else 'rigid'
    if mode == 'wallet':   # GPU particle physics selects its device itself
        visible = env.pop('CUDA_VISIBLE_DEVICES', None)
        if visible and visible.isdigit():
            env['V2W_ISAAC_GPU'] = visible
    cmd = [paths.tool('isaac_python'), '-m', 'v2w.tracks.inhouse.isaac', mode, '--package', str(package), '--out', str(run), '--robot', str(robot_dir() / 'robot.usda')]
    if kind == 'fryer':
        cmd += ['--collision', 'decomposition']   # keep the basket and handle gaps physical
    if video_dir is not None and mode == 'rigid':
        cmd += ['--video-dir', str(Path(video_dir).resolve())]
    with open(log, 'w') as f:
        subprocess.run(cmd, env=env, cwd=paths.REPO, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=6000)


def encode_video(run, video_dir):
    """Encode the captured head / left / right frames of an execution, including its settling tail."""
    ex = read(Path(run) / 'execution.json')
    capture = ex['video_capture']
    ff = paths.tool('ffmpeg')
    for view in capture['views']:
        frames = sorted((Path(video_dir) / view).glob('f_*.jpg'))
        if len(frames) != ex['frames']:
            raise ValueError(view + ': missing video frames')
        subprocess.run([ff, '-v', 'error', '-y', '-framerate', '20', '-i', str(Path(video_dir) / view / 'f_%04d.jpg'), '-c:v', 'libx264', '-preset', 'fast',
                        '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(Path(video_dir) / (view + '.mp4'))], check=True, timeout=300)


# ---------------------------------------------------------------- evaluation
def evaluate(kind, pkg, sd, out, video_dir=None):
    pkg, sd, out = (Path(p).resolve() for p in (pkg, sd, out))
    ev = out.parent
    ev.mkdir(parents=True, exist_ok=True)
    episode = (sd / 'gt_acceptance.json').is_file() and (sd / 'task_contract.json').is_file()
    if kind == 'point':
        from .gt import point_task_gt
        point_task_gt(sd)   # a GT problem is an evaluator failure, never the method's
    native = ev / 'pkg_baseframe'
    try:
        world = prepare('episode' if kind == 'camera' and episode else kind, pkg, sd, native)
    except AGENT_ERRORS as error:
        return method_failure(sd, out, error)
    run = ev / 'native'
    execute(kind, native, run, ev / 'native.log', video_dir)
    if kind == 'wallet':
        status = read(run / 'execution.json')
        if status.get('status') != 'executed':
            error = str(status.get('error'))
            if any(k in error for k in WALLET_DIVERGENCE):
                return method_failure(sd, out, error)
            raise RuntimeError('Native deformable runtime failed: ' + error)
    if video_dir is not None and kind != 'wallet':
        encode_video(run, video_dir)
    if kind == 'camera':
        result = scoring.episode(sd, native, run, out) if episode else scoring.provider(sd, ev, world)
    elif kind == 'point':
        result = scoring.point(sd, native, run, out)
    elif kind == 'wallet':
        result = scoring.wallet(sd, native, run, out)
    else:
        result = scoring.masked_episode(sd, native, run, out, kind)
    write(out, dict(build=True, task_success=result['task_success'], eval2=result))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--kind', choices=['camera', 'point', 'reviewed', 'wallet', 'fryer'], default='camera')
    p.add_argument('--run', required=True, help='candidate package')
    p.add_argument('--sample', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--video-dir')
    a = p.parse_args()
    evaluate(a.kind, a.run, a.sample, a.out, a.video_dir)


if __name__ == '__main__':
    main()
