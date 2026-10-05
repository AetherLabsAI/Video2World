"""RoboDojo evaluator: execute a submitted package with the native RoboDojo task in Isaac Sim and score it.

The agent delivers a camera-frame package (scene + dual-arm joint commands at 25 Hz). It is converted to the world
frame with the agent's own camera pose, executed by the native task on the submitted geometry, audited, and scored
against the source recording of the demonstration. Writes EVAL_DIR/eval.json with the metric record under 'eval2'.

Usage: python -m v2w.tracks.robodojo.evaluate --run PKG --sample SAMPLE_DIR --out EVAL_DIR/eval.json
"""
import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np

from v2w import paths
from . import isaac
from .audit import audit, initialization_failure, visibility_failure
from .contract import PROFILE, CandidateError, load, public_task, validate, world_scene
from .metrics import definition, prepare_source, read, score, write

XARM_MESHES = paths.asset('robodojo', 'xarm7')   # official xArm gripper and visual meshes


def agent_failure(out, reason, **extra):
    result = dict(schema_version='eval2/4', build=False, task_success=False, delivery_failure=reason, progress={'progress': 0.}, error_kind='agent', **extra)
    write(out, dict(build=False, error_kind='agent', eval2=result))
    return result


def execute(pkg, sample, out):
    """Execute and score a world-frame package (profile robodojo_candidate_task_v1)."""
    pkg, sample, out = (Path(p).resolve() for p in (pkg, sample, out))
    ev = out.parent; ev.mkdir(parents=True, exist_ok=True)
    source = paths.resolve(read(sample / 'hidden/source_binding.json')['record'])
    meta = read(sample / 'meta.json'); task = meta['task']; kind = meta['embodiment']
    spec = definition(task, source)
    target = ev / 'pkg_baseframe'
    if target.exists():
        raise FileExistsError(target)
    shutil.copytree(pkg, target)
    try:
        scene, _ = validate(target, spec, kind)
    except (CandidateError, KeyError, ValueError, OSError) as exc:
        return agent_failure(out, str(exc))
    runtime = ev / 'runtime'; runtime.mkdir()
    record = ev / 'native/execution_1'; record.parent.mkdir()
    cmd = isaac.command('record', '--candidate-task', '--task', task, '--embodiment', kind, '--layout', target / 'scene.json',
                        '--actions', target / 'actions.npy', '--out', record, '--robot-meshes', XARM_MESHES,
                        '--capture-product', 'tiled', '--render-flush-steps', '12')
    with (ev / 'execute.log').open('w') as log:
        rc = subprocess.run(['timeout', '--signal=TERM', '--kill-after=15', '6000', *cmd], cwd=paths.REPO, env=isaac.environment(runtime),
                            stdout=log, stderr=subprocess.STDOUT).returncode
    rejected = initialization_failure(record)
    if rejected is not None:
        write(out, dict(build=False, error_kind='agent', eval2=rejected, record=str(record)))
        return rejected
    if rc != 0:
        raise RuntimeError(f'RoboDojo recorder failed (rc={rc}); see {ev / "execute.log"}')
    report = audit(record)
    rejected = visibility_failure(report)
    if rejected is not None:
        write(out, dict(build=False, error_kind='agent', eval2=rejected, record=str(record)))
        return rejected
    if report['status'] != 'pass' or not report['render_gt_alignment_verified']:
        raise RuntimeError('Execution data/render audit failed; not a scored task failure')
    subprocess.run([paths.tool('ffmpeg'), '-v', 'error', '-i', str(record / 'video.mp4'), '-c:v', 'libx264', '-crf', '22', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', str(record / 'review.mp4')], check=True)
    prepare_source(task, source, ev / 'metric_source')
    result = score(task, sample, record, ev / 'metric_source', ev / 'metrics', submitted_scene=scene)
    write(out, dict(build=True, eval2=result, record=str(record)))
    return result


def evaluate(pkg, sample, out):
    """Evaluate an agent package: camera-frame packages are converted to the world frame first."""
    pkg, sample, out = map(Path, (pkg, sample, out))
    try:
        protocol = read(pkg / 'protocol.json')
        if not isinstance(protocol, dict):
            raise ValueError('protocol.json must be an object')
    except (ValueError, OSError) as error:
        return agent_failure(out, str(error))
    if protocol.get('physics_profile') == PROFILE:
        return execute(pkg, sample, out)
    try:
        pr, scene, actions, t = load(pkg, 'robodojo_candidate_task_v1', public_task(sample))
    except (ValueError, KeyError, TypeError, OSError) as error:
        return agent_failure(out, str(error), sample=sample.name)
    converted = out.parent / 'camera_conversion'
    if converted.exists():
        raise FileExistsError(converted)
    shutil.copytree(pkg, converted)
    world = world_scene(scene, t)
    width, height = read(sample / 'meta.json')['resolution']
    world['camera'] = dict(pos=t[:3, 3].tolist(), look_at=(t[:3, 3] + t[:3, 2]).tolist(), T_world_camera=t.tolist(), width=width, height=height,
                           focal_px=float(pr['camera']['focal_px']))
    write(converted / 'scene.json', world); np.save(converted / 'actions.npy', actions)
    pr.update(physics_profile=PROFILE, frame='world_z_up', scene_file='scene.json')
    pr['actions']['path'] = 'actions.npy'; write(converted / 'protocol.json', pr)
    write(out.parent / 'camera_conversion.json', dict(protocol='camera-stage1-native/1', T_world_camera=t.tolist(), estimated_by='candidate'))
    return execute(converted, sample, out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--run', required=True); p.add_argument('--sample', required=True); p.add_argument('--out', required=True)
    p.add_argument('--video-dir')
    a = p.parse_args(argv)
    evaluate(a.run, a.sample, a.out)


if __name__ == '__main__':
    main()
