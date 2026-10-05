"""Public material staged next to the agent's video: task specification, robot model and task rules.

Only public semantics and robot assets are staged; no file of the sample except its video ever enters the sandbox.
"""
import hashlib
import json
import shutil
from pathlib import Path

from v2w import paths
from v2w.agents.checks import TWIN_PROFILES

HERE = Path(__file__).resolve().parent
FILES = HERE / 'handouts'
TASKS = HERE / 'tasks.json'


def file_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file()}


def public_task(sample):
    return json.loads(TASKS.read_text()).get(Path(sample).name)


def stage_native(sample, destination):
    """Public task + robot handout for the RoboDojo and in-house native profiles; returns the public_task.json path."""
    destination = Path(destination)
    spec = dict(public_task(sample))
    dst = destination / 'robot'
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(paths.asset('robots', spec['robot_asset']), dst)
    rule = FILES / 'task_rules' / (spec['task'] + '.py')
    if rule.is_file():
        shutil.copyfile(rule, destination / 'task_rules.py')
        spec['task_rules_file'] = str(destination / 'task_rules.py')
    name = Path(sample).name
    if name.startswith('inhouse_') and spec.get('task_type') == 'initially_held_insert':
        spec['task_description'] = 'Carry the object through the receiver mouth, release it and leave it stably inside; protrusion above the rim is allowed.'
        spec['task_rule_revision'] = 'inhouse-owner-task/2'
        spec['task_rules'] = dict(spec.get('task_rules', {}) if isinstance(spec.get('task_rules'), dict) else {}, min_carry_m=.05, release_hold_s=.25,
                                  settle_s=1., settle_position_range_m=.005, progress_stages=['carry', 'through_mouth', 'release_inside', 'retained'],
                                  containment='object bbox center in physically validated receiver-local task region; not full mesh containment')
        shutil.copyfile(FILES / 'reviewed_task_rules.py', destination / 'task_rules.py')
        spec['task_rules_file'] = str(destination / 'task_rules.py')
    if name.startswith('robodojo_'):
        shutil.copyfile(FILES / 'completion_transition.py', destination / 'completion_transition.py')
        spec['completion_transition_file'] = str(destination / 'completion_transition.py')
        spec['completion_transition_rule'] = ('An initially completed object goal requires an observed exit and re-entry; robot-home or gripper changes alone '
                                              'do not complete it. Native task thresholds and normal-start temporal semantics are preserved.')
    if spec.get('task') == 'sort_nesting_dolls_by_size':
        shutil.copyfile(FILES / 'nesting_dolls_size_rule.py', destination / 'nesting_dolls_size_rule.py')
        spec['task_rule_revision'] = 'nesting-dolls-executed-size/1'
        spec['executed_size_rule_file'] = str(destination / 'nesting_dolls_size_rule.py')
        spec['size_order_rule'] = ('Overrides original task_rules.py model_id ranking: smallest to largest by local-z height of actual executed geometry, '
                                   'with root scale baked once. JSON array order, model_id and submitted bbox metadata do not define size. Heights within '
                                   '1e-6 m are ambiguous and cannot complete this task. Native spacing, alignment, upright, robot-home and completion-transition '
                                   'checks remain required.')
    if spec.get('state_kind') == 'complete_deformable_surface':
        spec = wallet_spec(spec)
        shutil.copyfile(FILES / 'reviewed_task_rules.py', destination / 'task_rules.py')
        spec['task_rules_file'] = str(destination / 'task_rules.py')
        shutil.copyfile(FILES / 'wallet_contract.py', destination / 'wallet_contract.py')
        spec['deformable_validator'] = str(destination / 'wallet_contract.py')
    spec['robot_asset_directory'] = str(dst)
    spec['robot_asset_sha256'] = file_hashes(dst)
    target = destination / 'public_task.json'
    target.write_text(json.dumps(spec, indent=2) + '\n')
    shutil.copyfile(FILES / 'native_contract.py', destination / 'native_contract.py')
    return target


def wallet_spec(spec):
    """Wallet tasks accept a rigid or an elastic target."""
    spec = json.loads(json.dumps(spec))
    spec['state_kind'] = 'rigid_or_elastic_surface'
    spec['allowed_state_kinds'] = ['rigid_object_pose', 'complete_deformable_surface']
    spec['wallet_dynamics'] = 'rigid_or_elastic'
    spec.pop('deformable_contract', None)
    spec['geometry_contract'] = ('Rigid wallet: triangle mesh and state_kind=rigid_object_pose, omit deformable. Elastic wallet: closed particle shell and '
                                 'state_kind=complete_deformable_surface. No post-initialization object pose writes or attachments.')
    c = spec.get('task_rules', {})
    kind = spec['task_type']
    c.update(schema='inhouse-owner-task/2', grasp_hold_s=.15, terminal_hold_s=.25, min_lift_m=.05, progress_lift_m=.03, release_open_max=.2,
             release_hold_s=.25, settle_s=1., settle_center_range_m=.005, min_carry_m=.05)
    c['stages'] = ['grasp', 'lift', 'held_high'] if kind == 'pick_and_hold' else ['carry', 'through_mouth', 'release_inside', 'retained']
    c['region_rule'] = 'Object bbox center passes through receiver mouth, releases and remains stable for final 1s; protrusion above rim allowed.'
    spec['task_rules'] = c
    if kind == 'initially_held_insert':
        spec['task_description'] = 'Pass the held wallet through the open bag mouth, release it and leave it stably inside. Protrusion above the rim is allowed.'
    return spec


def stage_twin(profile, destination):
    """Public robot handout for the reconstructed-twin profiles; returns the public_task.json path."""
    if profile not in TWIN_PROFILES:
        raise ValueError('unknown twin profile: ' + profile)
    dst = Path(destination)
    dst.mkdir(parents=True, exist_ok=True)
    robot = dst / 'robot'
    if robot.exists():
        raise FileExistsError('refusing to overwrite staged robot assets')
    shutil.copytree(paths.asset('robots', 'twin_xarm7'), robot)
    spec = dict(protocol='twin-camera-agent/1', profile=profile, physics_profile='twin_mujoco_v2', coordinate_frame='first_camera_opencv',
                video_only=True, robot='xarm7_gripper', robot_revision='xarm7-gripper/urdf-v2',
                control_columns=['joint' + str(i) for i in range(1, 8)] + ['drive_joint'], gripper='radians: 0 open, 0.85 closed',
                robot_assets=file_hashes(robot))
    path = dst / 'public_task.json'
    path.write_text(json.dumps(spec, indent=2) + '\n')
    return path


def media_info(video):
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError('video cannot be opened')
    try:
        w, h, n = [int(cap.get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_COUNT)]
        if min(w, h, n) <= 0:
            raise ValueError('video has invalid dimensions or frame count')
        return w, h, n
    finally:
        cap.release()
