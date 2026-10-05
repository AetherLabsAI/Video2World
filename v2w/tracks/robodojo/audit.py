"""Recording audit and the method-failure outcomes it can establish.

``audit`` checks a finished recording (trajectory, masks, video, camera projection, gripper/mimic state).
Two evidenced outcomes turn a recording into a method failure instead of an evaluator error: the robot could not
start from the prescribed home posture on the submitted scene, or no task object is ever visible.
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np

PROJECTION_MIN_PIXELS = 30
SMALL_REASON = 'insufficient task-object pixels for projection audit'
NO_VISIBLE_REASON = 'no visible task object'
INITIALIZATION_REASON = 'invalid candidate robot initialization'
INITIALIZATION_SCOPE = 'pre-rollout prescribed robot home and mimic integrity; not task success'


def check_initial_state(names, actual, expected, relations, tolerance=.02):
    """Robot start against its prescribed home posture and mimic relations (runs inside the recorder)."""
    q = np.asarray(actual, float); home = np.asarray(expected, float)
    finite = bool(np.isfinite(q).all())
    deviations = np.abs(q - home)
    bad = [{'joint': name, 'actual': float(q[i]), 'expected': float(home[i]), 'error': float(deviations[i])}
           for i, name in enumerate(names) if not np.isfinite(q[i]) or deviations[i] > tolerance]
    mimic = []
    for rel in relations:
        i = names.index(rel['follower']); j = names.index(rel['reference'])
        error = float(abs(q[i] - q[j] * rel['multiplier'] - rel['offset']))
        mimic.append(dict(rel, error_rad=error, passed=bool(np.isfinite(error) and error <= tolerance)))
    return {'passed': finite and not bad and all(v['passed'] for v in mimic),
            'home_tolerance_joint_units': tolerance, 'home_errors': bad, 'mimic_checks': mimic, 'scope': INITIALIZATION_SCOPE}


def initialization_failure(record):
    """Metric record of an evidenced initialization rejection, else None (ordinary error handling applies)."""
    record = Path(record)
    try:
        run_status = json.loads((record / 'run.json').read_text())
        if run_status.get('status') != 'error' or run_status.get('scope') != 'agent candidate scene execution':
            return None
        phase = next((phase for phase in ('before_scene_restore', 'after_scene_restore')
                      if run_status.get('error') == repr(RuntimeError('Invalid candidate robot initialization at ' + phase + '; do not score this rollout'))), None)
        if phase is None:
            return None
        gates = json.loads((record / ('initialization_gate_' + phase + '.json')).read_text())
        if not isinstance(gates, list) or not gates:
            return None
        failed = False
        for gate in gates:
            if gate.get('scope') != INITIALIZATION_SCOPE or type(gate.get('passed')) is not bool:
                return None
            tol = gate.get('home_tolerance_joint_units')
            if tol != .02:
                return None
            home, mimic = gate['home_errors'], gate['mimic_checks']
            if not isinstance(home, list) or not isinstance(mimic, list):
                return None
            for item in home:
                a, e, err = map(float, (item['actual'], item['expected'], item['error']))
                if not item['joint'] or not math.isfinite(e):
                    return None
                if math.isfinite(a):
                    if not math.isfinite(err) or not math.isclose(abs(a - e), err, rel_tol=1e-6, abs_tol=1e-9) or err <= tol:
                        return None
                elif math.isfinite(err):
                    return None
            for item in mimic:
                err = float(item['error_rad'])
                if type(item.get('passed')) is not bool or item['passed'] != (math.isfinite(err) and err <= tol):
                    return None
            expected = not home and all(v['passed'] for v in mimic)
            if gate['passed'] != expected:
                return None
            failed |= not expected
        if not failed:
            return None
        return dict(schema_version='eval2/4', build=False, task_success=False, progress=dict(progress=0.), error_kind='agent',
                    delivery_failure=INITIALIZATION_REASON, task_failure_reason=INITIALIZATION_REASON,
                    initialization_gate=dict(phase=phase, checks=gates),
                    execution_validity=dict(valid=False, scope='candidate initialization rejected before rollout'))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def mapped_instance_ids(info, objects):
    """Resolve exact semantic class or prim subtree, never name substrings."""
    result = {i: [] for i in range(len(objects))}; unknown = set()
    for sid, label in info.get('idToLabels', {}).items():
        path = str(label); semantic = info.get('idToSemantics', {}).get(str(sid), {})
        if isinstance(semantic, dict):
            semantic = semantic.get('class')
        matched = [i for i, o in enumerate(objects) if semantic == o.get('label') or
                   path == o.get('prim_path') or path.startswith(str(o.get('prim_path', '__missing__')) + '/')]
        if len(matched) == 1:
            result[matched[0]].append(int(sid))
        elif int(sid) > 0:
            unknown.add(path)
    return result, unknown


def visibility_reason(audit):
    objects = [o for o in audit.get('objects', []) if o.get('type') == 'Rigid']
    if objects and all(o.get('visible_frames') == 0 and o.get('max_pixels') == 0 for o in objects):
        return NO_VISIBLE_REASON
    if not objects:
        return None
    for o in objects:
        n = o.get('max_pixels'); frames = o.get('visible_frames')
        if type(n) is not int or type(frames) is not int or not 0 <= n < PROJECTION_MIN_PIXELS or frames < 0 or (n == 0) != (frames == 0):
            return None
    return SMALL_REASON


def _unobservable(audit):
    return (audit.get('status') == 'pass' and audit.get('invalid_instance_mask_frames') == []
            and audit.get('all_scene_instances_mapped') is True and visibility_reason(audit) is not None
            and (audit.get('projected_center_to_visible_mask_centroid_px') or {}).get('samples') == 0
            and not audit.get('render_gt_projected_bounds_check'))


def visibility_failure(audit):
    """Metric record when no task object is observable in an otherwise valid recording, else None."""
    if (_unobservable(audit) and audit.get('render_gt_projection_available') is True
            and audit.get('render_gt_alignment_verified', False) is None
            and audit.get('render_gt_alignment_reason') == visibility_reason(audit)):
        reason = visibility_reason(audit)
        return dict(schema_version='eval2/4', build=False, task_success=False, progress=dict(progress=0.), error_kind='agent',
                    delivery_failure=reason, task_failure_reason=reason, control_audit=audit,
                    execution_validity=dict(valid=True, scope='completed recording and instance-mask integrity; render/physics alignment unobservable'))
    return None


def audit(root):
    """Integrity audit of a finished recording; writes audit.json and visibility.npz."""
    import cv2
    root = Path(root)
    run_status = json.loads((root / 'run.json').read_text())
    if run_status['status'] != 'recorded':
        raise ValueError('Recording did not finish')
    z = np.load(root / 'trajectory.npz', allow_pickle=False)
    poses = z['object_pose_wxyz']; ts = z['time_s']; T, N, _ = poses.shape
    objects = json.loads((root / 'objects.json').read_text())
    maps = json.loads((root / 'instance_maps.json').read_text())
    masks = np.load(root / 'instance_masks.npz', allow_pickle=False)['masks']
    assert poses.shape == (T, len(objects), 7) and len(maps) == T and masks.shape[0] == T
    assert np.isfinite(poses).all() and np.isfinite(z['robot_actions']).all()
    assert np.allclose(np.diff(ts), 1 / 25, rtol=0, atol=1e-8)
    quat_error = float(np.max(np.abs(np.linalg.norm(poses[..., 3:], axis=-1) - 1)))
    assert quat_error < .002
    cap = cv2.VideoCapture(str(root / 'video.mp4')); count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    cap.release(); assert count == T, (count, T)
    inventory = json.loads((root / 'scene_inventory.json').read_text()).get('entities', []) if (root / 'scene_inventory.json').exists() else objects
    scene_vis = np.zeros((T, len(inventory)), dtype=int)
    vis = np.zeros((T, N), dtype=int); unknown = set(); invalid_masks = []
    for t, info in enumerate(maps):
        allowed = [int(k) for k in info.get('idToLabels', {})]
        invalid_count = int(np.count_nonzero(~np.isin(masks[t], allowed)))
        if invalid_count:
            invalid_masks.append({'frame': t, 'unknown_id_pixels': invalid_count, 'fraction': invalid_count / masks[t].size})
        active = dict(info, idToLabels={k: v for k, v in info.get('idToLabels', {}).items() if np.any(masks[t] == int(k))})
        scene_ids, extra = mapped_instance_ids(active, inventory); unknown.update(extra)
        for i, values in scene_ids.items():
            scene_vis[t, i] = int(np.count_nonzero(np.isin(masks[t], values)))
        ids, _ = mapped_instance_ids(active, objects)
        for i, values in ids.items():
            vis[t, i] = int(np.count_nonzero(np.isin(masks[t], values)))
    motion = np.linalg.norm(poses[:, :, :3] - poses[0:1, :, :3], axis=-1)
    per_object = [dict(label=o.get('label', o['inst_name']), type=o['object_type'], visible_frames=int((vis[:, i] > 0).sum()),
                       max_pixels=int(vis[:, i].max()), max_displacement_cm=float(motion[:, i].max() * 100),
                       final_position=poses[-1, i, :3].tolist()) for i, o in enumerate(objects)]
    report = {'status': 'fail' if invalid_masks else 'pass', 'invalid_instance_mask_frames': invalid_masks,
              'frames': T, 'fps': 25, 'tracked_objects': N, 'video_frames': count,
              'quaternion_norm_max_error': quat_error, 'objects': per_object,
              'unmapped_instance_labels': sorted(unknown),
              'all_task_rigid_objects_have_visible_mask': all(o['visible_frames'] > 0 for o in per_object if o['type'] == 'Rigid'),
              'all_scene_instances_mapped': not unknown and not invalid_masks,
              'scene_entities': [dict(o, visible_frames=int((scene_vis[:, i] > 0).sum()), max_pixels=int(scene_vis[:, i].max())) for i, o in enumerate(inventory)],
              'render_gt_alignment_verified': False,
              'native_predicates': run_status.get('native_predicates')}
    if (root / 'robot_limits.json').exists():
        limits = json.loads((root / 'robot_limits.json').read_text()); diagnostics = []
        for side, lim in enumerate(limits):
            q = z['robot_joint_positions'][:, side]; bounds = np.asarray(lim['joint_pos_limits'])
            over = np.maximum(np.maximum(bounds[:, 0] - q, q - bounds[:, 1]), 0)
            entry = {'side': side, 'joint_names': lim['joint_names'], 'max_position_limit_excess': over.max(axis=0).tolist(),
                     'frames_exceeding_1e_minus_4': int(np.any(over > 1e-4, axis=1).sum())}
            if 'robot_joint_velocities' in z and 'joint_vel_limits' in lim:
                vel = np.abs(z['robot_joint_velocities'][:, side]); vl = np.asarray(lim['joint_vel_limits'])
                entry['max_abs_velocity'] = vel.max(axis=0).tolist(); entry['max_velocity_excess'] = np.maximum(vel - vl, 0).max(axis=0).tolist()
            diagnostics.append(entry)
        report['runtime_joint_limit_diagnostics'] = diagnostics
    terminal = poses[max(0, T - 25):, :, :3]
    report['terminal_object_drift_last_second_cm'] = {o.get('label', o['inst_name']): float(np.linalg.norm(terminal[:, i] - terminal[-1, i], axis=-1).max() * 100)
                                                      for i, o in enumerate(objects)}
    if (root / 'robot_control_contract.json').exists():
        control = json.loads((root / 'robot_control_contract.json').read_text())['sides']
    elif run_status.get('embodiment') == 'x5':
        control = [{'side': side, 'gripper_open_joint_positions': {'joint7': .044, 'joint8': .044},
                    'gripper_joint_types': {'joint7': 'prismatic', 'joint8': 'prismatic'}} for side in range(2)]
    else:
        raise ValueError('A non-X5 gripper requires an explicit per-joint control contract')
    robot_names = json.loads((root / 'robot_joints.json').read_text()); open_checks = []; mimic_checks = []
    for side, contract in enumerate(control):
        expected = dict(contract['gripper_open_joint_positions'])
        relations = contract.get('mimic_relations', [])
        for _ in range(len(relations) + 1):
            for rel in relations:
                if rel['reference'] in expected:
                    expected[rel['follower']] = expected[rel['reference']] * rel['multiplier'] + rel['offset']
        for rel in relations:
            ref = robot_names[side].index(rel['reference']); child = robot_names[side].index(rel['follower'])
            residual = z['robot_joint_positions'][:, side, child] - (z['robot_joint_positions'][:, side, ref] * rel['multiplier'] + rel['offset'])
            maximum = float(np.max(np.abs(residual)))
            mimic_checks.append(dict(rel, side=side, max_abs_error_rad=maximum, rms_error_rad=float(np.sqrt(np.mean(residual ** 2))),
                                     tolerance_rad=.02, passed=maximum <= .02))
        for name, target in expected.items():
            kind = contract['gripper_joint_types'].get(name, 'unknown')
            tolerance = .001 if kind == 'prismatic' else .02 if kind in {'revolute', 'continuous'} else None
            actual = float(z['robot_joint_positions'][-1, side, robot_names[side].index(name)])
            open_checks.append({'side': side, 'joint': name, 'type': kind, 'actual': actual, 'open_target': target,
                                'tolerance': tolerance, 'passed': tolerance is not None and abs(actual - target) <= tolerance})
    report['mimic_constraint_checks'] = mimic_checks
    report['mimic_constraints_passed'] = all(c['passed'] for c in mimic_checks)
    report['gripper_open_checks'] = open_checks
    report['both_grippers_open'] = bool(open_checks) and all(c['passed'] for c in open_checks)
    # Rendered masks must lie inside the projected physics bounds (optical camera frame).
    camera = json.loads((root / 'camera.json').read_text())
    if 'world_to_camera_ros' in camera:
        w2c = np.asarray(camera['world_to_camera_ros']); K = np.asarray(camera['intrinsics'])
        errors = []; bbox_checks = []
        geometry = json.loads((root / 'geometry/manifest.json').read_text()) if (root / 'geometry/manifest.json').exists() else {'objects': []}
        bounds = {g['label']: np.asarray(g['bounds_root']) for g in geometry['objects'] if g.get('kind') == 'object' and g.get('bounds_root')}
        for t in range(T):
            mapping, _ = mapped_instance_ids(maps[t], objects)
            for i, o in enumerate(objects):
                if o['object_type'] != 'Rigid' or vis[t, i] < PROJECTION_MIN_PIXELS:
                    continue
                ids = mapping[i]
                if not ids:
                    continue
                yy, xx = np.nonzero(np.isin(masks[t], ids))
                point = w2c @ np.r_[poses[t, i, :3], 1.]
                if point[2] <= 0:
                    continue
                uv = K @ point[:3]; uv = uv[:2] / uv[2]
                errors.append(float(np.linalg.norm(uv - [xx.mean(), yy.mean()])))
                bound = bounds.get(o.get('label'))
                if bound is not None:
                    corners = np.array(list(itertools.product(*bound.T)))
                    w, x, y, zq = poses[t, i, 3:] / np.linalg.norm(poses[t, i, 3:])
                    R = np.array([[1 - 2 * (y * y + zq * zq), 2 * (x * y - zq * w), 2 * (x * zq + y * w)],
                                  [2 * (x * y + zq * w), 1 - 2 * (x * x + zq * zq), 2 * (y * zq - x * w)],
                                  [2 * (x * zq - y * w), 2 * (y * zq + x * w), 1 - 2 * (x * x + y * y)]])
                    world = corners @ R.T + poses[t, i, :3]
                    cam = np.c_[world, np.ones(8)] @ w2c.T
                    projected = cam[:, :3] @ K.T; projected = projected[:, :2] / projected[:, 2:3]
                    if np.all(cam[:, 2] > 0):
                        lo = projected.min(0); hi = projected.max(0)
                        excess = max(0, lo[0] - xx.min(), lo[1] - yy.min(), xx.max() - hi[0], yy.max() - hi[1])
                        bbox_checks.append({'frame': t, 'label': o.get('label'), 'excess_px': float(excess)})
        report['projected_center_to_visible_mask_centroid_px'] = {'samples': len(errors), 'median': float(np.median(errors)) if errors else None,
                                                                  'p95': float(np.percentile(errors, 95)) if errors else None}
        if bbox_checks:
            values = [b['excess_px'] for b in bbox_checks]
            report['render_gt_projected_bounds_check'] = {'samples': len(values), 'tolerance_px': 3., 'max_excess_px': max(values),
                                                          'p95_excess_px': float(np.percentile(values, 95)),
                                                          'failures': [b for b in bbox_checks if b['excess_px'] > 3.]}
            report['render_gt_alignment_verified'] = not invalid_masks and max(values) <= 3.
    transform = np.asarray(camera.get('world_to_camera_ros', []), dtype=float)
    intrinsics = np.asarray(camera.get('intrinsics', []), dtype=float)
    projection_available = (transform.shape == (4, 4) and intrinsics.shape == (3, 3) and np.isfinite(transform).all()
                            and np.isfinite(intrinsics).all() and intrinsics[0, 0] > 0 and intrinsics[1, 1] > 0)
    report['render_gt_projection_available'] = bool(projection_available)
    if projection_available and _unobservable(report):
        report['render_gt_alignment_verified'] = None
        report['render_gt_alignment_reason'] = visibility_reason(report)
    np.savez_compressed(root / 'visibility.npz', pixels=vis, time_s=ts, labels=z['object_labels'], scene_pixels=scene_vis,
                        scene_labels=[o.get('label', 'unknown') for o in inventory])
    (root / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
