"""Explicit task semantics for new in-house episodes, independent of object class.

This does not change the frozen five-episode provider contract. New episode
packages must name the acting hand and task type; no object-name fallback.
"""
import copy
import numpy as np


def validate(contract):
    if contract.get('schema') != 'inhouse-episode-task/1':
        raise ValueError('Expected inhouse-episode-task/1')
    if contract.get('hand') not in ('l', 'r'):
        raise ValueError('Explicit task hand must be l or r')
    if contract.get('kind') not in ('pick_and_hold', 'initially_held_place'):
        raise ValueError('Unsupported episode task kind')
    for key in ('bilateral_force_min_n', 'grasp_hold_s', 'terminal_hold_s',
                'min_lift_m', 'progress_lift_m', 'release_hold_s', 'settle_s',
                'settle_position_range_m', 'settle_angle_range_rad',
                'support_height_tolerance_m', 'min_carry_m'):
        value = contract.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError('Invalid task threshold: ' + key)
    if not 0 <= contract.get('release_open_threshold', -1) <= 1:
        raise ValueError('Invalid release threshold')
    if contract['kind'] == 'initially_held_place':
        zone = np.asarray(contract.get('destination_zone'), float)
        if zone.shape != (5,) or not np.isfinite(zone).all() or np.any(zone[2:4] <= 0):
            raise ValueError('Place task requires a measured/configured receiver region')
    return contract


def make(template, *, hand, kind, destination_zone=None):
    result = copy.deepcopy(template)
    result.update(schema='inhouse-episode-task/1', hand=hand, kind=kind,
                  initially_held=kind == 'initially_held_place',
                  destination_zone=destination_zone,
                  progress_stages=['carry', 'release', 'supported_settled']
                  if kind == 'initially_held_place' else ['grasp', 'lift', 'held_high'])
    return validate(result)


def contact_forces(observations, hand, target='/World/Entities/target', physics_hz=60.):
    """Extract forces on opposing pads of the specified hand and target only."""
    if hand not in ('l', 'r'):
        raise ValueError('Unknown hand')
    forces = np.zeros((len(observations), 2, 3))
    def within(path, root):
        return path == root or path.startswith(root + '/')
    for i, row in enumerate(observations):
        for c in row['contacts']:
            a, b = c['body0'], c['body1']
            if within(a, target): other = b
            elif within(b, target): other = a
            else: continue
            impulse = np.asarray(c['impulse'], float)
            if impulse.shape != (3,) or not np.isfinite(impulse).all():
                raise ValueError('Invalid contact impulse')
            for side, part in enumerate(('inner', 'outer')):
                if any(segment.startswith('gripper_' + hand + '_' + part)
                       for segment in other.split('/')):
                    forces[i, side] += impulse * physics_hz
    return forces


def score_run(observations, trajectory, grip, contract):
    from inhouse.rgbd_provider_semantics import score
    validate(contract)
    bounds = np.asarray([r['bounds'] for r in observations], float)
    if bounds.shape != (len(observations), 2, 3) or not np.isfinite(bounds).all():
        raise ValueError('Invalid object bounds')
    xyz, quat = np.asarray(trajectory['xyz']), np.asarray(trajectory['wxyz'])
    grip = np.asarray(grip, float)
    if xyz.shape != (len(observations), 3) or quat.shape != (len(observations), 4):
        raise ValueError('Object trace and observations have different lengths')
    if grip.shape != (len(observations),) or not np.isfinite(grip).all() or np.any((grip < 0) | (grip > 1)):
        raise ValueError('Invalid grip trace')
    trace = dict(pad_force_vectors=contact_forces(observations, contract['hand']),
                 obj_xyz=xyz, obj_quat=quat, fabric_obj_xyz=xyz, usd_obj_wxyz=quat,
                 grip=grip, bbox_center_xyz=bounds.mean(1), support_z=bounds[:, 0, 2])
    return score(trace, contract)


def valid_rpe_intervals(valid, delta=4):
    """Do not bridge a missing GT observation, including intermediate frames."""
    valid = np.asarray(valid, bool)
    if valid.ndim != 1 or delta < 1:
        raise ValueError('Invalid mask or RPE interval')
    return np.array([valid[i:i + delta + 1].all()
                     for i in range(max(0, len(valid) - delta))], bool)
