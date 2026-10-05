"""Trajectory metrics on the demonstration clock and the terminal position error."""
from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------- trajectory protocols
# Fixed relative action/trajectory contract; no candidate-dependent fitting or time warp.

FIXED_RELATIVE_VERSION = 'fixed-relative-actions/1.0'
DEMO_CLOCK_VERSION = 'demonstration-clock-trajectory/1.0'
DURATION_NORMALIZED_VERSION = 'duration-normalized-trajectory/1.0'


def vector3(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError('task anchor must be a finite 3-vector')
    return value


def translate_actions(actions, gt_anchor, candidate_anchor):
    """Only XYZ changes. Orientations, grip/fingers, ordering and timing remain frozen."""
    out = np.array(actions, dtype=float, copy=True)
    if out.ndim != 2 or out.shape[1] != 7 or not len(out) or not np.isfinite(out).all():
        raise ValueError('expected finite nonempty T x 7 TCP/palm actions')
    out[:, :3] += vector3(candidate_anchor) - vector3(gt_anchor)
    return out


def relative_pose_metrics(simulated, gt, candidate_anchor, gt_anchor, dt, symmetry=None, axis='z'):
    """Same declared sample clock. Never resample to the shorter output or align a trajectory."""
    from v2w.metrics import geometry as M
    a, b = np.asarray(simulated, float), np.asarray(gt, float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 7 or len(a) == 0:
        raise ValueError('reference/observed pose sequences must have identical T x 7 shape')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(dt) or dt <= 0:
        raise ValueError('nonfinite reference trajectory or invalid sample clock')
    ar, br = a.copy(), b.copy()
    ar[:, :3] -= vector3(candidate_anchor); br[:, :3] -= vector3(gt_anchor)
    translation = np.linalg.norm(ar[:, :3] - br[:, :3], axis=1) * 100
    rotation = np.array([M.sym_rot_err(x[3:], y[3:], symmetry, axis) for x, y in zip(ar, br)])
    ape = dict(M.ape_capped(ar, br), rot_deg=float(rotation.mean()),
               trans_per_frame_cm=translation.tolist(), rot_per_frame_deg=rotation.tolist(),
               trans_p90_cm=float(np.percentile(translation, 90)))
    # One physical interval, with its discretization recorded; never a candidate-chosen lag.
    delta = max(1, int(round(.2 / dt)))
    rpe = M.rpe(ar, br, delta=delta, symmetry=symmetry, axis=axis, time_s=np.arange(len(ar))*dt) if len(a) > delta else None
    if rpe is not None and not rpe['n']: rpe = None
    return dict(ape=ape, rpe=rpe, rpe_na_reason=None if rpe else 'sequence shorter than fixed RPE interval',
                time_s=(np.arange(len(a)) * dt).tolist(), n_frames=len(a), dt=dt,
                coordinate_frame='initial-object translation only; fixed base axes',
                anchor_gt=vector3(gt_anchor).tolist(), anchor_candidate=vector3(candidate_anchor).tolist(),
                translation_m=(vector3(candidate_anchor)-vector3(gt_anchor)).tolist())


def demo_clock_metrics(simulated, sim_dt, gt, gt_dt, symmetry=None, axis='z'):
    """Trajectory error of ANY execution against the demonstrated object motion, sampled on the demonstration clock
    (the agent's own execution is scored against the demonstration).

    The execution keeps its own declared clock: for each demonstration timestamp we take the execution's nearest recorded
    state. There is no time warping, no resampling of one sequence onto the other's index range and no trimming to the
    shorter sequence. An execution that ends before the demonstration holds its final state, and the held share is
    reported (`coverage`), so a short stream cannot be mistaken for an accurate one; frames past the demonstration window
    are outside the comparison (`unscored_tail_frames`). Both trajectories are then expressed relative to their own
    initial object position, as in the fixed-relative protocol, so initial placement stays with the geometric metrics."""
    a, b = np.asarray(simulated, float), np.asarray(gt, float)
    for name, x in (('execution', a), ('demonstration', b)):
        if x.ndim != 2 or x.shape[1] != 7 or not len(x) or not np.isfinite(x).all():
            raise ValueError(f'{name} pose sequence must be a finite nonempty T x 7 array')
    if not np.isfinite(sim_dt) or sim_dt <= 0 or not np.isfinite(gt_dt) or gt_dt <= 0:
        raise ValueError('both clocks must be finite positive sample intervals')
    t_gt = np.arange(len(b)) * gt_dt; t_sim = np.arange(len(a)) * sim_dt
    idx = np.clip(np.rint(t_gt / sim_dt).astype(int), 0, len(a) - 1)
    held = int((t_gt > t_sim[-1] + .5 * sim_dt).sum())
    ape, rpe = _paired_errors(a[idx], b, gt_dt, symmetry, axis)
    out = dict(protocol=DEMO_CLOCK_VERSION, ape=ape, rpe=rpe,
               rpe_na_reason=None if rpe else 'demonstration shorter than the fixed RPE interval',
               n_frames=len(b), dt=gt_dt, execution_dt=sim_dt, execution_frames=len(a),
               duration_s=float(t_gt[-1]), execution_duration_s=float(t_sim[-1]),
               held_frames=held, coverage=float(1 - held / len(b)),
               unscored_tail_frames=int((t_sim > t_gt[-1] + .5 * sim_dt).sum()),
               sampling='nearest state on the execution\'s own declared clock; no time warp',
               coordinate_frame='initial-object translation only; fixed base axes')
    out['duration_normalized'] = duration_normalized_metrics(a, sim_dt, b, gt_dt, symmetry, axis)
    return out


def _paired_errors(sampled, gt, gt_dt, symmetry, axis):
    """capped APE (symmetry-aware rotation) and RPE at the fixed 0.2 s interval for two sequences already put in correspondence,
    each re-based on its own initial object position"""
    from v2w.metrics import geometry as M
    ar, br = np.asarray(sampled, float).copy(), np.asarray(gt, float).copy()
    ar[:, :3] -= ar[0, :3]; br[:, :3] -= br[0, :3]
    translation = np.linalg.norm(ar[:, :3] - br[:, :3], axis=1) * 100
    rotation = np.array([M.sym_rot_err(x[3:], y[3:], symmetry, axis) for x, y in zip(ar, br)])
    ape = dict(M.ape_capped(ar, br), rot_deg=float(rotation.mean()),
               trans_per_frame_cm=translation.tolist(), rot_per_frame_deg=rotation.tolist(),
               trans_p90_cm=float(np.percentile(translation, 90)))
    delta = max(1, int(round(.2 / gt_dt)))
    rpe = M.rpe(ar, br, delta=delta, symmetry=symmetry, axis=axis, time_s=np.arange(len(ar))*gt_dt) if len(br) > delta else None
    if rpe is not None and not rpe['n']: rpe = None
    return ape, rpe


def duration_normalized_metrics(simulated, sim_dt, gt, gt_dt, symmetry=None, axis='z'):
    """DIAGNOSTIC companion of demo_clock_metrics: the same errors after rescaling the execution's duration to the
    demonstration's. An agent that reproduces the demonstrated path several times slower keeps a
    large error on the demonstration clock; this column separates the path from its timing.

    Each demonstration frame k is paired with the execution state at the same fraction of the execution's own duration
    (a single uniform rescaling, never a per-frame alignment or a dynamic time warp). The factor is reported as `time_scale`,
    so a comparison made at a different speed can never be read as one made at the demonstrated speed. Positions are
    re-based on each sequence's own initial object position, as in the headline."""
    a, b = np.asarray(simulated, float), np.asarray(gt, float)
    n = len(b)
    idx = np.clip(np.rint(np.linspace(0, len(a) - 1, n)).astype(int), 0, len(a) - 1) if n > 1 else np.zeros(1, int)
    ape, rpe = _paired_errors(a[idx], b, gt_dt, symmetry, axis)
    t_sim, t_gt = (len(a) - 1) * sim_dt, (n - 1) * gt_dt
    return dict(protocol=DURATION_NORMALIZED_VERSION, ape=ape, rpe=rpe,
                rpe_na_reason=None if rpe else 'demonstration shorter than the fixed RPE interval',
                time_scale=(float(t_sim / t_gt) if t_gt > 0 else None), n_frames=n,
                duration_s=float(t_gt), execution_duration_s=float(t_sim),
                sampling='execution duration rescaled uniformly onto the demonstration; diagnostic only',
                coordinate_frame='initial-object translation only; fixed base axes')


# ---------------------------------------------------------------- terminal position
# GT-selected endpoint position error, independent of execution timing and scene placement.
#
# This is an endpoint metric, not task success or demonstration-clock trajectory APE.
# Candidate poses must already use the evaluator's GT object-origin convention.

TERMINAL_POSITION_VERSION = 'terminal-position/1.1'
REGION_TYPES = {'in_region', 'in_region_upright', 'hoi4d_place'}


def anchor_kind(phi):
    """Never select an easier anchor because candidate geometry/metrics are missing."""
    if phi.get('target'):
        return 'receiver'
    if phi.get('type') in REGION_TYPES:
        return 'initial_object'
    if phi.get('gt_rel') is not None:
        return 'receiver'
    return None


def pose7(value):
    a = np.asarray(value, dtype=float)
    if a.shape != (7,) or not np.isfinite(a).all() or np.linalg.norm(a[3:]) == 0:
        raise ValueError('endpoint requires a finite pose7 with a nonzero quaternion')
    return a


def terminal_position(phi, candidate_initial, candidate_final, gt_trajectory,
                      candidate_target=None, gt_target_pose=None, axis='z',
                      candidate_receiver_missing=False):
    from v2w.metrics import geometry as M
    anchor = anchor_kind(phi)
    if anchor is None:
        raise ValueError('GT task does not define a terminal position anchor')
    ci, cf = pose7(candidate_initial), pose7(candidate_final)
    gt = np.asarray(gt_trajectory, dtype=float)
    if gt.ndim != 2 or gt.shape[1] != 7 or len(gt) < 1:
        raise ValueError('missing GT object trajectory')
    gi, gf = pose7(gt[0]), pose7(gt[-1])
    if anchor == 'initial_object':
        cd, gd = cf[:3] - ci[:3], gf[:3] - gi[:3]
        raw = 100 * float(np.linalg.norm(cd - gd))
        result = dict(rel_trans_err_cm=min(raw, M.CAP_CM), rel_trans_err_raw_cm=raw,
                      left_workspace=raw > M.CAP_CM, rel_rot_err_deg=None,
                      rotation_na_reason='initial-position anchor defines translation only',
                      frame='initial-object translation only; fixed base axes',
                      candidate_anchor=ci[:3].tolist(), gt_anchor=gi[:3].tolist(),
                      candidate_displacement_m=cd.tolist(), gt_displacement_m=gd.tolist())
    else:
        ref = phi.get('gt_rel')
        if ref is None and phi.get('rel_pos') is not None:
            # Some region goals declare a receiver-relative position but no rotation target.
            pos = np.asarray(phi['rel_pos'], dtype=float)
            if pos.shape != (3,) or not np.isfinite(pos).all():
                raise ValueError('invalid GT receiver-relative position')
            ref = np.eye(4); ref[:3, 3] = pos
            rotation_defined = False
        elif ref is None:
            if gt_target_pose is None:
                raise ValueError('missing GT receiving-object pose')
            ref = np.linalg.inv(M.T_of(pose7(gt_target_pose))) @ M.T_of(gf)
            rotation_defined = True
        else:
            rotation_defined = True
        ref = np.asarray(ref, dtype=float)
        if ref.shape != (4, 4) or not np.isfinite(ref).all():
            raise ValueError('invalid GT terminal relative transform')
        if candidate_target is None:
            if candidate_receiver_missing is not True:
                raise ValueError('GT requires a receiving object but candidate target pose is missing or unresolved')
            anchor = 'receiver_missing'
            result = dict(rel_trans_err_cm=M.CAP_CM, rel_trans_err_raw_cm=None,
                          rel_rot_err_deg=None, receiver_present=False,
                          penalty='missing_candidate_receiver',
                          reason='candidate delivered no receiving object',
                          raw_na_reason='no candidate receiver exists; distance cannot be measured',
                          rotation_na_reason='candidate receiving object missing',
                          left_workspace=None)
        else:
            if candidate_receiver_missing:
                raise ValueError('receiver marked missing but candidate target pose is present')
            result = M.terminal_relative(cf, pose7(candidate_target), ref, phi.get('axis', axis))
            result['receiver_present'] = True
        result.update(target_source='missing' if anchor == 'receiver_missing' else 'scene', target_name=phi.get('target'),
                      gt_reference_source='gt_rel' if phi.get('gt_rel') is not None else
                      ('rel_pos' if phi.get('rel_pos') is not None else 'gt_terminal_poses'))
        if not rotation_defined and anchor != 'receiver_missing':
            result.update(rel_rot_err_deg=None, rotation_na_reason='GT declares relative position only')
    result.update(protocol=TERMINAL_POSITION_VERSION, anchor=anchor, source='own_actions', stage='terminal',
                  cap_cm=M.CAP_CM, candidate_initial=ci.tolist(), candidate_final=cf.tolist(),
                  gt_initial=gi.tolist(), gt_final=gf.tolist(),
                  endpoint_rule='candidate final_poses after execution; GT last object pose; no time resampling')
    return result


def record_terminal_position(out, phi, rollout, name, gt_trajectory, canonical_transform,
                             candidate_target=None, gt_target_pose=None, axis='z',
                             candidate_receiver_missing=False):
    """Only submitted execution may fill the headline; initial/hidden metrics never do."""
    from v2w.metrics import geometry as M
    from v2w.metrics.record import metric_error
    out['terminal_position_metric_version'] = TERMINAL_POSITION_VERSION
    if anchor_kind(phi) is None:
        return
    try:
        if not rollout or not rollout.get('traj') or name not in rollout['traj']:
            raise ValueError('missing submitted execution trajectory')
        result = terminal_position(phi,
            M.canon_pose7(pose7(rollout['init_poses'][name]), canonical_transform),
            M.canon_pose7(pose7(rollout['final_poses'][name]), canonical_transform),
            gt_trajectory, candidate_target, gt_target_pose, axis, candidate_receiver_missing)
        out['relative'] = result
        out.setdefault('metric_status', {})['relative.rel_trans_err_cm'] = {
            'status': 'ok', 'version': TERMINAL_POSITION_VERSION, 'anchor': result['anchor']}
        if result.get('reason'):
            out['metric_status']['relative.rel_trans_err_cm']['reason'] = result['reason']
        if result.get('rotation_na_reason'):
            out['metric_status']['relative.rel_rot_err_deg'] = {
                'status': 'na', 'reason': result['rotation_na_reason']}
    except (KeyError, TypeError, ValueError, IndexError, np.linalg.LinAlgError) as e:
        out['relative'] = None
        metric_error(out, ('relative.rel_trans_err_cm',), f'{type(e).__name__}: {e}')
