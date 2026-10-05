"""Success predicates of the hand track, added to the shared predicates of `v2w.metrics.task`.

  in_container  the object's surface centroid lies inside the receiving prop (radial distance to the prop's symmetry axis
                <= rim radius, axial coordinate between the inner bottom and the rim plus the object's half height);
                released and quiescent required.
  hoi4d_place   resting in the demonstrated region with the demonstrated orientation, released and quiescent.
  lifted        the object origin rose by at least `min_dz` (release only when the predicate requires it).

`patch()` routes these predicate types through `eval_phi_hand` inside the metric layer (runtime only).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from v2w.metrics import task as T

HAND_TYPES = ('in_container', 'lifted', 'hoi4d_place')
_ORIG = dict(eval_phi2=T.eval_phi2, verdict=T.verdict, find_target_parts=T.find_target_parts)


def T_of(p):
    p = np.asarray(p, float); M = np.eye(4); M[:3, :3] = R.from_quat([p[4], p[5], p[6], p[3]]).as_matrix(); M[:3, 3] = p[:3]; return M


def in_container(phi, fin7, tgt7):
    """geometry part of `in_container` on one terminal pose (object pose7, receiving prop pose7)."""
    c = phi['container']; i = 'xyz'.index(c['axis']); e = np.eye(3)[i]
    Tt = T_of(tgt7); To = T_of(fin7)
    cen = To[:3, :3] @ np.asarray(phi['source_centroid_local'], float) + To[:3, 3]      # object surface centroid, base frame
    loc = np.linalg.inv(Tt)[:3, :3] @ cen + np.linalg.inv(Tt)[:3, 3] - np.asarray(c['axis_offset'], float)
    ax = float(loc @ e); rad = float(np.linalg.norm(loc - ax * e))
    hh = float(phi.get('source_half_height', 0.0))
    ok_r = rad <= c['rim_radius']; ok_a = (c['bottom'] - 0.01) <= ax <= (c['top'] + hh)
    return dict(radial_m=rad, axial_m=ax, rim_radius=c['rim_radius'], axial_range=[c['bottom'] - 0.01, c['top'] + hh],
                geometry_ok=bool(ok_r and ok_a), reason=None if (ok_r and ok_a) else ('outside rim radius' if not ok_r else ('below the container bottom' if ax < c['bottom'] - 0.01 else 'above the rim')))


def eval_phi_hand(phi: dict, final: dict, init: dict, names_map: dict, extra: dict | None = None) -> dict:
    t = phi['type']
    if t not in HAND_TYPES:
        return _ORIG['eval_phi2'](phi, final, init, names_map, extra)
    extra = extra or {}
    n = names_map.get(phi['source'])
    if n is None or n not in final:
        return dict(type=t, success=False, reason='source object not matched')
    gf = extra.get('grasping_final') or {}
    released = (not gf[n]) if type(gf.get(n)) is bool else None
    quiescent = extra.get('quiescent')
    out = dict(type=t, released=released, quiescent=quiescent, goal_kind=phi.get('goal_kind'))
    if t == 'hoi4d_place':
        # missing release evidence fails closed
        p = np.asarray(final[n], float)
        if p.shape != (7,) or not np.isfinite(p).all() or np.linalg.norm(p[3:]) < 1e-12:
            return dict(type=t, success=False, reason='non-finite or invalid pose')
        err = float(np.linalg.norm(p[:3] - np.asarray(phi['center'], float)))
        rot = float(np.degrees((R.from_quat(p[[4,5,6,3]]).inv() * R.from_quat(np.asarray(phi['terminal_quat'])[[1,2,3,0]])).magnitude()))
        bottom = float(p[2] - phi['half_height'])
        resting = abs(bottom - phi['support_z']) <= phi['support_tol_m']
        geo_ok = err <= phi['radius'] and rot <= phi['orientation_tol_deg'] and resting
        success = bool(geo_ok and released is True and quiescent is True)
        out.update(pos_err_m=err, rot_err_deg=rot, bottom_above_support_m=bottom-phi['support_z'], resting=bool(resting),
                   in_region=bool(err <= phi['radius']), upright=bool(rot <= phi['orientation_tol_deg']), geometry_ok=bool(geo_ok), success=success,
                   reason='ok' if success else 'geometry' if not geo_ok else 'release evidence missing or still held' if released is not True else 'quiescence evidence missing or moving',
                   intake_predicate='place_released_resting_in_region', benchmark_settle='shared hand executor, including explicit quiescence audit')
        return out
    if t == 'in_container':
        tgt = extra.get('target_pose'); out['target_source'] = 'scene' if tgt is not None else 'gt'
        if tgt is None: tgt = phi['target_pose']
        g = in_container(phi, final[n], tgt); out.update(g); geo_ok = g['geometry_ok']
        # This is the geometric predicate, also used by diagnostics. A completed
        # entry is checked on the full record in verdict_hand, never by a fixed
        # displacement threshold (moving across the same tray is not entering it).
        gi = in_container(phi, init[n], tgt) if init and n in init else None
        out['init_geometry_ok'] = None if gi is None else bool(gi['geometry_ok'])
        # diagnostic: relative pose vs the demonstrated one (the 'seated'-style numbers), spin about the object axis ignored
        rel = np.linalg.inv(T_of(tgt)) @ T_of(final[n]); ref = np.asarray(phi['gt_rel'], float)
        out['rel_pos_m'] = float(np.linalg.norm(rel[:3, 3] - ref[:3, 3])); ax_ = phi.get('axis')
        if ax_ is None:   # no declared symmetry axis: full rotation error
            out['rel_axis_deg'] = float(np.degrees(np.linalg.norm(R.from_matrix(ref[:3, :3].T @ rel[:3, :3]).as_rotvec()))); out['rel_rot_metric'] = 'full'
        else:
            i = 'xyz'.index(ax_); out['rel_axis_deg'] = float(np.degrees(np.arccos(np.clip(float(rel[:3, i] @ ref[:3, i]), -1, 1)))); out['rel_rot_metric'] = f'symmetry axis {ax_}'
    else:   # lifted
        dz = float(final[n][2] - init[n][2]); out.update(dz_m=dz, min_dz=float(phi.get('min_dz', 0.05)))
        geo_ok = dz >= float(phi.get('min_dz', 0.05))
    rel_ok = not phi.get('require_released', True) or released is True
    q_ok = not phi.get('require_quiescent', True) or quiescent is True
    out.update(geometry_ok=bool(geo_ok), success=bool(geo_ok and rel_ok and q_ok))
    if not out['success']:
        out['reason'] = (out.get('reason') or out.get('reason_geo') or 'geometry') if not geo_ok else (('release state not recorded' if released is None else 'still held') if not rel_ok else ('quiescence not recorded' if quiescent is None else 'not quiescent'))
    else:
        out['reason'] = 'ok'
    if t == 'in_container' and extra.get('target_pose') is None:
        out.update(success=False, reason='receiving part pose not recorded in evaluated scene')
    return out


def verdict_hand(ctx: dict, phi: dict, rec: dict, name: str) -> dict:
    """`task.verdict` for the hand predicates; other predicate types go to the original."""
    if phi.get('type') not in HAND_TYPES:
        return _ORIG['verdict'](ctx, phi, rec, name)
    n = name
    if n not in rec.get('final_poses', {}) or n not in rec.get('init_poses', {}):
        return dict(type=phi['type'], success=False, geometry_ok=False, reason='source pose not recorded')
    fin = T.canon_final(ctx, rec['final_poses'], n); ini = T.canon_final(ctx, rec['init_poses'], n)
    extra = dict(grasping_final=rec.get('grasping_final'), quiescent=rec.get('quiescent'), target_pose=ctx.get('target_pose'))
    r = eval_phi_hand(phi, fin, ini, {phi['source']: n}, extra)
    r['canonical'] = dict(A_identity=ctx.get('identity'), axis_rot_deg=ctx.get('axis_rot_deg'), origin_shift_cm=ctx.get('origin_shift_cm'), target_source='scene' if ctx.get('target_pose') is not None else 'gt'); r['final_pose_canonical'] = fin[n]
    if phi['type'] == 'in_container' and not ctx.get('parts_c'):
        r.update(success=False, reason='receiving part not found')
    def geometry(pose):
        canonical = T.canon_final(ctx, {n: pose}, n)
        return bool(eval_phi_hand(phi, canonical, ini, {phi['source']: n}, extra).get('geometry_ok'))
    return T.gate_initial_goal(r, phi, rec, n, geometry)


def find_target_parts_hand(scene, phi, gt_scene, gt_pos):
    """`task.find_target_parts` minus the manipulated part of a candidate scene: the receiving-prop name rule must not pick
    the candidate's single manipulated object as the receiving prop."""
    parts = _ORIG['find_target_parts'](scene, phi, gt_scene, gt_pos)
    if phi.get('type') not in HAND_TYPES: return parts
    objs = scene.get('objects') or []; drop = set()
    if len(objs) == 1: drop.add(objs[0]['name'])
    return [p for p in parts if p['name'] not in drop]


def patch():
    """Make the shared metric layer evaluate the hand predicates (idempotent)."""
    T.eval_phi2 = eval_phi_hand; T.verdict = verdict_hand; T.find_target_parts = find_target_parts_hand
