"""Metric record of one rigid-object configuration, built from the evaluator's artifacts (no simulation here).

Inputs: the sample directory (hidden GT), the evaluated package in the robot base frame, the scene json (build, Scene
CD inputs) and the rollout json (the package's own actions: task success, progress, trajectory and terminal metrics).
Failures stay explicit in ``metric_status``; a missing artifact is never turned into a measured value.
"""
from __future__ import annotations

import math
from numbers import Real
from pathlib import Path

import numpy as np

from v2w.metrics import geometry as M
from v2w.metrics import task as T
from v2w.metrics.task import jload, gt_scene_objects, gt_clock, cand_scene, gt_target

SCHEMA_VERSION = 'eval2/4'
SCENE_VERSION = 'scene_cd_v5'
SCENE_REVISION = 'scene_cd_v5/rev8-assembly-object-alignment'


def metric_error(record, keys, reason):
    for key in keys:
        record.setdefault('metric_status', {})[key] = {'status': 'error', 'reason': str(reason)}
        if '.' not in key:
            record[key] = None


def set_scene_result(record, result):
    record['schema_version'] = SCHEMA_VERSION
    record['scene_metric_version'] = SCENE_VERSION
    record['scene_metric_revision'] = result.get('revision')
    record[SCENE_VERSION] = result
    value = result.get('scene_cd_cm')
    if result.get('error') or result.get('metric_valid') is False or not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value):
        metric_error(record, ('scene_chamfer_cm', 'scene_missing_fraction', 'scene_extra_fraction', 'table_depth_err_cm'),
                     result.get('error') or result.get('invalid_reason') or 'non-finite scene metric')
        return
    record['scene_chamfer_cm'] = float(value)
    record.setdefault('metric_status', {})['scene_chamfer_cm'] = {'status': 'ok', 'version': SCENE_VERSION}
    for target, source in (('table_depth_err_cm', 'table_depth_err_cm'), ('scene_missing_fraction', 'missing_fraction'),
                           ('scene_extra_fraction', 'extra_fraction')):
        record[target] = result.get(source)
    status = record.setdefault('metric_status', {})
    for target in ('scene_chamfer_cm', 'scene_missing_fraction', 'scene_extra_fraction'):
        value = record.get(target)
        if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value):
            metric_error(record, (target,), 'missing or non-finite scene submetric')
        else:
            status[target] = {'status': 'ok', 'version': SCENE_VERSION, 'revision': result.get('revision')}
            if result.get('penalty'):
                status[target]['reason'] = result['penalty']
    if result.get('extra_rule'):
        status['scene_extra_fraction']['reason'] = result['extra_rule']
    table = result.get('table') or {}
    if table.get('status') == 'na':
        status['table_depth_err_cm'] = {'status': 'na', 'reason': table['reason']}
    elif record.get('table_depth_err_cm') is not None:
        status['table_depth_err_cm'] = {'status': 'ok', 'revision': result.get('revision')}
        if table.get('penalty'):
            status['table_depth_err_cm']['reason'] = table['penalty']
    alignment = result.get('alignment') or {}
    for key, source in (('scene_align_cm', 'trans_cm'), ('scene_align_scale', 'scale'), ('scene_align_rot_deg', 'rot_deg')):
        record[key] = alignment.get(source)
        if alignment.get('status') == 'na':
            status[key] = {'status': 'na', 'reason': alignment['reason']}
    record['scene_cd_unaligned_cm'] = result.get('scene_cd_unaligned_cm', record['scene_chamfer_cm'])


def hand_validity(out, rollout, required=False):
    """Gate task success and progress on the hand-track physical execution contract."""
    rollout = rollout or {}
    is_hand = str(rollout.get('physics_protocol', '')).startswith('hand-contact-physics/') or rollout.get('embodiment') in ('arm', 'dexhand')
    if not required and not is_hand:
        return out
    from v2w.tracks.hand.physics import gate_success
    computed_success = bool(out.get('task_success'))
    out['task_success_raw'] = computed_success if 'phi' in out else bool(rollout.get('task_success_raw', computed_success))
    out['task_success'], out['physical_validity'] = gate_success(computed_success, rollout)
    if not out['physical_validity']['valid']:
        if 'phi' in out:
            out['phi_raw'] = out['phi']; out['phi'] = dict(out['phi'], success=False, reason='invalid physical execution')
        if out.get('progress'):
            out['progress_raw'] = out['progress']; out['progress'] = dict(out['progress'], progress=0., physical_execution_valid=False)
    return out


def _gt_scene_initial(gt_scene, gt_traj):
    return {'support': gt_scene.get('support'), 'props': gt_scene['props'],
            'objects': [dict(o, pos=list(gt_traj[o['name']][0][:3]), quat=list(gt_traj[o['name']][0][3:])) for o in gt_scene['objects']]}


def rigid_record(sd: Path, pkg: Path, scene_j, rollout_j, kind_hint=None, require_hand_physics=False) -> dict:
    """Record of a rigid-object configuration: build, Scene CD, object geometry, task success and progress from the
    package's own actions, demonstration-clock trajectory errors and the terminal position error."""
    sd, pkg = Path(sd), Path(pkg)
    out = dict(sample=sd.name, package=str(pkg), schema_version=SCHEMA_VERSION)
    phi = T.clip_goal(sd.name, jload(sd / 'hidden/phi.json')); gt_scene, gt_traj = gt_scene_objects(sd); gname = phi['source']
    ge = next(o for o in gt_scene['objects'] if o['name'] == gname); sym = ge.get('symmetry'); axis = ge.get('symmetry_axis', 'z')
    sc = jload(scene_j) if scene_j else None; ro = jload(rollout_j) if rollout_j else None
    evidence = sc if sc is not None else ro
    out['build'] = evidence.get('build') if evidence and type(evidence.get('build')) is bool else None
    if out['build'] is None:
        metric_error(out, ('build',), 'missing or invalid build evidence')
    out['scene_chamfer_cm'] = None
    if out['build'] is False:
        out.update(task_success=False, scene_metric_version=SCENE_VERSION)
        return out
    if out['build']:
        try:
            scene, _ = cand_scene(pkg); cam = jload(sd / 'hidden/camera.json'); tg_, _ = gt_target(gt_scene, phi, gt_traj)
            g_scene = _gt_scene_initial(gt_scene, gt_traj)
            own_name = (sc or {}).get('object_match', {}).get(gname) or gname
            own_entity = next((e for e in scene['objects'] if e['name'] == own_name), None) or (scene['objects'][0] if scene['objects'] else None)
            if own_entity is not None:
                scene, out['scene_scope'] = T.scoped_scene(scene, pkg, g_scene, sd, phi, own_entity['name'])
            scene_result = M.scene_cd_v5(scene, pkg, g_scene, sd / 'gt_pkg', cam, roles=dict(object=gname, target=(tg_ or {}).get('name') or phi.get('target')))
        except Exception as e:
            scene_result = dict(error=f'{type(e).__name__}: {e}')
        if scene_result.get('metric_valid'):
            scene_result['scope'] = out.get('scene_scope', {})
        set_scene_result(out, scene_result)
    tgt_pose_scene = None; candidate_receiver_missing = False
    tgt_g, tg_pose = gt_target(gt_scene, phi, gt_traj)
    # object-level geometry at the initial pose
    try:
        scene, m = cand_scene(pkg); cname = (sc or {}).get('object_match', {}).get(gname) or gname
        ce = next((o for o in scene['objects'] if o['name'] == cname), None) or (scene['objects'][0] if scene['objects'] else None)
        gpos = gt_traj[gname][0]
        if ce is not None:
            cpose = np.r_[ce['pos'], ce.get('quat', (1, 0, 0, 0))]
            out['object'] = M.object_level(ce, pkg, cpose, ge, sd / 'gt_pkg', gpos)
            tgt_c = T.find_target(scene, phi, gt_scene, gpos)
            receiver_assembly = None
            if sd.name in T.DROID_RULES:
                rule = T.DROID_RULES[sd.name]
                match_phi = dict(phi, target=rule.get('target'), target_match=rule.get('target_match'))
                tgt_c, _, receiver_assembly = T.task_receiver(scene, pkg, gt_scene, sd, match_phi, ce['name'])
            # only a loaded scene and completed matching certify a missing receiver; parser failures stay metric errors
            candidate_receiver_missing = tgt_c is None and tgt_g is not None
            if phi.get('require_lid'):   # a container with a lid must be delivered closed
                out['lid'] = T.lid_ok(receiver_assembly or tgt_c, pkg, phi['require_lid']) if tgt_c is not None else dict(ok=False, reason='no receiving object', min_coverage=phi['require_lid'].get('min_coverage', 0.5))
            if tgt_c is not None and tgt_g is not None:
                tc_pose = np.r_[tgt_c['pos'], tgt_c.get('quat', (1, 0, 0, 0))]; tgt_pose_scene = tc_pose
                if tgt_g['name'] != 'socket':
                    out['target'] = dict(name_cand=tgt_c['name'], name_gt=tgt_g['name'], **M.object_level(tgt_c, pkg, tc_pose, tgt_g, sd / 'gt_pkg', tg_pose))
                else:
                    out['target'] = dict(name_cand=tgt_c['name'], name_gt='socket', center_err_cm=100 * float(np.linalg.norm(tc_pose[:3] - tg_pose[:3])))
                out['relative_initial'] = M.relative_pose_err(cpose, tc_pose, gpos, tg_pose, sym, axis)
    except Exception as e:
        out['object_error'] = f'{type(e).__name__}: {e}'
        metric_error(out, ('object.shape_cd_cm', 'object.center_err_cm', 'object.size_err_cm'), out['object_error'])
    # canonicalize the candidate part / target into the GT mesh conventions; identity for the GT package
    try:
        ctx = T.context(sd, pkg, scene_j)
    except Exception as e:
        out['canonical'] = dict(error=f'{type(e).__name__}: {e}', identity=None)
        metric_error(out, ('task_success', 'relative', 'progress'), 'canonicalization failed: ' + str(e))
        return out
    out['canonical'] = dict(identity=ctx['identity'], axis_rot_deg=ctx['axis_rot_deg'], origin_shift_cm=ctx['origin_shift_cm'], target=ctx.get('target_name'), error=ctx.get('error'))
    if tgt_pose_scene is not None and ctx['target_pose'] is not None:
        tgt_pose_scene = np.asarray(ctx['target_pose'], float)
    extra_base = dict(target_pose=(tgt_pose_scene.tolist() if tgt_pose_scene is not None else None))
    canon = lambda poses, n: M.canon_pose7(poses[n], ctx['A_obj']).tolist()
    # the package's own actions: success, progress, trajectory
    if ro and ro.get('traj'):
        n = T.trajectory_object(ro, ctx, sc, gname); out['own_object_name'] = n
        r = T.verdict(ctx, phi, ro, n) if 'ce' in ctx else T.eval_phi2(phi, {n: canon(ro['final_poses'], n)}, {n: canon(ro['init_poses'], n)}, {gname: n},
                                                                        dict(extra_base, grasping_final=ro.get('grasping_final'), quiescent=ro.get('quiescent')))
        if out.get('lid') is not None and not out['lid']['ok'] and r.get('success'):
            r['success'] = False; r['reason'] = 'no lid'
        out['task_success'] = bool(r['success']); out['phi'] = r
        kind = T.rigid_kind(phi, kind_hint)
        goal = phi.get('goal') or (phi.get('center') if phi.get('type') in ('in_region', 'in_region_upright') else (gt_traj[phi['target']][-1][:3].tolist() if phi.get('type') == 'on_top' and phi.get('target') in gt_traj else None))
        if sd.name in T.DROID_RULES and T.DROID_RULES[sd.name]['kind'] in ('move_direction', 'side_of', 'remove_to_support', 'upright_supported'):
            goal = None
        try:
            own_dt = float((cand_scene(pkg)[1].get('actions') or {})['dt'])
            out['progress'] = dict(source='own_actions', **M.task_progress(kind, dict(traj=M.canon_traj(ro['traj'][n], ctx['A_obj']).tolist(), attach_events=ro.get('attach_events'), release_events=ro.get('release_events'),
                                                                               dt=own_dt, phi=phi, phi_success=bool(r['success']), goal_xy=goal,
                                                                               goal_z=(goal[2] if goal and len(goal) > 2 else None), goal_radius=phi.get('radius'))))
        except (ValueError, KeyError, TypeError) as exc:
            metric_error(out, ('progress.progress',), str(exc))
        from v2w.metrics.trajectory import demo_clock_metrics
        try:
            own_dt = float((cand_scene(pkg)[1].get('actions') or {})['dt'])
            out['own_trajectory'] = dict(source='own_actions', **demo_clock_metrics(M.canon_traj(ro['traj'][n], ctx['A_obj']), own_dt, gt_traj[gname], gt_clock(sd), symmetry=sym, axis=axis))
        except Exception as e:
            out['own_trajectory'] = dict(source='own_actions', error=f'{type(e).__name__}: {e}')
            metric_error(out, ('own_trajectory.ape.trans_cm', 'own_trajectory.ape.rot_deg', 'own_trajectory.rpe.trans_rpe_cm'), out['own_trajectory']['error'])
    from v2w.metrics.trajectory import record_terminal_position
    own_name = ctx.get('cname') or ((sc or {}).get('object_match', {}).get(gname)) or gname
    gt_terminal_target = gt_traj[tgt_g['name']][-1] if tgt_g is not None and tgt_g['name'] in gt_traj else tg_pose
    record_terminal_position(out, phi, ro, own_name, gt_traj[gname], ctx['A_obj'], tgt_pose_scene, gt_terminal_target, axis,
                             candidate_receiver_missing=candidate_receiver_missing)
    if ro and ctx.get('semantic_axis'):
        T.apply_metrics(out, ctx, ro, own_name, gt_traj[gname], float((cand_scene(pkg)[1]['actions'])['dt']), gt_clock(sd))
    return hand_validity(out, ro, required=require_hand_physics)


def rigid_metrics(family, sample_dir, eval_dir, pkg_name='pkg'):
    """Record from a rigid evaluator's output directory: <pkg>_baseframe, scene_<pkg>.json, rollout_<pkg>.json."""
    ev = Path(eval_dir)
    return rigid_record(Path(sample_dir), ev / f'{pkg_name}_baseframe', ev / f'scene_{pkg_name}.json', ev / f'rollout_{pkg_name}.json',
                        kind_hint=family.get('eval2_kind'), require_hand_physics=family.get('evaluator') in ('ego_arm', 'ego_dexhand', 'ego_bimanual'))
