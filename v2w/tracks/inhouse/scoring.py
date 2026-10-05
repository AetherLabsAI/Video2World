"""Metric records of executed in-house candidates (``eval.json['eval2']``).

Every record carries the task outcome from the candidate's own execution, its object trajectory error against the
accepted GT on the demonstration clock, and the geometry metrics (Scene CD, object shape / size) of its scene.
"""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from v2w import paths
from v2w.metrics import geometry as M
from v2w.metrics.record import set_scene_result
from v2w.metrics.trajectory import terminal_position

from . import gt as GT
from . import tasks
from .package import read, scene_geometry, entity_geometry

PROVIDER_SOURCES = 'sources/inhouse'
TRAJECTORY_PROTOCOL = 'demonstration-clock-trajectory/1.0'


def verify_run(sd, run):
    """A complete rigid-target execution of this sample with a consistent physics clock and physical audit."""
    ex = read(run / 'execution.json')
    if ex.get('status') != 'executed' or ex.get('smoke_test') or ex.get('sample') != Path(sd).name:
        raise ValueError('Full, matching native execution required')
    obs = read(run / 'observations.json')
    z = np.load(run / 'trajectory.npz', allow_pickle=False)
    times = np.array([o['physics_time_s'] for o in obs])
    times -= times[0]
    if len(times) != ex['frames'] or not np.allclose(times, np.arange(len(times)) * .05, atol=2e-5, rtol=0):
        raise ValueError('Physics observation clock mismatch')
    if not np.allclose(z['xyz'], [o['xyz'] for o in obs], atol=0, rtol=0) or not np.allclose(z['wxyz'], [o['wxyz'] for o in obs], atol=0, rtol=0):
        raise ValueError('Observations and native states differ')
    audit = ex['audit']
    if audit['contact_decode_errors'] or not audit['no_post_start_rigid_body_state_writes'] or not audit['no_kinematic_target']:
        raise ValueError('Physical/contact evidence invalid')
    return ex, obs, z, times


def candidate_poses(raw, ce, dest, ge, gt_dir):
    """Executed target poses in the GT model frame (fixed canonical registration of the candidate geometry)."""
    poses = M.canon_traj(raw, M.canonical_transform(ce, dest, ge, gt_dir))
    T = np.repeat(np.eye(4)[None], len(poses), axis=0)
    T[:, :3, 3] = poses[:, :3]
    T[:, :3, :3] = Rotation.from_quat(np.roll(poses[:, 3:], -1, axis=1)).as_matrix()
    return poses, T


def grip_trace(q, n, hand):
    return q[np.minimum(np.arange(n), len(q) - 1), 14 if hand == 'l' else 15]


def base_record(sd, ex, semantic, own, **extra):
    return dict(schema_version='eval2/4', sample=Path(sd).name, family=read(Path(sd) / 'meta.json')['family'], build=True,
                task_success=semantic['task_success'], progress=semantic['progress'], task_observations=semantic.get('task_observations'),
                own_trajectory=own, ape=own.get('ape'), rpe=own.get('rpe'), relative=None, physical_audit=ex['audit'], metric_status={},
                terminal_position_metric_version='terminal-position/1.1', **extra)


# ---------------------------------------------------------------- provider samples (camera-native candidates)
def provider(sd, ev, authored):
    sd, ev = Path(sd), Path(ev)
    data = paths.DATA / PROVIDER_SOURCES
    supp = data / 'rgbd_formal/gt_supplement' / sd.name
    meta = read(sd / 'meta.json')
    orientation = read(data / 'admitted' / Path(meta['original_accepted_gt']).name / 'admission.json')['orientation_metric']
    gt = dict(np.load(sd / 'hidden/object_poses.npz', allow_pickle=False))
    gs = GT.scene_doc(supp / 'gt_pkg/scene.json')
    dest = ev / 'geometry'
    dest.mkdir(exist_ok=True)
    scene = scene_geometry(authored, ev / 'pkg_baseframe', dest)
    ce = next(e for e in scene['objects'] if e['name'] == 'target')
    ge = gs['objects'][0]
    run = ev / 'native'
    ex = read(run / 'execution.json')
    obs = read(run / 'observations.json')
    z = np.load(run / 'trajectory.npz', allow_pickle=False)
    if ex.get('status') != 'executed' or ex['audit']['contact_decode_errors'] or not ex['audit']['no_post_start_rigid_body_state_writes'] or not ex['audit']['no_kinematic_target']:
        raise ValueError('Native execution evidence incomplete')
    times = np.array([o['physics_time_s'] for o in obs])
    times -= times[0]
    if not np.allclose(times, np.arange(len(times)) * .05, rtol=0, atol=2e-5):
        raise ValueError('Native physics clock mismatch')
    p, T = candidate_poses(np.c_[z['xyz'], z['wxyz']], ce, dest, ge, supp / 'gt_pkg')
    ts = np.rint(times * 1e9).astype(np.int64)
    demo = (gt['timestamp_ns'] - gt['timestamp_ns'][0]).astype(np.int64)
    # Fixed demonstration clock; after an early stop the executed endpoint is held.
    P, _ = GT.interpolate(ts, T, np.clip(demo, ts[0], ts[-1]))
    ape, rpe = GT.pose_metrics(P, gt['T_model_base_object'], orientation, time_s=demo * 1e-9)
    nt = np.rint(demo / max(int(demo[-1]), 1) * ts[-1]).astype(np.int64)
    Pn, _ = GT.interpolate(ts, T, nt)
    na, nr = GT.pose_metrics(Pn, gt['T_model_base_object'], orientation, time_s=demo * 1e-9)
    kind = read(ev / 'pkg_baseframe/protocol.json')['kind']
    contract = read(data / 'rgbd_provider/packages' / sd.name / 'task_contract.json')
    hand = 'l' if kind == 'cracker' else 'r'
    forces = np.zeros((len(obs), 2, 3))
    for i, o in enumerate(obs):
        for c in o['contacts']:
            bodies = c['body0'] + ' ' + c['body1']
            if '/World/Entities/target' not in bodies:
                continue
            for side, part in enumerate(('inner', 'outer')):
                if 'gripper_' + hand + '_' + part in bodies:
                    forces[i, side] += np.asarray(c['impulse']) * 60.
    grip = np.load(ev / 'pkg_baseframe/actions.npy')[:, 14 if kind == 'cracker' else 15]
    grip = np.clip(grip[np.minimum(np.arange(len(obs)), len(grip) - 1)], 0, 1)
    bounds = np.asarray([o['bounds'] for o in obs])
    trace = dict(pad_force_vectors=forces, obj_xyz=z['xyz'], obj_quat=z['wxyz'], fabric_obj_xyz=z['xyz'], usd_obj_wxyz=z['wxyz'], grip=grip,
                 bbox_center_xyz=bounds.mean(1), support_z=bounds[:, 0, 2])
    receiver = next(e for e in scene['props'] if e['name'] == 'receiver') if kind in ('book', 'box') else None
    if receiver is not None:
        contract = dict(contract, destination_zone=tasks.surface_height(receiver, dest))
    task = tasks.pick_or_place(trace, contract)
    own = dict(**task, ape=ape, rpe=rpe, protocol=TRAJECTORY_PROTOCOL, n_frames=len(demo), dt=.05, execution_dt=.05, execution_frames=len(p),
               coverage=float(np.mean(demo <= ts[-1])), held_frames=int(np.sum(demo > ts[-1])), duration_s=float(demo[-1] * 1e-9), execution_duration_s=float(times[-1]),
               duration_normalized=dict(protocol='duration-normalized-trajectory/1.0', ape=na, rpe=nr, time_scale=float(demo[-1] * 1e-9 / max(times[-1], 1e-9))))
    r = base_record(sd, ex, task, own, orientation_support=orientation)
    set_scene_result(r, M.scene_cd_v5(scene, dest, gs, supp / 'gt_pkg', read(supp / 'camera.json'), read(supp / 'roles.json')))
    r['object'] = M.object_level(ce, dest, p[0], ge, supp / 'gt_pkg', GT.poses7(gt['T_model_base_object'][:1])[0])
    if receiver is not None:
        target_gt = read(supp / 'receiver.json')
        r['relative'] = terminal_position(dict(target='receiver', gt_rel=target_gt['gt_terminal_relative'], axis=None if kind == 'book' else 'z'),
                                          p[0], p[-1], GT.poses7(gt['T_model_base_object']), receiver['pos'] + receiver['quat'])
    return r


# ---------------------------------------------------------------- episodes with accepted pose GT
def episode(sd, pkg, run, out):
    sd, pkg, run, out = map(Path, (sd, pkg, run, out))
    ex, obs, z, times = verify_run(sd, run)
    gt = GT.pose_sequence(sd)
    supp, dest = sd / 'scene_supplement', out.parent / 'geometry'
    dest.mkdir(exist_ok=True)
    authored = read(pkg / 'scene.json')
    scene = scene_geometry(authored, pkg, dest)
    gs = GT.scene_doc(supp / 'gt_pkg/scene.json')
    ce, ge = next(e for e in scene['objects'] if e['name'] == 'target'), gs['objects'][0]
    raw = np.c_[z['xyz'], z['wxyz']]
    poses, T = candidate_poses(raw, ce, dest, ge, supp / 'gt_pkg')
    demo = (gt['timestamp_ns'] - gt['timestamp_ns'][0]).astype(np.int64)
    ts = np.rint(times * 1e9).astype(np.int64)
    G = gt['primary__T']
    P, _ = GT.interpolate(ts, T, np.clip(demo, ts[0], ts[-1]))
    ape, rpe = GT.pose_metrics(P, G, 'axis_direction_only')
    Pn, _ = GT.interpolate(ts, T, np.rint(demo / max(int(demo[-1]), 1) * ts[-1]).astype(np.int64))
    na, nr = GT.pose_metrics(Pn, G, 'axis_direction_only')
    contract = tasks.validate_episode(read(sd / 'task_contract.json'))
    q = np.load(pkg / 'actions.npy')
    receiver = None
    if contract['kind'] == 'initially_held_place':
        receiver = next(e for e in scene['props'] if e['name'] == 'receiver')
        contract = dict(contract, destination_zone=tasks.surface_height(receiver, dest))
    semantic = tasks.episode_outcome(obs, z, grip_trace(q, len(obs), contract['hand']), contract)
    own = dict(**semantic, ape=ape, rpe=rpe, protocol=TRAJECTORY_PROTOCOL, n_frames=len(G), dt=.05, execution_frames=len(poses), coverage=float(np.mean(demo <= ts[-1])),
               held_frames=int(np.sum(demo > ts[-1])), duration_s=float(demo[-1] * 1e-9), execution_duration_s=float(times[-1]),
               duration_normalized=dict(protocol='duration-normalized-trajectory/1.0', ape=na, rpe=nr, time_scale=float(demo[-1] * 1e-9 / max(times[-1], 1e-9))))
    result = base_record(sd, ex, semantic, own, task_contract=contract, orientation_support='axis_direction_only', gt_frames=len(G), scored_frames=len(G))
    result['object'] = M.object_level(ce, dest, raw[0], ge, supp / 'gt_pkg', GT.poses7(G[:1])[0])
    set_scene_result(result, M.scene_cd_v5(scene, dest, gs, supp / 'gt_pkg', read(supp / 'camera.json'), read(supp / 'roles.json')))
    if receiver is not None:
        R = np.asarray(read(supp / 'receiver.json')['T_base_receiver'])
        C = tasks.receiver_frame(receiver, dest)
        result['relative'] = terminal_position(dict(target='receiver', gt_rel=(np.linalg.inv(R) @ G[-1]).tolist(), axis='z'), poses[0], poses[-1], GT.poses7(G), GT.poses7(C[None])[0])
    np.savez_compressed(out.parent / 'per_frame.npz', timestamp_ns=gt['timestamp_ns'], T_gt=G, T_candidate=P, source_valid=gt['primary__valid'], scored=np.ones(len(G), bool))
    return result


def masked_episode(sd, pkg, run, out, kind):
    """Reviewed and fryer samples: accepted poses on the full clock, unknown frames excluded without filling."""
    sd, pkg, run, out = map(Path, (sd, pkg, run, out))
    gt, mask = GT.masked_pose_sequence(sd)
    ex, obs, z, times = verify_run(sd, run)
    c = read(sd / 'task_contract.json')
    q = np.load(pkg / 'actions.npy')
    tail = 2 if c['kind'] == 'pick_and_hold' else 40
    if len(q) != len(mask) or ex['frames'] != len(q) + tail:
        raise ValueError('Truncated or extended execution clock')
    dest = out.parent / 'geometry'
    dest.mkdir(exist_ok=True)
    authored = read(pkg / 'scene.json')
    scene = scene_geometry(authored, pkg, dest)
    supp = sd / 'scene_supplement'
    gs = GT.scene_doc(supp / 'gt_pkg/scene.json')
    ce, ge = next(e for e in scene['objects'] if e['name'] == 'target'), gs['objects'][0]
    raw = np.c_[z['xyz'], z['wxyz']]
    poses, T = candidate_poses(raw, ce, dest, ge, supp / 'gt_pkg')
    demo = (gt['timestamp_ns'] - gt['timestamp_ns'][0]).astype(np.int64)
    P, coverage = GT.interpolate(np.rint(times * 1e9).astype(np.int64), T, demo)
    if not coverage.all():
        raise ValueError('Native trace does not cover the demonstration clock')
    G = gt['primary__T']
    ape, rpe = GT.pose_metrics(P[mask], G[mask], 'axis_direction_only', time_s=demo[mask] * 1e-9)
    grip = grip_trace(q, len(obs), c['hand'])
    if kind == 'fryer':
        c = tasks.fryer_aperture(c, scene, dest)
        semantic = tasks.fryer_outcome(obs, z, grip, c, np.asarray(M.entity_mesh(ce, dest).vertices))
    elif c['kind'] == 'pick_and_hold':
        semantic = tasks.episode_outcome(obs, z, grip, c)
    else:
        tasks.validate_episode(dict(c, schema='inhouse-episode-task/1', kind='pick_and_hold'))
        forces = np.linalg.norm(tasks.contact_forces(obs, c['hand']), axis=2)
        verts = np.asarray(M.entity_mesh(ce, dest).vertices)
        rr = Rotation.from_quat(np.roll(z['wxyz'], -1, axis=1)).as_matrix()
        v = np.einsum('tij,vj->tvi', rr, verts) + z['xyz'][:, None, :]
        receiver = next(e for e in authored['props'] if e['name'] == 'receiver')
        semantic = tasks.bag_candidate_region((v.min(1) + v.max(1)) / 2, receiver, pkg, grip, forces)
    own = dict(**semantic, ape=ape, rpe=rpe, protocol=TRAJECTORY_PROTOCOL, n_frames=int(mask.sum()), timeline_frames=len(mask), scored_frame_indices=np.flatnonzero(mask).tolist(),
               coverage=1., gt_observation_coverage=float(mask.mean()), dt=.05, execution_frames=len(poses), duration_s=float(demo[-1] * 1e-9), execution_duration_s=float(times[-1]))
    result = base_record(sd, ex, semantic, own, task_contract=c, orientation_support='axis_direction_only', gt_frames=len(G), scored_frames=int(mask.sum()),
                         unknown_gt_frames=np.flatnonzero(~mask).tolist())
    result['object'] = M.object_level(ce, dest, raw[0], ge, supp / 'gt_pkg', GT.poses7(G[:1])[0])
    set_scene_result(result, M.scene_cd_v5(scene, dest, gs, supp / 'gt_pkg', read(supp / 'camera.json'), read(supp / 'roles.json')))
    np.savez_compressed(out.parent / 'per_frame.npz', timestamp_ns=gt['timestamp_ns'], T_gt=G, T_candidate=P, source_valid=gt['primary__valid'], scored=mask)
    return result


# ---------------------------------------------------------------- point tasks
def point(sd, pkg, run, out):
    import trimesh
    sd, pkg, run, out = map(Path, (sd, pkg, run, out))
    gt, b = GT.point_task_gt(sd)
    ex, obs, z, times = verify_run(sd, run)
    c = tasks.validate_episode(read(sd / 'task_contract.json'))
    q = np.load(pkg / 'actions.npy')
    tail = 40 if c['kind'] == 'initially_held_place' else 2
    if len(q) != gt['frames'] or len(obs) != len(q) + tail or not ex['audit']['complete']:
        raise ValueError('Incomplete full-clock native execution')
    n = gt['frames']
    xyz, wxyz = np.asarray(z['xyz'])[:n], np.asarray(z['wxyz'])[:n]
    authored = read(pkg / 'scene.json')
    dest = out.parent / 'geometry'
    dest.mkdir(exist_ok=True)
    ce = entity_geometry(next(e for e in authored['objects'] if e['name'] == 'target'), pkg, dest)
    mesh = M.entity_mesh(ce, dest)
    if b['point_kind'] == 'registered_model_origin':
        ref = sd / 'hidden/registered_model.obj'
        gmesh = trimesh.load(ref, force='mesh', process=False)
        if mesh.vertices.shape == gmesh.vertices.shape and np.allclose(mesh.vertices, gmesh.vertices, atol=1e-10, rtol=0) and np.array_equal(mesh.faces, gmesh.faces):
            A = np.eye(4)
        else:
            A = M.canonical_transform(ce, dest, dict(kind='mesh', mesh_path=str(ref), pos=[0, 0, 0], quat=[1, 0, 0, 0]), sd / 'hidden')
        local = np.linalg.inv(A)[:3, 3]
        pred = xyz + Rotation.from_quat(np.roll(wxyz, -1, axis=1)).apply(np.broadcast_to(local, xyz.shape).copy())
        coverage, hits = np.ones(n), np.zeros(n, np.int64)
    else:
        with np.load(sd / 'hidden/surface_observation.npz', allow_pickle=False) as support:
            pred, coverage, hits = GT.observed_surface(mesh, xyz, wxyz, support)
    metrics = GT.point_metrics(gt['xyz'], pred, coverage)
    arrays = metrics.pop('arrays')
    if c['kind'] == 'initially_held_place':
        receiver = next(e for e in authored['props'] if e['name'] == 'receiver')
        c = dict(c, destination_zone=tasks.surface_height(entity_geometry(receiver, pkg, dest), dest))
    task = tasks.episode_outcome(obs, z, grip_trace(q, len(obs), c['hand']), c)
    result = dict(schema_version='eval2/4', sample=sd.name, family=read(sd / 'meta.json')['family'], build=True, **task, **metrics, metric_status={},
                  gt_frames=n, scored_frames=n, point_definition=gt['definition'], coordinate_frame='source_model_robot_base', execution_frames=len(obs),
                  execution_duration_s=float(times[-1]), physical_audit=ex['audit'])
    obj = GT.object_model_metrics(ce, dest, list(xyz[0]) + list(wxyz[0]), sd, out.parent / 'object_gt_geometry')
    if obj is not None:
        result['object'] = obj
    GT.attach_visible_scene(result, sd, pkg, out.parent / 'visible_scene_geometry')
    GT.attach_rotation(result, sd, times, out.parent / 'observable_rotation.npz', vertices=mesh.vertices, faces=mesh.faces, wxyz=z['wxyz'])
    np.savez_compressed(out.parent / 'per_frame.npz', timestamp_ns=gt['timestamp_ns'], point_gt=gt['xyz'], point_candidate=pred, target_xyz=xyz, target_wxyz=wxyz, surface_hit_count=hits, **arrays)
    return result


# ---------------------------------------------------------------- wallet tasks
def wallet(sd, pkg, run, out):
    import trimesh
    sd, pkg, run, out = map(Path, (sd, pkg, run, out))
    ex = read(run / 'execution.json')
    if ex.get('status') != 'executed' or ex.get('smoke_test') or ex.get('physics_steps', 0) <= 0 or ex['sample'] != sd.name:
        raise ValueError('Full native physical execution of this sample required')
    audit = ex['audit']
    if not audit['complete'] or not audit['no_post_start_object_state_writes'] or not audit['no_object_attachment'] or audit['critical_property_mutations']:
        raise ValueError('Invalid execution audit')
    z = np.load(run / 'trajectory.npz', allow_pickle=False)
    if len(z['video_time_s']) != ex['frames'] or not np.allclose(z['video_time_s'], np.arange(ex['frames']) * .05, atol=1e-9, rtol=0):
        raise ValueError('Physical clock changed')
    contract = read(sd / 'task_contract.json')
    # Holding evidence: target surface to fingertip pad distance in the pad frame, and the surface centre in the gripper.
    v, f, hand = z['vertices_base'], z['faces'], contract['hand']
    centers = np.array([GT.surface_center(x, f) for x in v])
    start = 0 if hand == 'l' else 2
    pad_poses = z['pad_pose_xyzw'][:, start:start + 2]
    dist = np.zeros((len(v), 2))
    pads = np.load(run / 'pad_geometry.npz')
    for k, side in enumerate(['inner', 'outer']):
        key = f'gripper_{hand}_{side}_link4'
        m = trimesh.Trimesh(pads[key + '_vertices'], pads[key + '_faces'], process=False)
        for i in range(len(v)):
            local = (v[i] - pad_poses[i, k, :3]) @ Rotation.from_quat(pad_poses[i, k, 3:]).as_matrix()
            bound = np.linalg.norm(np.maximum(np.maximum(m.bounds[0] - local, local - m.bounds[1]), 0), axis=1)
            near = bound <= contract['proximity_m']
            # A point cannot be nearer than its AABB lower bound; distant hands skip the exact query.
            dist[i, k] = float(trimesh.proximity.closest_point(m, local[near])[1].min()) if near.any() else float(bound.min())
    rotation = Rotation.from_quat(pad_poses[:, 0, 3:]).as_matrix()
    actions = np.load(pkg / 'actions.npy')
    features = dict(center=centers, center_in_gripper=np.einsum('tji,tj->ti', rotation, centers - pad_poses[:, :, :3].mean(1)), pad_distances_m=dist,
                    grip=actions[np.minimum(np.arange(len(v)), len(actions) - 1), 18 if hand == 'l' else 19], bottom_z=v[:, :, 2].min(1),
                    surface_speed_m_s=np.linalg.norm(z['velocities_base'], axis=2).max(1))
    receiver = None
    if contract['kind'] == 'initially_held_insert':
        receiver = next(e for e in read(pkg / 'scene.json')['props'] if e['name'] == 'receiver')
        features['inside'] = tasks.inside_cavity(v, receiver, pkg, contract['containment_tolerance_m'])
    semantic = tasks.wallet_outcome(features, z, pkg, contract, receiver)
    gt = GT.surface_sequence(sd)
    geometry = GT.surface_metrics(gt, z)
    r = dict(schema_version='eval2/4', sample=sd.name, family=read(sd / 'meta.json')['family'], build=True, **geometry, **semantic,
             track='robot_execution', task_contract=contract, physical_audit=audit, metric_status={})
    r['deformable']['initial_surface_cd_cm'] = geometry['per_frame'][0]['surface_cd_cm']
    if gt['valid'][-1]:
        r['deformable']['terminal_surface_cd_cm'] = geometry['per_frame'][-1]['surface_cd_cm']
    GT.attach_visible_scene(r, sd, pkg, out.parent / 'visible_scene_geometry')
    GT.attach_rotation(r, sd, z['video_time_s'], out.parent / 'observable_rotation.npz', surfaces=z['vertices_base'], faces=z['faces'])
    np.savez_compressed(out.parent / 'task_features.npz', **features)
    return r
