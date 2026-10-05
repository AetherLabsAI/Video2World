"""Physics audit, task predicates and deformable metrics for the reconstructed-twin tracks (rope routing, toy packing).

All quantities are in the sample's reference frame (metres), obtained from the candidate's first-camera frame through
the GT ``reference_from_camera`` transform. Success is judged on the candidate's own receiver: the clip aperture or box
cavity is derived from the candidate's colliding box geometry, never from the reference.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

AUDIT_PROTOCOL = 'native-twin-reference/1'


def cd(a, b):
    """Symmetric Chamfer distance in cm: sum of the two mean nearest-neighbour distances."""
    a, b = np.asarray(a), np.asarray(b)
    if min(len(a), len(b)) == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('empty/nonfinite metric cloud')
    return float(100 * (cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean()))


def centerline(points, n=40):
    """Centerline of a rope point cloud: means of n bins along its principal axis."""
    c = points.mean(0)
    _, _, vt = np.linalg.svd(points - c)
    t = (points - c) @ vt[0]
    edges = np.linspace(t.min(), t.max(), n + 1)
    return np.array([points[(t >= edges[k]) & (t <= edges[k + 1])].mean(0)
                     for k in range(n) if ((t >= edges[k]) & (t <= edges[k + 1])).any()])


def transform(points, pose):
    pose = np.asarray(pose)
    return np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]


def rope_through_aperture(spec, nodes, edges):
    """True when a connected rope path inside the clip aperture crosses both of its y faces."""
    if nodes is None or edges is None or len(edges) == 0:
        return False, {'reason': 'No physical cable/flex connectivity in object'}
    nodes = transform(nodes, np.linalg.inv(spec['clip_pose']))
    bounds = np.asarray(spec['aperture_bounds'])
    parent = list(range(len(nodes)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    hits = []
    for i, j in np.asarray(edges, int):
        a, b = nodes[i], nodes[j]
        delta = b - a
        lo, hi = 0., 1.
        valid = True
        for axis in range(3):
            if abs(delta[axis]) < 1e-12:
                if not bounds[0, axis] <= a[axis] <= bounds[1, axis]:
                    valid = False
                    break
            else:
                t = sorted((bounds[:, axis] - a[axis]) / delta[axis])
                lo, hi = max(lo, t[0]), min(hi, t[1])
        if not valid or lo > hi:
            continue
        # clipped segments join only at shared original vertices inside the aperture
        i_inside = np.all((a >= bounds[0] - 1e-8) & (a <= bounds[1] + 1e-8))
        j_inside = np.all((b >= bounds[0] - 1e-8) & (b <= bounds[1] + 1e-8))
        edge_node = len(parent)
        parent.append(edge_node)
        if i_inside:
            parent[root(i)] = root(edge_node)
        if j_inside:
            parent[root(j)] = root(edge_node)
        ends = np.array([a + lo * delta, a + hi * delta])
        hits.append((edge_node, [bool(np.any(np.abs(ends[:, 1] - face) < 1e-7)) for face in bounds[:, 1]]))
    components = {}
    for node, pair in hits:
        r = root(node)
        components[r] = np.logical_or(components.get(r, [False, False]), pair)
    ok = any(np.all(v) for v in components.values())
    return bool(ok), {'crosses_both_aperture_faces': bool(ok)}


def toy_in_box(points, bounds, tolerance=.002):
    """Toy footprint inside the box cavity (xy, bounds x 1.05) with its centroid between bottom and rim."""
    p, b = np.asarray(points), np.asarray(bounds)
    c, half = b.mean(0), (b[1] - b[0]) / 2
    inside = np.all(np.abs(p[:, :2] - c[:2]) <= half[:2] * 1.05, axis=1)
    xy = float(np.mean(inside))
    z = float(p[:, 2].mean())
    below = float(p[inside, 2].min()) if inside.any() else float(p[:, 2].min())
    ok = xy >= .85 and b[0, 2] <= z <= b[1, 2] and below >= b[0, 2] - tolerance
    return bool(ok), dict(inside_xy_fraction=xy, centroid_z_m=z, lowest_surface_z_m=below, bottom_m=float(b[0, 2]),
                          rim_m=float(b[1, 2]), penetration_tolerance_m=tolerance)


def stable_result(checks, times):
    """Success: unsolved at the start and satisfied over the whole final 0.5 s."""
    times = np.asarray(times, float)
    if len(times) != len(checks) or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('invalid execution clock')
    start = max(0, int(np.searchsorted(times, times[-1] - .5, side='right') - 1))   # sample at/before the window start
    return bool(not checks[0] and times[-1] - times[start] >= .5 - 1e-6 and all(checks[start:]))


def trajectory(sim, sim_time, gt, gt_time, prefix=0.):
    """Surface Chamfer along the demonstration clock: each GT frame against the nearest executed state."""
    st = np.asarray(sim_time, float) - prefix
    gt_time = np.asarray(gt_time, float)
    if len(sim) != len(st) or len(gt) != len(gt_time) or len(st) < 2 or np.any(np.diff(st) <= 0) or np.any(np.diff(gt_time) <= 0):
        raise ValueError('invalid trajectory clock')
    query = np.clip(gt_time, st[0], st[-1])
    hi = np.searchsorted(st, query).clip(1, len(st) - 1)
    lo = hi - 1
    take = np.where(query - st[lo] <= st[hi] - query, lo, hi)
    values = [cd(sim[i], g) for i, g in zip(take, gt)]
    return dict(protocol='deformable-demonstration-clock/1', shape_cd_cm=float(np.mean(values)), per_frame_cm=values,
                time_s=gt_time.tolist(), executed_indices=take.tolist(),
                coverage=float(np.mean((gt_time >= st[0]) & (gt_time <= st[-1]))), prefix_hold_s=prefix,
                short_execution='nearest actual state; final state held after execution ends',
                unscored_tail_s=float(max(0, st[-1] - gt_time[-1])))


def audit_model(pkg, sd, task):
    """Check the candidate model's physics and derive its receiver cavity (task frame) from its box collision geometry."""
    import mujoco
    from video2sim.native_twin import load_package
    manifest, m, controls, dt, obj, robot, target, flex = load_package(pkg)
    if not robot or not target:
        raise ValueError('physical robot and receiver roles required')
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    definition = json.loads((sd / 'hidden/metrics_v1/definition.json').read_text())
    T = np.asarray(definition['reference_from_camera'])
    gravity = T[:3, :3] @ m.opt.gravity
    if not np.allclose(gravity, [0, 0, -9.81], atol=.02):
        raise ValueError('gravity must be Earth gravity in the declared camera frame')
    if int(m.opt.disableflags):
        raise ValueError('physics disable flags unsupported')

    def geoms(bodies):
        return [i for i in range(m.ngeom) if m.geom_bodyid[i] in bodies and (m.geom_contype[i] or m.geom_conaffinity[i])]
    og, rg, tg = geoms(obj), geoms(robot), geoms(target)

    def collides(a, b):
        return bool((m.geom_contype[a] & m.geom_conaffinity[b]) or (m.geom_contype[b] & m.geom_conaffinity[a]))
    if not all(any(collides(a, b) for b in rg) and any(collides(a, b) for b in tg) for a in og):
        raise ValueError('object/robot/receiver collision masks incompatible')
    if not og or not tg:
        raise ValueError('physical collision geometry required')
    if any(m.body_gravcomp[i] != 0 for i in obj | robot):
        raise ValueError('gravity compensation unsupported')
    joints = [i for i in range(m.njnt) if m.jnt_bodyid[i] in robot]
    for b in target:
        while b:
            if m.body_dofnum[b]:
                raise ValueError('receiver must be fixed during execution')
            b = int(m.body_parentid[b])
    pose = np.asarray(definition['task']['clip_pose' if task == 'rope' else 'box_pose'])
    R = pose[:3, :3]
    boxes = []
    for g in tg:
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError('v2 receiver collision must use box plate/walls')
        rot = R.T @ T[:3, :3] @ d.geom_xmat[g].reshape(3, 3)
        if not np.allclose(np.sort(np.abs(rot), axis=0), np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]]), atol=.06):
            raise ValueError('receiver walls must align with task frame')
        c = R.T @ (T[:3, :3] @ d.geom_xpos[g] + T[:3, 3])
        h = np.abs(rot) @ m.geom_size[g]
        boxes.append((c - h, c + h))
    plate = min(boxes, key=lambda b: (b[1] - b[0])[2])
    bottom, center = float(plate[1][2]), (plate[0] + plate[1]) / 2
    walls = [b for b in boxes if (b[1] - b[0])[2] > .01 and b[1][2] > bottom + .01]

    def inner(axis):
        low = [b for b in walls if (b[1] - b[0])[axis] < .02 and (b[0][axis] + b[1][axis]) / 2 < center[axis]]
        high = [b for b in walls if (b[1] - b[0])[axis] < .02 and (b[0][axis] + b[1][axis]) / 2 > center[axis]]
        if not low or not high:
            raise ValueError('receiver lacks opposing physical walls')
        a, b = max(low, key=lambda b: b[1][axis]), min(high, key=lambda b: b[0][axis])
        return a[1][axis], b[0][axis], min(a[1][2], b[1][2])
    xmin, xmax, rim = inner(0)
    if task == 'toy':
        ymin, ymax, ry = inner(1)
        rim = min(rim, ry)
    else:
        ymin, ymax = plate[0][1], plate[1][1]
    bounds = np.array([[xmin, ymin, bottom], [xmax, ymax, rim]])
    if np.any(bounds[1] <= bounds[0]):
        raise ValueError('receiver has no physical cavity')
    return dict(protocol=AUDIT_PROTOCOL, verified=True, gravity_base_m_s2=gravity.tolist(), robot_joints=len(joints),
                physics_dt=float(m.opt.timestep), control_dt=dt, receiver_bounds_task_frame=bounds.tolist(),
                receiver_rotation=R.tolist(), object_attachment=False, object_state_override=False,
                collision_pairs_verified=True)


def score(sd, states, audit, prefix=0.):
    """Metric record (bench schema) of an executed candidate against the sample's hidden twin reference."""
    sd = Path(sd)
    definition = json.loads((sd / 'hidden/metrics_v1/definition.json').read_text())
    T = np.asarray(definition['reference_from_camera'])
    task = definition['task']['kind']
    with np.load(states, allow_pickle=False) as z:
        points = z['object_points'] @ T[:3, :3].T + T[:3, 3]
        nodes = z['nodes'] @ T[:3, :3].T + T[:3, 3]
        times, edges = z['time_s'], z['edges']
        target = z['initial_target'] @ T[:3, :3].T + T[:3, 3]
    with np.load(sd / 'hidden/reference/states.npz', allow_pickle=False) as z:
        gt, gt_time = z['particles'], z['time_s']
    for arr in (points, nodes, times, gt, gt_time):
        if not np.isfinite(arr).all():
            raise ValueError('nonfinite scientific/execution input')
    R = np.asarray(audit['receiver_rotation'])
    bounds = np.asarray(audit['receiver_bounds_task_frame'])
    checks, details = [], []
    for p, n in zip(points, nodes):
        if task == 'pack':
            ok, detail = toy_in_box(p @ R, bounds)
        else:
            spec = dict(definition['task'], clip_pose=np.block([[R, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]).tolist(),
                        aperture_bounds=bounds.tolist())
            ok, detail = rope_through_aperture(spec, n, edges)
        checks.append(ok)
        details.append(detail)
    success = stable_result(checks, times)
    centers = points.mean(1)
    moved = np.linalg.norm(centers - centers[0], axis=1)
    stage1 = bool(np.any(moved >= (.05 if task == 'pack' else .02)))
    if task == 'pack':
        stage2 = bool(np.any(centers[:, 2] - centers[0, 2] >= .05))
    else:
        stage2 = bool(np.any(np.linalg.norm(nodes @ R - bounds.mean(0), axis=2) <= .05))
    done = [stage1, stage2, success]
    achieved = next((i for i, x in enumerate(done) if not x), 3)
    target_points = np.load(sd / 'hidden/metrics_v1/geometry.npz', allow_pickle=False)['target_points']
    deform = dict(initial_surface_cd_cm=cd(points[0], gt[0]),
                  scene_points_cd_cm=cd(np.concatenate([points[0], target]), np.concatenate([gt[0], target_points])),
                  surface_center_err_cm=float(np.linalg.norm(points[0].mean(0) - gt[0].mean(0)) * 100),
                  world_aabb_size_err_cm=float(np.linalg.norm(np.sort(np.ptp(points[0], axis=0)) - np.sort(np.ptp(gt[0], axis=0))) * 100),
                  terminal_surface_cd_cm=cd(points[-1], gt[-1]), trajectory=trajectory(points, times, gt, gt_time, prefix))
    if task == 'rope':
        deform['terminal_centerline_cm'] = cd(nodes[-1], centerline(gt[-1]))
    stage_names = ['moved', 'lifted' if task == 'pack' else 'near_clip', 'placed']
    return dict(schema_version='eval2/4', sample=sd.name, build=True, task_success=success,
                progress=dict(kind=task, stages=[dict(name=k, done=v) for k, v in zip(stage_names, done)],
                              achieved=achieved, total=3, progress=achieved / 3),
                deformable=deform, geometry_version='deformable-surface/2', metric_status={}, physics_audit=audit,
                task_detail=dict(protocol='twin-task/2', initial=details[0], final=details[-1], initial_satisfied=bool(checks[0]),
                                 stable_terminal=success, receiver='candidate physical cavity', all_frame_satisfied=checks),
                metric_definitions=dict(deformable='Unaligned surface/particle distances in frozen reference frame; not rigid ICP Shape CD or rigid Scene CD.',
                                        pose='Rigid pose APE/RPE not applicable to these deforming surfaces'))
