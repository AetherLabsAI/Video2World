"""Task outcome predicates for the in-house tasks, evaluated from the executed physics only (never from GT poses).

pick_and_hold          bilateral pad force grasp, 3 cm lift, held 5 cm high at the end
initially_held_place   carry, release, settle supported inside the receiver region
initially_held_insert  carry, body centre through the receiver mouth, release inside, retained
pull_out_and_release   grasp, extract the basket clear of the housing aperture, release, settle on support
wallet tasks           the same stages with geometric pad proximity as the holding witness (no force in particle physics)
"""
import numpy as np
from scipy.spatial.transform import Rotation

DT = .05
EPISODE_THRESHOLDS = ('bilateral_force_min_n', 'grasp_hold_s', 'terminal_hold_s', 'min_lift_m', 'progress_lift_m', 'release_hold_s',
                      'settle_s', 'settle_position_range_m', 'settle_angle_range_rad', 'support_height_tolerance_m', 'min_carry_m')


def first_run(mask, count, start=0):
    mask = np.asarray(mask, bool)
    for i in range(max(0, start), len(mask) - count + 1):
        if mask[i:i + count].all():
            return int(i)
    return None


def sustained(mask, n):
    mask = np.asarray(mask, bool)
    return np.flatnonzero(np.convolve(mask.astype(int), np.ones(n, int), 'valid') == n) + n - 1 if len(mask) >= n else np.array([], int)


def consecutive(mask, seconds, dt):
    n = max(1, int(np.ceil(seconds / dt - 1e-9)))
    m = np.asarray(mask, bool)
    out = np.zeros(len(m), bool)
    run = 0
    for i, value in enumerate(m):
        run = run + 1 if value else 0
        out[i] = run >= n
    return out


def stable(points, seconds, dt, limit):
    n = max(2, int(np.ceil(seconds / dt - 1e-9)))
    x = np.asarray(points)
    out = np.zeros(len(x), bool)
    for i in range(n - 1, len(x)):
        out[i] = np.linalg.norm(np.ptp(x[i - n + 1:i + 1], axis=0)) <= limit
    return out


def staged(stages):
    achieved = 0
    for s in stages:
        if not s['done']:
            break
        achieved += 1
    return achieved


# ---------------------------------------------------------------- task contracts
def validate_episode(contract):
    if contract.get('schema') != 'inhouse-episode-task/1':
        raise ValueError('Expected inhouse-episode-task/1')
    if contract.get('hand') not in ('l', 'r'):
        raise ValueError('Explicit task hand must be l or r')
    if contract.get('kind') not in ('pick_and_hold', 'initially_held_place'):
        raise ValueError('Unsupported episode task kind')
    for key in EPISODE_THRESHOLDS:
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


def validate_fryer(c):
    if c.get('schema') != 'inhouse-fryer-task/1' or c.get('kind') != 'pull_out_and_release' or c.get('hand') != 'r':
        raise ValueError('Expected right-hand fryer extraction task')
    for k in ['dt', 'bilateral_force_min_n', 'grasp_hold_s', 'release_hold_s', 'settle_s', 'settle_position_range_m', 'settle_angle_range_rad', 'support_height_tolerance_m', 'min_contact_pull_m']:
        if not isinstance(c.get(k), (int, float)) or not np.isfinite(c[k]) or c[k] <= 0:
            raise ValueError('Invalid ' + k)
    a, p = np.asarray(c['pull_axis_base'], float), np.asarray(c['aperture_point_base'], float)
    if a.shape != (3,) or p.shape != (3,) or not np.isfinite(np.r_[a, p]).all() or not np.isclose(np.linalg.norm(a), 1):
        raise ValueError('Invalid aperture frame')
    if not np.isfinite(c['support_z']) or not 0 <= c['release_open_threshold'] <= 1:
        raise ValueError('Invalid support/release')
    return c


# ---------------------------------------------------------------- contact evidence
def contact_forces(observations, hand, target='/World/Entities/target', physics_hz=60.):
    """Forces on the two opposing pads of one hand, from contacts with the target only."""
    if hand not in ('l', 'r'):
        raise ValueError('Unknown hand')
    forces = np.zeros((len(observations), 2, 3))
    within = lambda path, root: path == root or path.startswith(root + '/')
    for i, row in enumerate(observations):
        for c in row['contacts']:
            a, b = c['body0'], c['body1']
            if within(a, target):
                other = b
            elif within(b, target):
                other = a
            else:
                continue
            impulse = np.asarray(c['impulse'], float)
            if impulse.shape != (3,) or not np.isfinite(impulse).all():
                raise ValueError('Invalid contact impulse')
            for side, part in enumerate(('inner', 'outer')):
                if any(segment.startswith('gripper_' + hand + '_' + part) for segment in other.split('/')):
                    forces[i, side] += impulse * physics_hz
    return forces


# ---------------------------------------------------------------- predicates
def pick_or_place(trace, contract):
    """Force-based pick-and-hold or initially-held placement on a measured receiver region."""
    place = contract['kind'] == 'initially_held_place'
    fps = 20.
    xyz = np.asarray(trace['obj_xyz'] if place else trace['fabric_obj_xyz'], float)
    q = np.asarray(trace['obj_quat'] if place else trace['usd_obj_wxyz'], float)
    n = len(xyz)
    forces = np.linalg.norm(np.asarray(trace['pad_force_vectors']), axis=-1)
    if forces.shape != (n, 2) or not np.isfinite(forces).all():
        raise ValueError('missing/invalid target-specific bilateral forces')
    held = (forces > contract['bilateral_force_min_n']).all(axis=1)
    span = lambda key: max(1, int(round(contract[key] * fps)))
    grasp = first_run(held, span('grasp_hold_s'))
    if place:
        grip = np.asarray(trace['grip'])
        zone = np.asarray(contract['destination_zone'])
        center = np.asarray(trace['bbox_center_xyz'])
        bottom = np.asarray(trace['support_z'])
        # Destination correctness belongs to the final stage; a wrong-target release is still a release.
        carry = first_run(held & (np.linalg.norm(xyz - xyz[0], axis=1) >= contract['min_carry_m']), span('grasp_hold_s'))
        opened = (grip <= contract['release_open_threshold']) & (forces <= contract['bilateral_force_min_n']).all(axis=1)
        release = first_run(opened, span('release_hold_s'), start=carry if carry is not None else n)
        in_region = (np.abs(center[:, :2] - zone[:2]) <= zone[2:4]).all(axis=1)
        on_surface = np.abs(bottom - zone[4]) <= contract['support_height_tolerance_m']
        count = span('settle_s')
        tail = slice(max(0, n - count), n)
        trans_range = float(np.linalg.norm(np.ptp(center[tail], axis=0)))
        rots = Rotation.from_quat(np.roll(q[tail], -1, axis=1))
        angle_range = float((rots[-1].inv() * rots).magnitude().max())
        settled = bool(trans_range <= contract['settle_position_range_m'] and angle_range <= contract['settle_angle_range_rad'])
        goal = bool(n >= count and in_region[tail].all() and on_surface[tail].all() and opened[tail].all() and settled)
        success = bool(carry is not None and release is not None and goal)
        stages = [dict(name='carry', done=carry is not None, frame=carry), dict(name='release', done=release is not None, frame=release),
                  dict(name='supported_settled', done=success, frame=n - 1 if success else None)]
        details = dict(destination_zone=zone.tolist(), final_center_world_m=center[-1].tolist(), final_bottom_world_m=float(bottom[-1]),
                       receiver_region=bool(in_region[-1]), support_height_error_m=float(bottom[-1] - zone[4]), released=release is not None, stable=settled,
                       terminal_translation_range_m=trans_range, terminal_angle_range_rad=angle_range, carry_frame=carry, release_frame=release)
    else:
        dz = xyz[:, 2] - xyz[0, 2]
        lift = first_run(held & (dz >= contract['progress_lift_m']), span('grasp_hold_s'), start=grasp if grasp is not None else n)
        high = held & (dz >= contract['min_lift_m'])
        count = span('terminal_hold_s')
        terminal = bool(n >= count and high[-count:].all())
        success = bool(grasp is not None and lift is not None and terminal)
        stages = [dict(name='grasp', done=grasp is not None, frame=grasp), dict(name='lift', done=lift is not None, frame=lift),
                  dict(name='held_high', done=success, frame=n - 1 if success else None)]
        details = dict(final_lift_m=float(dz[-1]), max_lift_m=float(dz.max()), terminal_bilateral_held=bool(held[-count:].all()), terminal_hold_s=count / fps, lift_frame=lift)
    achieved = staged(stages)
    progress = dict(kind=contract['kind'], stages=stages, achieved=achieved, total=len(stages), progress=achieved / len(stages), order_ok=True)
    return dict(task_success=success, progress=progress,
                task_observations=dict(grasp_frame=grasp, terminal_pad_forces_n=forces[-1].tolist(), target_contact_nonzero_samples=int(held.sum()), **details),
                task_contract=contract)


def pick_with_witness(center, held, *, dt=DT):
    """3 cm lift / 5 cm terminal height with an explicitly supplied holding witness."""
    center, held = np.asarray(center, float), np.asarray(held, bool)
    n = len(center)
    if center.shape != (n, 3) or held.shape != (n,) or not np.isfinite(center).all():
        raise ValueError('Invalid hold observations')
    steps = lambda s: max(1, int(np.ceil(s / dt - 1e-9)))
    g = first_run(held, steps(.15))
    dz = center[:, 2] - center[0, 2]
    lift = first_run(held & (dz >= .03), steps(.15), start=g if g is not None else n)
    done = bool(g is not None and lift is not None and n >= steps(.25) and (held & (dz >= .05))[-steps(.25):].all())
    stages = [dict(name=k, done=v is not None, frame=v) for k, v in [('grasp', g), ('lift', lift), ('held_high', n - 1 if done else None)]]
    return dict(task_success=done, progress=dict(progress=staged(stages) / 3, stages=stages),
                task_observations=dict(final_lift_m=float(dz[-1]), holding_witness='supplied separately; no inferred force'))


def episode_outcome(observations, trajectory, grip, contract):
    """Outcome of a pick-and-hold or initially-held placement episode."""
    validate_episode(contract)
    bounds = np.asarray([r['bounds'] for r in observations], float)
    if bounds.shape != (len(observations), 2, 3) or not np.isfinite(bounds).all():
        raise ValueError('Invalid object bounds')
    xyz, quat = np.asarray(trajectory['xyz']), np.asarray(trajectory['wxyz'])
    grip = np.asarray(grip, float)
    if xyz.shape != (len(observations), 3) or quat.shape != (len(observations), 4):
        raise ValueError('Object trace and observations have different lengths')
    if grip.shape != (len(observations),) or not np.isfinite(grip).all() or np.any((grip < 0) | (grip > 1)):
        raise ValueError('Invalid grip trace')
    trace = dict(pad_force_vectors=contact_forces(observations, contract['hand']), obj_xyz=xyz, obj_quat=quat, fabric_obj_xyz=xyz, usd_obj_wxyz=quat,
                 grip=grip, bbox_center_xyz=bounds.mean(1), support_z=bounds[:, 0, 2])
    if contract['kind'] == 'pick_and_hold':
        held = (np.linalg.norm(trace['pad_force_vectors'], axis=-1) > .05).all(1)
        return pick_with_witness(np.c_[np.zeros((len(bounds), 2)), bounds[:, 0, 2]], held)
    return pick_or_place(trace, contract)


def bag_insert(center, forces, grip, mouth, half_xy, bottom, *, dt=DT, held_witness=None, released_witness=None):
    """Body centre through the mouth, release inside, terminal retention (a region proxy, not full mesh containment)."""
    no_forces = forces is None
    if no_forces and (held_witness is None or released_witness is None):
        raise ValueError('Missing force or geometric holding/release witnesses')
    center, forces, grip, mouth, half_xy, bottom = map(lambda x: np.asarray(x, float), (center, forces, grip, mouth, half_xy, bottom))
    n = len(center)
    if n < 3 or center.shape != (n, 3) or (not no_forces and forces.shape != (n, 2)) or grip.shape != (n,) or mouth.shape != (n, 3) or half_xy.shape != (n, 2) or bottom.shape != (n,):
        raise ValueError('Incomplete bag observations')
    if not all(np.isfinite(x).all() for x in [center, grip, mouth, half_xy, bottom] + ([] if no_forces else [forces])) or (half_xy <= 0).any():
        raise ValueError('Nonfinite/invalid bag observations')
    if (grip < 0).any() or (grip > 1).any() or (not no_forces and (forces < 0).any()) or dt <= 0:
        raise ValueError('Invalid grip, forces or clock')
    rel = center - mouth
    localxy = rel[:, :2]
    inside = (np.abs(localxy) <= half_xy).all(1) & (rel[:, 2] < 0) & (center[:, 2] >= bottom)
    held = (forces > .05).all(1) if held_witness is None else np.asarray(held_witness, bool)
    opened = (grip <= .2) & (forces <= .05).all(1) if released_witness is None else np.asarray(released_witness, bool)
    if held.shape != (n,) or opened.shape != (n,):
        raise ValueError('Invalid holding/release witnesses')
    steps = lambda s: max(1, int(np.ceil(s / dt - 1e-9)))
    carry = first_run(held & (np.linalg.norm(center - center[0], axis=1) >= .05), steps(.15))
    crosses = []
    for i in range(1, n):
        if rel[i - 1, 2] > 0 and rel[i, 2] <= 0:
            alpha = rel[i - 1, 2] / (rel[i - 1, 2] - rel[i, 2])
            xy = localxy[i - 1] * (1 - alpha) + localxy[i] * alpha
            half = half_xy[i - 1] * (1 - alpha) + half_xy[i] * alpha
            if (np.abs(xy) <= half).all() and carry is not None and i >= carry:
                crosses.append(i)
    entry = crosses[0] if crosses else None
    release = first_run(opened & inside, steps(.25), start=entry if entry is not None else n)
    count = steps(1.)
    terminal_range = float(np.linalg.norm(np.ptp(rel[-count:], axis=0)))
    retained = bool(n >= count and inside[-count:].all() and opened[-count:].all() and terminal_range <= .005)
    success = bool(entry is not None and release is not None and retained)
    stages = [dict(name=k, done=v is not None, frame=v) for k, v in [('carry', carry), ('through_mouth', entry), ('release_inside', release), ('retained', n - 1 if success else None)]]
    achieved = staged(stages)
    return dict(task_success=success, progress=dict(progress=achieved / 4, achieved=achieved, total=4, stages=stages),
                task_observations=dict(initial_inside=bool(inside[0]), terminal_inside=bool(inside[-1]), terminal_released=bool(opened[-count:].all()),
                                       terminal_relative_range_m=terminal_range, terminal_retained=retained, mouth_crossing_frames=crosses,
                                       final_center_relative_m=rel[-1].tolist(), final_half_spans_m=half_xy[-1].tolist(),
                                       region_source='Candidate receiver entity-local frame with physical cavity validation'))


def bag_candidate_region(center, receiver, pkg, grip, forces=None, held_witness=None, released_witness=None):
    """bag_insert in the authored receiver's local frame, after validating its physical cavity."""
    center = np.asarray(center, float)
    n = len(center)
    inside_cavity(center[:, None, :], receiver, pkg, 0.)
    R = Rotation.from_quat(np.roll(receiver['quat_wxyz'], -1)).as_matrix()
    local = (center - receiver['pos']) @ R
    inter = receiver['interior']
    mouth = np.tile([*inter['center'][:2], inter['rim_z']], (n, 1))
    half = np.tile(inter['half_extents_xy'], (n, 1))
    return bag_insert(local, forces, grip, mouth, half, np.full(n, inter['bottom_z']), held_witness=held_witness, released_witness=released_witness)


def inside_cavity(vertices, receiver, pkg, tolerance):
    """Per frame: is the whole surface inside the candidate cavity? The cavity must have real side and bottom walls
    and an open mouth (ray checks), so metadata cannot describe empty space or a filled box."""
    import trimesh
    from pathlib import Path
    q = Rotation.from_quat(np.roll(receiver['quat_wxyz'], -1)).as_matrix()
    local = (vertices - np.array(receiver['pos'])) @ q
    r = receiver['interior']
    c, half, bottom, top = np.array(r['center']), np.array(r['half_extents_xy']), r['bottom_z'], r['rim_z']
    with np.load(Path(pkg) / receiver['geometry_npz']) as z:
        m = trimesh.Trimesh(z['vertices'], z['face_vertex_indices'].reshape(-1, 3), process=False)
    probes = np.array([[c[0] + x * half[0], c[1] + y * half[1], bottom + (top - bottom) * z] for x in [-.5, 0, .5] for y in [-.5, 0, .5] for z in [.35, .65]])
    if m.is_watertight and m.contains(probes).any():
        raise ValueError('Receiver interior lies in solid material')
    dirs = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, -1]])
    _, idx, _ = m.ray.intersects_location(np.repeat(probes, 5, axis=0), np.tile(dirs, (len(probes), 1)), multiple_hits=False)
    valid = np.zeros(len(probes) * 5, bool)
    valid[idx] = True
    if not valid.all():
        raise ValueError('Receiver cavity lacks physical side/bottom walls')
    mouths = np.array([[c[0] + x * half[0], c[1] + y * half[1], top - .01] for x, y in [(0, 0), (.4, 0), (-.4, 0), (0, .4), (0, -.4)]])
    if m.ray.intersects_any(mouths, np.tile([0, 0, 1], (len(mouths), 1))).any():
        raise ValueError('Receiver has a blocked opening')
    xy = (np.abs(local[..., :2] - c[:2]) <= half + tolerance).all((1, 2))
    vertical = (local[..., 2].min(1) >= bottom - tolerance) & (local[..., 2].max(1) <= top + tolerance)
    return xy & vertical


def fryer_outcome(observations, trajectory, grip, c, vertices):
    """Grasp inside the housing, contact-driven pull until the whole basket clears the aperture, release, settle on support."""
    validate_fryer(c)
    xyz, quat, grip, v = (np.asarray(x, float) for x in (trajectory['xyz'], trajectory['wxyz'], grip, vertices))
    N = len(xyz)
    if xyz.shape != (N, 3) or quat.shape != (N, 4) or grip.shape != (N,) or len(observations) != N or v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(np.r_[xyz.ravel(), quat.ravel(), grip, v.ravel()]).all():
        raise ValueError('Invalid physical observations')
    rr = Rotation.from_quat(np.roll(quat, -1, axis=1))
    world = np.einsum('nij,vj->nvi', rr.as_matrix(), v) + xyz[:, None, :]
    axis, plane = np.asarray(c['pull_axis_base']), np.asarray(c['aperture_point_base'])
    clearance = ((world - plane) @ axis).min(1)
    bottom = np.asarray([o['bounds'] for o in observations])[:, 0, 2]
    force = np.linalg.norm(contact_forces(observations, c['hand']), axis=2)
    held = (force >= c['bilateral_force_min_n']).all(1)
    released = grip <= c['release_open_threshold']
    steps = lambda k: max(1, int(np.ceil(c[k] / c['dt'])))
    h = sustained(held, steps('grasp_hold_s'))
    initial_inside = bool(clearance[0] < 0)
    grasp = bool(len(h) and initial_inside)
    g = int(h[0]) if grasp else N
    tol = c.get('clearance_tolerance_m', .003)
    cleared = np.flatnonzero((np.arange(N) >= g) & (clearance >= tol))
    engaged_pull = bool(grasp and np.any(held & (np.arange(N) >= g) & (((xyz - xyz[min(g, N - 1)]) @ axis) >= c['min_contact_pull_m'])))
    extract = bool(len(cleared) and engaged_pull)
    e = int(cleared[0]) if extract else N
    rel = sustained(released & (np.arange(N) >= e), steps('release_hold_s'))
    release = bool(len(rel) and extract)
    n = steps('settle_s')
    tail = xyz[-n:]
    angle = (rr[-n:] * rr[-1].inv()).magnitude()
    settled = bool(N >= n and np.linalg.norm(np.ptp(tail, axis=0)) <= c['settle_position_range_m'] and angle.max() <= c['settle_angle_range_rad'])
    supported = bool(np.max(np.abs(bottom[-n:] - c['support_z'])) <= c['support_height_tolerance_m'])
    terminal_clear = bool((clearance[-n:] >= tol).all())
    done = bool(release and settled and supported and terminal_clear and released[-n:].all())
    stages = dict(grasp=grasp, extracted=extract, released=release, supported_settled=done)
    return dict(task_success=done, progress=dict(progress=sum(stages.values()) / 4, stages=stages),
                task_observations=dict(initial_inside=initial_inside, contact_driven_pull=engaged_pull, first_grasp_frame=None if not grasp else g,
                                       first_extracted_frame=None if not extract else e, first_release_frame=None if not release else int(rel[0]),
                                       maximum_clearance_m=float(clearance.max()), terminal_clearance_m=float(clearance[-1]), bilateral_contact_frames=int(held.sum()),
                                       terminal_supported=supported, terminal_settled=settled))


def wallet_outcome(features, z, pkg, contract, receiver=None):
    """Wallet pick-and-hold / insertion with pad proximity plus sustained relative holding as the holding witness."""
    c = contract
    n = len(features['center'])
    near = (features['pad_distances_m'] <= c['proximity_m']).all(1) & (features['grip'] >= c['grip_closed_min'])
    held = near & stable(features['center_in_gripper'], .15, .05, c['hold_center_range_m'])
    if c['kind'] == 'pick_and_hold':
        result = pick_with_witness(np.c_[np.zeros((n, 2)), features['bottom_z']], held)
    else:
        v = z['vertices_base']
        center = (v.min(1) + v.max(1)) / 2
        released = (features['grip'] <= .2) & (features['pad_distances_m'] > c['proximity_m']).all(1)
        result = bag_candidate_region(center, receiver, pkg, features['grip'], held_witness=held, released_witness=released)
    result['evidence_kind'] = 'Native rigid/elastic surface and pad proximity plus sustained relative holding; no invented forces'
    return result


# ---------------------------------------------------------------- task geometry from the candidate scene
def surface_height(e, pkg):
    """Receiver region [cx, cy, hx, hy, top z] from the candidate receiver mesh."""
    from v2w.metrics import geometry as M
    mesh = M.entity_mesh(e, pkg)
    v = np.asarray(mesh.vertices)
    r = Rotation.from_quat(np.roll(e['quat'], -1)).as_matrix()
    v = v @ r.T + e['pos']
    tris = v[np.asarray(mesh.faces)]
    lo, hi = v.min(0), v.max(0)
    xy = (lo[:2] + hi[:2]) / 2
    heights = []
    for tri in tris:
        a = (tri[1:, :2] - tri[0, :2]).T
        if abs(np.linalg.det(a)) < 1e-12:
            continue
        uv = np.linalg.solve(a, xy - tri[0, :2])
        if np.min(uv) >= -1e-8 and uv.sum() <= 1 + 1e-8:
            heights.append(float(tri[0, 2] + uv @ (tri[1:, 2] - tri[0, 2])))
    if not heights:
        raise ValueError('Receiver has no horizontal support at its center')
    return [*xy, *((hi[:2] - lo[:2]) / 2), max(heights)]


def receiver_frame(entity, geodir):
    """Receiver top-face frame: long edge of the minimum-area rectangle of its top vertices, z up."""
    import cv2
    from v2w.metrics import geometry as M
    v = np.asarray(M.entity_mesh(entity, geodir).vertices)
    v = v @ Rotation.from_quat(np.roll(entity['quat'], -1)).as_matrix().T + entity['pos']
    top = v[v[:, 2] > v[:, 2].max() - .005]
    rect = cv2.minAreaRect(top[:, :2].astype(np.float32))
    corners = cv2.boxPoints(rect)
    edges = np.roll(corners, -1, axis=0) - corners
    axis = np.r_[edges[np.argmax(np.linalg.norm(edges, axis=1))], 0.]
    axis /= np.linalg.norm(axis)
    axis *= 1 if axis[np.argmax(abs(axis))] >= 0 else -1
    T = np.eye(4)
    T[:3, :3] = np.c_[axis, np.cross([0, 0, 1], axis), [0, 0, 1]]
    T[:3, 3] = [*rect[0], float(top[:, 2].max())]
    return T


def fryer_aperture(contract, scene, pkg):
    """Extraction frame from the submitted housing walls and support; no GT input."""
    from v2w.metrics import geometry as M

    def vertices(e):
        v = np.asarray(M.entity_mesh(e, pkg).vertices, float)
        return Rotation.from_quat(np.roll(e.get('quat', [1, 0, 0, 0]), -1)).apply(v) + np.asarray(e['pos'])

    c = dict(contract)
    by = {e['name']: e for e in scene.get('props', [])}
    back, left, right = vertices(by['housing_back']), vertices(by['housing_left']), vertices(by['housing_right'])
    center = lambda v: (v.min(0) + v.max(0)) / 2
    sides = (center(left) + center(right)) / 2
    axis = sides - center(back)
    axis /= np.linalg.norm(axis)
    front = float(max((left @ axis).max(), (right @ axis).max()))
    point = sides + axis * (front - sides @ axis)
    support = next(e for e in scene.get('props', []) if e['name'] in ('support', 'countertop', 'receiver'))
    sv = vertices(support)
    if not np.isfinite(np.r_[axis, point, sv.ravel()]).all():
        raise ValueError('Invalid physical housing geometry')
    c.update(pull_axis_base=axis.tolist(), aperture_point_base=point.tolist(), support_z=float(sv[:, 2].max()),
             geometry_frame_source='Submitted housing side-wall front edges; outward direction from the back wall toward the side-wall centres.')
    return c
