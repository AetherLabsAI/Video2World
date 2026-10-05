"""In-house ground truth and the metrics that compare an execution with it.

GT kinds: accepted object pose sequences (full or masked clock), complete 3D point trajectories with REAL depth-ray
support, complete wallet surface meshes, observable object axes, visible-scene surfaces and provider object models.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from v2w import paths

POINT_CAP_CM = 10.
POINT_DELTA = 4
MIN_SURFACE_POINTS = 40
SURFACE_SAMPLES = 2048
SURFACE_SEED = 20260920
D2 = tuple(np.diag(s) for s in [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)])
ROTATION_VERSION = 'inhouse-observable-rotation/1'


def read(p):
    return json.loads(Path(p).read_text())


def scene_doc(path):
    """A GT scene document with recorded mesh locations resolved into the data release."""
    doc = read(path)
    for e in doc.get('objects', []) + doc.get('props', []):
        for key in [k for k in e if k.endswith('_path')]:
            if isinstance(e[key], str) and Path(e[key]).is_absolute():
                e[key] = str(paths.resolve(e[key]))
    return doc


@lru_cache(maxsize=1)
def registry():
    return read(Path(__file__).with_name('registry.json'))


def se3(T, what):
    if not np.allclose(T[:, 3, :], [0, 0, 0, 1], atol=1e-5) or not np.allclose(T[:, :3, :3].transpose(0, 2, 1) @ T[:, :3, :3], np.eye(3), atol=1e-4) or not np.allclose(np.linalg.det(T[:, :3, :3]), 1, atol=1e-4):
        raise ValueError(what + ' are not SE3')


# ---------------------------------------------------------------- object pose GT
def pose_sequence(sd):
    """Accepted full-clock object poses (episode samples)."""
    sd = Path(sd)
    acceptance = read(sd / 'gt_acceptance.json')
    if acceptance.get('schema') != 'inhouse-owner-accepted-estimate/1' or acceptance.get('scope') != 'reviewed_full_sequence' or acceptance['sample'] != sd.name:
        raise ValueError('Missing full-sequence pose GT acceptance')
    with np.load(sd / 'hidden/objects_6d.npz', allow_pickle=False) as z:
        gt = {k: z[k] for k in z.files}
    T, ts = gt['primary__T'], gt['timestamp_ns']
    if T.shape != (len(ts), 4, 4) or len(ts) < 2 or not np.isfinite(T).all() or not np.all(np.diff(ts) > 0):
        raise ValueError('Invalid accepted pose/time sequence')
    se3(T, 'Accepted poses')
    return gt


def masked_pose_sequence(sd):
    """Accepted object poses on the full clock with explicitly unknown frames (reviewed and fryer samples)."""
    sd = Path(sd)
    a = read(sd / 'gt_acceptance.json')
    if a.get('schema') != 'inhouse-owner-accepted-masked-estimate/1' or a.get('scope') != 'reviewed_available_frames_on_full_timeline' or a['sample'] != sd.name:
        raise ValueError('Wrong masked pose GT acceptance')
    with np.load(sd / 'hidden/objects_6d.npz', allow_pickle=False) as z:
        gt = dict(z)
    with np.load(sd / 'hidden/scoring_validity.npz', allow_pickle=False) as z:
        mask, clock = z['pose_valid'], z['timestamp_ns']
    T, ts = gt['primary__T'], gt['timestamp_ns']
    if T.shape != (len(ts), 4, 4) or mask.dtype != bool or mask.shape != (len(ts),) or not mask.any() or not mask[0] or not np.array_equal(ts, clock) or not np.all(np.diff(ts) > 0):
        raise ValueError('Invalid full clock or initial GT')
    if not np.array_equal(mask, np.isfinite(T).all((1, 2))) or not np.isnan(T[~mask]).all():
        raise ValueError('Unknown GT frames must be NaN and only those')
    se3(T[mask], 'Known poses')
    if int(mask.sum()) != a['accepted_frames'] or len(mask) != a['timeline_frames'] or np.flatnonzero(~mask).tolist() != a['unknown_frames']:
        raise ValueError('Accepted frame range changed')
    return gt, mask


def poses7(T):
    return np.c_[T[:, :3, 3], np.roll(Rotation.from_matrix(T[:, :3, :3]).as_quat(), 1, axis=1)]


def pose_metrics(P, G, orientation, dt=.05, time_s=None):
    """Initial-object-anchored APE (capped) and, with full registered axes, rotation APE and RPE."""
    from v2w.metrics import geometry as M
    a, b = poses7(P), poses7(G)
    a[:, :3] -= a[0, :3]
    b[:, :3] -= b[0, :3]
    e = np.linalg.norm(a[:, :3] - b[:, :3], axis=1) * 100
    ape = M.ape_capped(a, b)
    ape.update(trans_p90_cm=float(np.percentile(e, 90)), trans_per_frame_cm=e.tolist(), n_frames=len(e), coordinate_frame='initial-object anchor translation; fixed model-base axes')
    if orientation == 'full':
        rot = np.array([M.sym_rot_err(x[3:], y[3:], None, 'z') for x, y in zip(a, b)])
        ape.update(rot_deg=float(rot.mean()), rot_per_frame_deg=rot.tolist())
        rpe = M.rpe(a, b, delta=4, time_s=np.arange(len(a)) * dt if time_s is None else time_s)
    else:
        ape.update(rot_deg=None, rot_na_reason='Accepted GT does not register full object axes')
        rpe = None
    return ape, rpe


def interpolate(t, T, q):
    """Poses at query times (linear / SLERP); spans longer than 1.5 control steps are never bridged."""
    if t.dtype != np.int64 or t.ndim != 1 or len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError('strictly increasing int64 nanosecond clock required')
    if T.shape != (len(t), 4, 4) or not np.isfinite(T).all():
        raise ValueError('finite Nx4x4 poses required')
    if not np.allclose(T[:, 3, :], [0, 0, 0, 1], atol=1e-7):
        raise ValueError('invalid homogeneous bottom row')
    R = T[:, :3, :3]
    if not np.allclose(R.transpose(0, 2, 1) @ R, np.eye(3), atol=1e-5) or not np.allclose(np.linalg.det(R), 1, atol=1e-5):
        raise ValueError('SE3 poses only, no scale/shear/reflection')
    covered = (q >= t[0]) & (q <= t[-1])
    out = np.full((len(q), 4, 4), np.nan)
    origin = int(t[0])
    x = (t - origin) * 1e-9
    idx = np.searchsorted(t, q[covered], side='right').clip(1, len(t) - 1)
    ok = (t[idx] - t[idx - 1] <= 75_000_000) | np.isin(q[covered], t)
    valid = np.flatnonzero(covered)[ok]
    xs = (q[valid] - origin) * 1e-9
    out[valid] = np.eye(4)
    out[valid, :3, :3] = Slerp(x, Rotation.from_matrix(T[:, :3, :3]))(xs).as_matrix()
    for j in range(3):
        out[valid, j, 3] = np.interp(xs, x, T[:, j, 3])
    covered[:] = False
    covered[valid] = True
    return out, covered


# ---------------------------------------------------------------- point GT
def point_sequence(sample):
    """Complete accepted 3D point trajectory on the 20 Hz video clock."""
    sample = Path(sample)
    acceptance = read(sample / 'gt_acceptance.json')
    if acceptance.get('schema') != 'inhouse-owner-accepted-point/1' or acceptance.get('status') != 'accepted' or acceptance['sample_id'] != sample.name:
        raise ValueError('Accepted point GT required')
    definition = read(sample / 'point_definition.json')
    with np.load(sample / 'hidden/point_trajectory.npz', allow_pickle=False) as z:
        xyz, times, timestamps, valid = z['point_base_m'].copy(), z['time_s'].copy(), z['timestamp_ns'].copy(), z['available'].copy()
    n = len(times)
    if n < POINT_DELTA + 1 or xyz.shape != (n, 3) or not np.isfinite(xyz).all():
        raise ValueError('Complete finite XYZ GT required')
    if valid.shape != (n,) or valid.dtype.kind != 'b' or not valid.all():
        raise ValueError('Every task frame must be available')
    if timestamps.shape != (n,) or timestamps.dtype.kind not in 'iu' or not np.all(np.diff(timestamps) == 50_000_000):
        raise ValueError('GT sensor-grid clock changed')
    if not np.allclose(times, np.arange(n) * .05, atol=1e-9, rtol=0):
        raise ValueError('Full video clock required')
    return dict(xyz=xyz, time_s=times, timestamp_ns=timestamps, definition=definition, frames=n)


def point_task_gt(sd):
    """The point GT bound to a robot point task sample."""
    b = read(Path(sd) / 'point_gt_binding.json')
    gt = point_sequence(paths.resolve(b['source_point_sample']))
    if gt['frames'] != b['frames'] or gt['definition']['point_kind'] != b['point_kind']:
        raise ValueError('Point GT identity changed')
    return gt, b


def observed_surface(mesh, xyz, wxyz, support):
    """The target point seen along the exact REAL depth rays: median of the executed target surface hits per frame.
    Other entities are excluded; frames with fewer than MIN_SURFACE_POINTS hits are missing."""
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    raycaster = RayMeshIntersector(mesh)
    n = len(xyz)
    points, coverage, hits_count = np.full((n, 3), np.nan), np.zeros(n), np.zeros(n, np.int64)
    offsets, rays, heads = support['offsets'], support['ray_head'], support['T_base_head']
    if len(offsets) != n + 1 or heads.shape != (n, 4, 4):
        raise ValueError('Observation clock mismatch')
    rotations = Rotation.from_quat(np.roll(wxyz, -1, axis=1)).as_matrix()
    for f in range(n):
        rays_h, H, R, p = rays[offsets[f]:offsets[f + 1]], heads[f], rotations[f], xyz[f]
        origin = (H[:3, 3] - p) @ R
        directions = rays_h @ H[:3, :3].T @ R
        _, ray, loc = raycaster.intersects_id(np.broadcast_to(origin, directions.shape), directions, multiple_hits=False, return_locations=True)
        if not len(ray):
            continue
        base = loc @ R.T + p
        ph = (base - H[:3, 3]) @ H[:3, :3]
        z = ph[:, 2]
        valid = np.isfinite(ph).all(1) & (z >= .2) & (z <= 2.5)
        ph, z = ph[valid], z[valid]
        if not len(z):
            continue
        med = np.median(z)
        mad = np.median(abs(z - med))
        ph = ph[abs(z - med) <= max(.015, 3 * 1.4826 * mad)]
        hits_count[f] = len(ph)
        if len(ph) < MIN_SURFACE_POINTS:
            continue
        points[f] = np.median(ph, axis=0) @ H[:3, :3].T + H[:3, 3]
        coverage[f] = len(ph) / len(rays_h)
    return points, coverage, hits_count


def point_metrics(truth, prediction, coverage):
    """Point position / 0.2 s displacement errors over every frame; a missing observation costs the cap."""
    truth, prediction, coverage = (np.asarray(x, float) for x in (truth, prediction, coverage))
    n, cap, delta = len(truth), POINT_CAP_CM, POINT_DELTA
    if truth.shape != (n, 3) or prediction.shape != truth.shape or n <= delta or not np.isfinite(truth).all():
        raise ValueError('Invalid complete point truth or prediction dimensions')
    if coverage.shape != (n,) or not np.isfinite(coverage).all() or np.any((coverage < 0) | (coverage > 1)):
        raise ValueError('Invalid evaluator coverage')
    available = np.isfinite(prediction).all(1)
    if np.any((coverage > 0) & ~available):
        raise ValueError('Positive coverage without a finite point')
    raw = np.full(n, np.nan)
    raw[available] = np.linalg.norm(prediction[available] - truth[available], axis=1) * 100
    penalized = np.full(n, cap)
    penalized[available] = coverage[available] * np.minimum(raw[available], cap) + (1 - coverage[available]) * cap
    interval_coverage = np.array([min(coverage[f:f + delta + 1]) for f in range(n - delta)])
    iv = interval_coverage > 0
    rd = np.full(n - delta, np.nan)
    rd[iv] = np.linalg.norm((prediction[delta:] - prediction[:-delta])[iv] - (truth[delta:] - truth[:-delta])[iv], axis=1) * 100
    pd = np.full(n - delta, cap)
    pd[iv] = interval_coverage[iv] * np.minimum(rd[iv], cap) + (1 - interval_coverage[iv]) * cap
    if not np.isfinite(penalized).all() or not np.isfinite(pd).all():
        raise ValueError('Point errors overflowed')
    stat = lambda x: dict(mean_cm=float(x.mean()), median_cm=float(np.median(x)), p95_cm=float(np.percentile(x, 95)), max_cm=float(x.max()), count=len(x))
    observed = lambda x: stat(x[np.isfinite(x)]) if np.isfinite(x).any() else dict(mean_cm=None, median_cm=None, p95_cm=None, max_cm=None, count=0)
    return dict(metric_version='inhouse-native-point-observation/2.0', point_position=stat(penalized), point_displacement=stat(pd),
                point_position_observed_uncapped=observed(raw), point_displacement_observed_uncapped=observed(rd),
                point_observation=dict(mean_coverage=float(coverage.mean()), frames=n, available_frames=int((coverage > 0).sum()), scored_frames=n,
                                       missing_frames=np.flatnonzero(coverage == 0).tolist(), intervals=n - delta, minimum_surface_points=MIN_SURFACE_POINTS, penalty_cm=cap),
                arrays=dict(point_error_cm=raw, penalized_point_error_cm=penalized, displacement_error_cm=rd, penalized_displacement_error_cm=pd,
                            coverage=coverage, interval_coverage=interval_coverage))


# ---------------------------------------------------------------- wallet surface GT
def triangles(v, f):
    v, f = np.asarray(v, float), np.asarray(f)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) < 3 or not np.isfinite(v).all():
        raise ValueError('Invalid/nonfinite required mesh vertices')
    if f.ndim != 2 or f.shape[1] != 3 or not np.issubdtype(f.dtype, np.integer) or not len(f) or f.min() < 0 or f.max() >= len(v):
        raise ValueError('Invalid triangle topology')
    t = v[f]
    area = np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1) / 2
    if not np.isfinite(area).all() or area.sum() <= 1e-12:
        raise ValueError('Zero-area surface')
    return t, area


def surface_center(v, f):
    t, a = triangles(v, f)
    return np.average(t.mean(axis=1), weights=a, axis=0)


def surface_points(v, f, n=SURFACE_SAMPLES):
    t, a = triangles(v, f)
    rng = np.random.default_rng(SURFACE_SEED)
    ids = rng.choice(len(t), n, p=a / a.sum())
    u, w = np.sqrt(rng.random(n)), rng.random(n)
    return (1 - u)[:, None] * t[ids, 0] + (u * (1 - w))[:, None] * t[ids, 1] + (u * w)[:, None] * t[ids, 2]


def surface_sequence(sd):
    """Accepted complete surface mesh sequence (wallet samples)."""
    sd = Path(sd)
    accept = read(sd / 'gt_acceptance.json')
    if accept.get('schema') != 'inhouse-owner-accepted-deformable/1' or accept.get('status') != 'accepted':
        raise ValueError('Accepted deformable GT required')
    z = np.load(sd / 'hidden/complete_mesh_sequence.npz', allow_pickle=False)
    with np.load(sd / 'hidden/scoring_validity.npz', allow_pickle=False) as masks:
        valid, times = masks['shape_valid'].copy(), masks['video_time_s'].copy()
    v, f, available = z['vertices_base'], z['faces'], z['candidate_available']
    if not np.array_equal(valid, available) or len(v) != len(times) or not valid.any():
        raise ValueError('Accepted GT range changed')
    if np.any(np.diff(times) <= 0) or not np.isfinite(times).all():
        raise ValueError('Invalid GT clock')
    return dict(vertices=v, faces=f, valid=valid, time_s=times)


def prediction_at(vertices, times, query):
    """Fixed-topology prediction at a GT time: exact frame or linear interpolation; no extrapolation."""
    if times.ndim != 1 or len(times) != len(vertices) or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Invalid prediction clock')
    if query < times[0] - 1e-8 or query > times[-1] + 1e-8:
        raise ValueError('Prediction does not cover every accepted GT timestamp')
    k = int(np.searchsorted(times, query))
    if k < len(times) and abs(times[k] - query) < 1e-8:
        return vertices[k]
    if k == 0:
        return vertices[0]
    if k == len(times):
        return vertices[-1]
    alpha = (query - times[k - 1]) / (times[k] - times[k - 1])
    return vertices[k - 1] * (1 - alpha) + vertices[k] * alpha


def surface_distances(query, vertices, faces, chunk=64):
    """Exact point-to-triangle distances, pruned with certified triangle AABB lower bounds."""
    import trimesh
    from scipy.spatial import cKDTree
    t, _ = triangles(vertices, faces)
    lo, hi = t.min(1), t.max(1)
    tree = cKDTree(t.mean(1))
    out = []
    for begin in range(0, len(query), chunk):
        p = np.asarray(query[begin:begin + chunk])
        ids = np.asarray(tree.query(p, k=min(8, len(t)))[1]).reshape(len(p), -1)
        cp = trimesh.triangles.closest_point(t[ids].reshape(-1, 3, 3), np.repeat(p, ids.shape[1], axis=0))
        upper = (np.square(cp.reshape(len(p), -1, 3) - p[:, None]).sum(2)).min(1)
        delta = np.maximum(np.maximum(lo[None] - p[:, None], p[:, None] - hi[None]), 0)
        lower = np.einsum('pfi,pfi->pf', delta, delta)
        pi, fi = np.nonzero(lower <= upper[:, None] + 1e-14)
        cp = trimesh.triangles.closest_point(t[fi], p[pi])
        np.minimum.at(upper, pi, np.square(cp - p[pi]).sum(1))
        out.append(np.sqrt(np.maximum(upper, 0)))
    return np.concatenate(out)


def surface_cd(v, f, w, g):
    if np.array_equal(v, w) and np.array_equal(f, g):
        return 0.
    return float(100 * (surface_distances(surface_points(v, f), w, g).mean() + surface_distances(surface_points(w, g), v, f).mean()))


def surface_metrics(gt, pred):
    """Surface CD, centroid and world-AABB size errors on every accepted GT frame; no alignment."""
    pv, pf, pt = pred['vertices_base'], pred['faces'], pred['video_time_s']
    rows = []
    for i in np.flatnonzero(gt['valid']):
        t = float(gt['time_s'][i])
        v, w, g = prediction_at(pv, pt, t), gt['vertices'][i], gt['faces']
        rows.append(dict(frame=int(i), time_s=t, surface_cd_cm=surface_cd(v, pf, w, g),
                         surface_center_err_cm=float(100 * np.linalg.norm(surface_center(v, pf) - surface_center(w, g))),
                         world_aabb_size_err_cm=float(100 * np.abs(np.ptp(v, axis=0) - np.ptp(w, axis=0)).mean())))
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    total = len(gt['valid'])
    return dict(evaluation_protocol='inhouse-deformable-surface/1.0',
                deformable=dict(trajectory=dict(shape_cd_cm=mean('surface_cd_cm'), coverage=len(rows) / total, scored_frames=len(rows), total_frames=total),
                                surface_center_err_cm=mean('surface_center_err_cm'), world_aabb_size_err_cm=mean('world_aabb_size_err_cm')),
                per_frame=rows, excluded_gt_frames=np.flatnonzero(~gt['valid']).tolist())


# ---------------------------------------------------------------- observable object axes
def mesh_frame(vertices, faces):
    v, f = np.asarray(vertices, float), np.asarray(faces, int)
    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all() or not len(f):
        raise ValueError('Observable rotation requires finite candidate surface geometry')
    t = v[f]
    area = np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1) / 2
    if not np.isfinite(area).all() or area.sum() <= 0:
        raise ValueError('Candidate surface has zero area')
    weights = area / area.sum()
    s = t.sum(1)
    center = np.sum(weights[:, None] * s / 3, axis=0)
    moment = np.sum(weights[:, None, None] * (np.einsum('fvi,fvj->fij', t, t) + np.einsum('fi,fj->fij', s, s)) / 12, axis=0)
    w, A = np.linalg.eigh(moment - np.outer(center, center))
    w, A = w[::-1], A[:, ::-1]
    if np.linalg.det(A) < 0:
        A[:, -1] *= -1
    return center, w, A


def sample_axes(times, axes, query):
    times, axes, query = np.asarray(times, float), np.asarray(axes, float), np.asarray(query, float)
    if times.ndim != 1 or not len(times) or axes.shape != (len(times), 3, 3) or not np.isfinite(times).all() or not np.isfinite(axes).all():
        raise ValueError('Invalid executed orientation trajectory')
    if len(times) > 1 and not np.all(np.diff(times) > 0):
        raise ValueError('Execution timestamps must strictly increase')
    if not np.allclose(np.swapaxes(axes, 1, 2) @ axes, np.eye(3), atol=1e-6) or not np.allclose(np.linalg.det(axes), 1, atol=1e-6):
        raise ValueError('Executed orientation is not a proper rotation')
    covered = (query >= times[0] - 1e-8) & (query <= times[-1] + 1e-8)
    if len(times) == 1:
        return np.repeat(axes, len(query), axis=0), covered
    return Slerp(times, Rotation.from_matrix(axes))(np.clip(query, times[0], times[-1])).as_matrix(), covered


def compare_axes(gt, candidate_axes, covered, primary_axis):
    A, B = np.asarray(gt['axes_base']), np.asarray(candidate_axes)
    valid = np.isfinite(A).all((1, 2))
    errors, triad = np.full((len(A), 3), np.nan), np.full(len(A), np.nan)
    errors[valid] = np.degrees(np.arccos(np.clip(abs(np.einsum('tij,tij->tj', A[valid], B[valid])), 0, 1)))
    delta = np.swapaxes(A[valid], 1, 2) @ B[valid]
    triad[valid] = np.min([np.degrees(Rotation.from_matrix(delta @ g).magnitude()) for g in D2], axis=0)
    masks = {'long_axis_deg': np.asarray(gt['long_valid'], bool), 'normal_axis_deg': np.asarray(gt['normal_valid'], bool), 'frame_D2_deg': np.asarray(gt['frame_valid'], bool)}
    values = {'long_axis_deg': errors[:, 0], 'normal_axis_deg': errors[:, 2], 'frame_D2_deg': triad}
    key = 'normal_axis_deg' if primary_axis == 2 else 'long_axis_deg'
    mask = masks[key]
    if not mask.any():
        raise ValueError('Registered REAL orientation has no valid primary frames')
    stats = {}
    for name, m in masks.items():
        if np.any(m & ~valid) or not np.isfinite(values[name][m]).all():
            raise ValueError('Nonfinite scored orientation')
        stats[name] = dict(mean_deg=float(np.mean(values[name][m])), median_deg=float(np.median(values[name][m])), p90_deg=float(np.percentile(values[name][m], 90)), count=int(m.sum())) if m.any() else None
    result = {name: value['mean_deg'] if value else None for name, value in stats.items()}
    result['long_axis_deg' if primary_axis == 2 else 'normal_axis_deg'] = None   # the non-primary axis is a diagnostic only
    result.update(version=ROTATION_VERSION, primary_metric=key, gt_coverage=float(mask.mean()), prediction_coverage=float(np.mean(covered[mask])),
                  valid_frames=int(mask.sum()), total_frames=len(mask), held_valid_frames=int(np.sum(mask & ~covered)), statistics=stats)
    arrays = dict(timestamp_ns=gt['timestamp_ns'], candidate_axes_base=B, axis_errors_deg=errors, frame_D2_deg=triad, long_valid=masks['long_axis_deg'],
                  normal_valid=masks['normal_axis_deg'], frame_valid=masks['frame_D2_deg'], prediction_in_clock=covered)
    return result, arrays


def attach_rotation(record, sample, times, output, *, vertices=None, faces=None, wxyz=None, surfaces=None):
    """Observable long/normal-axis and unsigned-frame errors for samples with registered REAL axes."""
    row = registry()['observable_rotation'].get(Path(sample).name)
    if row is None:
        return
    gt = dict(np.load(paths.DATA / row['gt'], allow_pickle=False))
    query = (gt['timestamp_ns'] - gt['timestamp_ns'][0]) * 1e-9
    if surfaces is not None:
        frames, previous = [], None
        for v in surfaces:
            _, _, A = mesh_frame(v, faces)
            if previous is not None:
                A = min((A @ g for g in D2), key=lambda a: np.linalg.norm(a - previous))
            frames.append(A)
            previous = A
        axes = np.asarray(frames)
    else:
        _, _, local = mesh_frame(vertices, faces)
        q = np.asarray(wxyz, float)
        if q.shape != (len(times), 4) or not np.isfinite(q).all() or np.any(np.linalg.norm(q, axis=1) < 1e-10):
            raise ValueError('Invalid executed quaternion trajectory')
        axes = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix() @ local
    sampled, covered = sample_axes(times, axes, query)
    result, arrays = compare_axes(gt, sampled, covered, row['primary_axis'])
    np.savez_compressed(output, **arrays)
    record['observable_rotation'] = result


# ---------------------------------------------------------------- visible scene and object model GT
def attach_visible_scene(record, sample, pkg, dest):
    """Scene CD against the registered REAL visible-surface scene GT."""
    from v2w.metrics import geometry as M
    from v2w.metrics.record import set_scene_result
    from .package import entity_geometry
    row = registry()['visible_scene'].get(Path(sample).name)
    if row is None:
        return
    pkg, dest = Path(pkg), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    a = read(pkg / 'scene.json')
    scene = dict(objects=[entity_geometry(e, pkg, dest) for e in a.get('objects', [])], props=[entity_geometry(e, pkg, dest) for e in a.get('props', [])], support=None)
    if a.get('table'):
        scene['props'].append(entity_geometry(a['table'], pkg, dest))
    d = paths.DATA / row['path']
    result = M.scene_cd_v5(scene, dest, scene_doc(d / 'gt_pkg/scene.json'), d / 'gt_pkg', read(d / 'camera.json'), read(d / 'roles.json'))
    result['gt_scope'] = row['scope']
    set_scene_result(record, result)


def object_model_metrics(candidate, candidate_dir, candidate_pose7, sample, dest):
    """Shape / size error of the candidate target against the registered provider object model, if any."""
    from v2w.metrics import geometry as M
    from .package import entity_geometry
    row = registry()['object_model'].get(Path(sample).name)
    if row is None:
        return None
    scene_path = paths.DATA / row['scene']
    entity = next(e for e in read(scene_path)['objects'] if e['name'] == row['entity'])
    Path(dest).mkdir(parents=True, exist_ok=True)
    ge = entity_geometry(entity, scene_path.parent, dest)
    result = M.object_level(candidate, candidate_dir, candidate_pose7, ge, dest, list(ge['pos']) + list(ge['quat']))
    result['gt_geometry_scope'] = 'provider simulation target model'
    return result
