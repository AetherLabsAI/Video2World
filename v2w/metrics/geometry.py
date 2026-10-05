"""Geometry primitives: entity meshes, object-level shape / size error, Scene CD, trajectory errors, support surfaces."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation as R

from video2sim.bench.metrics import sym_rot_err, chamfer, _cube_group

# ---------------------------------------------------------------- metric primitives
# Metric primitives (pure numpy / trimesh; no simulator).
#
# object_level(...)      object-level Chamfer + centre / size error for task-relevant entities (object, target, socket/container)
# relative_pose_err(...) object<->target relative pose error (trans cm / rot deg, symmetry-aware)
# rpe(...)               relative pose error over a stride (trajectory SHAPE, insensitive to a constant offset); symmetry-aware
# fscore(...)            3-D F-score of two point sets at tau (F@2cm / F@4cm)
# footprint_iou(...)     top-view (table-plane) footprint IoU of two point sets + end/start area-ratio error
# task_progress(...)     stage decomposition from simulator state (grasp -> lift -> transport -> release -> placed, ...),
#                        progress = achieved stages / total, with an order check

# ---------------------------------------------------------------- geometry helpers
def T_of(p):
    p = np.asarray(p, float); M = np.eye(4); M[:3, :3] = R.from_quat([p[4], p[5], p[6], p[3]]).as_matrix(); M[:3, 3] = p[:3]; return M


def entity_mesh(e: dict, pkg_dir: Path | None):
    k = e['kind']
    if k == '_scene_assembly':
        # Internal scoring representation only. Constituent poses are world poses;
        # the wrapper has identity pose. Merge BEFORE sampling to remove part-count bias.
        meshes = []
        for part in e['_scene_parts']:
            mesh = entity_mesh(part, pkg_dir).copy()
            mesh.apply_transform(T_of(part['pos'] + part['quat']))
            meshes.append(mesh)
        return trimesh.util.concatenate(meshes)
    if k == 'box': return trimesh.creation.box(extents=2 * np.asarray(e['half_size'], float))
    if k == 'cylinder': return trimesh.creation.cylinder(radius=float(e['half_size'][0]), height=2 * float(e['half_size'][2]))
    if k == 'sphere': return trimesh.creation.icosphere(radius=float(e['half_size'][0]) if 'half_size' in e else float(e.get('radius', 0.03)))
    if k == 'container':
        from video2sim.bench.silhouette import load_mesh; return load_mesh(e)
    if k == 'mesh':
        p = e.get('mesh_path') or e['collision_path']; p = Path(p) if Path(p).is_absolute() else Path(pkg_dir) / p
        if not p.exists() and Path(e.get('mesh_path') or e['collision_path']).exists(): p = Path(e.get('mesh_path') or e['collision_path'])   # scene_in_base_frame already prefixed a RELATIVE package dir
        m = trimesh.load(str(p), force='mesh'); m.apply_scale(float(e.get('scale', 1.0))); return m
    if k == 'library':
        from video2sim.bench.bridge_env import library_model_dir, _visual_file
        d = library_model_dir(e['library_id'])
        try: m = trimesh.load(_visual_file(d), force='mesh')
        except Exception: m = trimesh.load(str(d / 'collision.obj'), force='mesh')
        m.apply_scale(float(e.get('scale', 1.0))); return m
    raise ValueError(k)


def surface_points(mesh, pose7, n=4000, seed=0):
    p, _ = trimesh.sample.sample_surface(mesh, n, seed=seed); T = T_of(pose7); return p @ T[:3, :3].T + T[:3, 3]


def extents_sorted(mesh): return np.sort(np.asarray(mesh.bounding_box.extents, float))


def _kabsch(X, Y):
    """Rotation R and translation t minimising sum |R x + t - y|^2 over paired rows (no reflection)."""
    mx, my = X.mean(0), Y.mean(0); U, _, Vt = np.linalg.svd((X - mx).T @ (Y - my))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))]); Rm = Vt.T @ D @ U.T; return Rm, my - Rm @ mx


def shape_align(A, B, coarse=12, fine=40, keep=3):
    """Pose-free comparison of two surface samples: A is moved rigidly onto B (rotation + translation, NO scale, so size stays
    part of the shape) to minimise the bidirectional Chamfer. Both are centred on their surface centroids, then symmetric ICP
    (A->B and B->A nearest-neighbour pairs solved together) runs from each of the 24 axis-aligned rotations -- mesh axis
    conventions differ between packages -- the `keep` best are refined, and the lowest Chamfer wins.
    Returns (chamfer dict of the aligned sets, R, t) with A_aligned = (A - mean A) @ R.T + t in B's centred frame."""
    A0 = np.asarray(A, float) - np.mean(A, 0); B0 = np.asarray(B, float) - np.mean(B, 0); tb = cKDTree(B0)

    def icp(Rm, t, iters):
        for _ in range(iters):
            X = A0 @ Rm.T + t; j = tb.query(X)[1]; i = cKDTree(X).query(B0)[1]
            dR, dt = _kabsch(np.r_[X, X[i]], np.r_[B0[j], B0]); Rm, t = dR @ Rm, dR @ t + dt
        return Rm, t, chamfer(A0 @ Rm.T + t, B0)
    runs = sorted((icp(G.astype(float), np.zeros(3), coarse) for G in _cube_group()), key=lambda r: r[2]['chamfer_m'])[:keep]
    Rm, t, c = min((icp(Rm, t, fine) for Rm, t, _ in runs), key=lambda r: r[2]['chamfer_m'])
    return c, Rm, t


def object_level(cand_e: dict, cand_pkg: Path, cand_pose7, gt_e: dict, gt_dir: Path, gt_pose7, n=4000) -> dict:
    """Candidate entity vs GT entity. Shape CD (cm): bidirectional Chamfer after the best rigid alignment (position and
    orientation removed, scale kept) -- the benchmark's shape column. Also: Chamfer at the INITIAL poses (pose + shape, kept for
    reference), centroid-aligned Chamfer (orientation still in), the rotation the alignment needed, centre error, size error."""
    mc, mg = entity_mesh(cand_e, cand_pkg), entity_mesh(gt_e, gt_dir)
    A, B = surface_points(mc, cand_pose7, n), surface_points(mg, gt_pose7, n)
    c = chamfer(A, B); cc = chamfer(A - A.mean(0), B - B.mean(0))
    # Intrinsic shape must not depend on the submitted object's world pose.
    identity = [0, 0, 0, 1, 0, 0, 0]
    s, Rlocal, _ = shape_align(surface_points(mc, identity, n), surface_points(mg, identity, n))
    Rs = T_of(gt_pose7)[:3, :3] @ Rlocal @ T_of(cand_pose7)[:3, :3].T
    # Size uses independently fitted oriented bounding-box full lengths.  It is
    # independent of shape ICP, mesh origin, axis naming and external world pose.
    # Sorted lengths compare physical dimensions, not semantic width/height axes.
    _, ec = trimesh.bounds.oriented_bounds(mc, angle_digits=5)
    _, eg = trimesh.bounds.oriented_bounds(mg, angle_digits=5)
    ec, eg = np.sort(ec), np.sort(eg)
    return dict(size_protocol='intrinsic-obb-full-extents/2', shape_protocol='mesh-local-icp/2',
                size_axis_order='ascending full lengths; no semantic axis correspondence',
                shape_cd_cm=100 * s['chamfer_m'], shape_a_to_b_cm=100 * s['a_to_b_m'], shape_b_to_a_cm=100 * s['b_to_a_m'],
                shape_centred_cd_cm=100 * cc['chamfer_m'], shape_align_rot_deg=float(np.degrees(np.linalg.norm(R.from_matrix(Rs).as_rotvec()))),
                chamfer_cm=100 * c['chamfer_m'], a_to_b_cm=100 * c['a_to_b_m'], b_to_a_cm=100 * c['b_to_a_m'],
                center_err_cm=100 * float(np.linalg.norm(A.mean(0) - B.mean(0))), origin_err_cm=100 * float(np.linalg.norm(np.asarray(cand_pose7[:3]) - np.asarray(gt_pose7[:3]))),   # centre = surface centroid (independent of each mesh's origin convention)
                size_err_cm=100 * float(np.linalg.norm(ec - eg)),
                size_cand_cm=(100 * ec).tolist(), size_gt_cm=(100 * eg).tolist())


def symmetry_axis(mesh, n=3000, seed=0):
    """Which local axis the mesh is a body of revolution about: self-Chamfer after a 90 deg turn about each axis (mm);
    returns (axis 'x'|'y'|'z' or None when no axis is clearly symmetric, the three residuals)."""
    P, _ = trimesh.sample.sample_surface(mesh, n, seed=seed); P = P - P.mean(0); t = cKDTree(P)
    res = {ax: float(t.query(R.from_euler(ax, 90, degrees=True).apply(P))[0].mean() * 1000) for ax in 'xyz'}
    best = min(res, key=res.get); others = sorted(v for k, v in res.items() if k != best)
    return (best if res[best] < 0.5 * others[0] else None), res


def canonical_transform(cand_e: dict, cand_pkg: Path, gt_e: dict, gt_dir: Path, n=3000, mode='object'):
    """4x4 A mapping candidate-mesh-frame coordinates onto the GT mesh's frame (x_gt = A x_c). A candidate part delivered as its
    own primitive / mesh has its own origin and axis convention (agent parts are often symmetric about local z,
    the GT meshes about local y; primitive origins sit at the centre, the GT origin is the CAD origin). With A^-1 folded into a
    pose, T_canon = T_cand @ A^-1 is the pose the GT mesh would have in that place, so the GT symmetry axis, origin and the
    demonstrated relative pose all apply.
      mode='object': rotation = the candidate's own symmetry axis (symmetry_axis) mapped onto the GT's, sign chosen by the lower
                     Chamfer of the two candidates (up / down); falls back to the Shape-CD ICP when the part has no clear axis.
                     translation = surface-centroid to surface-centroid.
      mode='target': translation only (centroid to centroid); the receiving part keeps its delivered orientation.
    Identity when the candidate mesh is the GT mesh file."""
    try:
        if cand_e.get('kind') == 'mesh' and gt_e.get('kind') == 'mesh' and Path(cand_e.get('mesh_path', 'a')).name == Path(gt_e.get('mesh_path', 'b')).name and Path(cand_e.get('mesh_path', '')).stat().st_size == Path(gt_e.get('mesh_path', '')).stat().st_size:
            return np.eye(4)
    except Exception: pass
    mc, mg = entity_mesh(cand_e, cand_pkg), entity_mesh(gt_e, gt_dir)
    Pc, _ = trimesh.sample.sample_surface(mc, n, seed=0); Pg, _ = trimesh.sample.sample_surface(mg, n, seed=0); mc_, mg_ = Pc.mean(0), Pg.mean(0)
    A = np.eye(4)
    if mode == 'target':
        A[:3, 3] = mg_ - mc_; return A
    ax_c, _ = symmetry_axis(mc); ax_g, _ = symmetry_axis(mg); ax_g = ax_g or gt_e.get('symmetry_axis', 'y')
    if ax_c is None:
        _, Rm, t = shape_align(Pc, Pg); A[:3, :3] = Rm; A[:3, 3] = t + mg_ - Rm @ mc_; return A
    ec = np.eye(3)['xyz'.index(ax_c)]; eg = np.eye(3)['xyz'.index(ax_g)]
    def rot_to(u, v):   # rotation taking unit u to unit v (minimal)
        c = float(np.clip(u @ v, -1, 1)); w = np.cross(u, v); s_ = np.linalg.norm(w)
        if s_ < 1e-9: return np.eye(3) if c > 0 else R.from_rotvec(np.pi * (np.eye(3)[np.argmin(np.abs(u))] - u * (np.eye(3)[np.argmin(np.abs(u))] @ u)) / np.linalg.norm(np.eye(3)[np.argmin(np.abs(u))] - u * (np.eye(3)[np.argmin(np.abs(u))] @ u))).as_matrix()
        return R.from_rotvec(w / s_ * np.arccos(c)).as_matrix()
    best = None
    for sign in (1, -1):
        Rm = rot_to(ec, sign * eg); c = chamfer((Pc - mc_) @ Rm.T, Pg - mg_)['chamfer_m']
        if best is None or c < best[0]: best = (c, Rm)
    Rm = best[1]
    # translation: match the part's BOTTOM point on its axis (the end that seats in / rests on the receiving part), not the
    # centroid -- a candidate part of a different length still seats at the same place
    eg_v = np.eye(3)['xyz'.index(ax_g)]; Qc = (Pc - mc_) @ Rm.T; bc = eg_v * float(np.percentile(Qc @ eg_v, 1)); bg = eg_v * float(np.percentile((Pg - mg_) @ eg_v, 1))
    A[:3, :3] = Rm; A[:3, 3] = mg_ + bg - bc - Rm @ mc_; return A


def canon_pose7(pose7, A):
    T = T_of(pose7) @ np.linalg.inv(A); q = R.from_matrix(T[:3, :3]).as_quat(); return np.r_[T[:3, 3], q[3], q[0], q[1], q[2]]


def canon_traj(traj, A):
    return np.array([canon_pose7(p, A) for p in np.asarray(traj, float)])


def relative_pose_err(obj_c, tgt_c, obj_g, tgt_g, symmetry=None, axis='z') -> dict:
    """Error of the object->target relative pose (candidate vs GT): translation of the target in the object frame (cm) and
    the symmetry-aware angle between the two relative rotations (deg)."""
    Rc = np.linalg.inv(T_of(obj_c)) @ T_of(tgt_c); Rg = np.linalg.inv(T_of(obj_g)) @ T_of(tgt_g)
    qc = R.from_matrix(Rc[:3, :3]).as_quat(); qg = R.from_matrix(Rg[:3, :3]).as_quat()
    return dict(rel_trans_err_cm=100 * float(np.linalg.norm(Rc[:3, 3] - Rg[:3, 3])),
                rel_rot_err_deg=float(sym_rot_err(np.r_[qc[3], qc[:3]], np.r_[qg[3], qg[:3]], symmetry, axis)))


def terminal_relative(obj7, tgt7, gt_rel, axis='y') -> dict:
    """Rel pose v2: the part's TERMINAL pose expressed in the receiving part's frame, against the
    demonstrated terminal relative pose `gt_rel` (4x4): translation error (cm) in the target frame and the angle between the
    part's symmetry axis and the demonstrated one (deg; spin about the axis is free)."""
    rel = np.linalg.inv(T_of(tgt7)) @ T_of(obj7); ref = np.asarray(gt_rel, float)
    d = 100 * float(np.linalg.norm(rel[:3, 3] - ref[:3, 3]))
    if axis is None:   # non-symmetric part: full rotation error
        rot = float(np.degrees(np.linalg.norm(R.from_matrix(ref[:3, :3].T @ rel[:3, :3]).as_rotvec())))
    else:
        i = 'xyz'.index(axis); rot = float(np.degrees(np.arccos(np.clip(float(rel[:3, i] @ ref[:3, i]), -1, 1))))
    out = dict(rel_trans_err_cm=min(d, CAP_CM), rel_trans_err_raw_cm=d, left_workspace=bool(d > CAP_CM), rel_rot_err_deg=rot, frame='target', stage='terminal')
    if axis is None: out['rot_metric'] = 'full'
    return out


CAP_CM = 100.0   # a part that left the workspace (fell off the bench, out of the world) saturates at 1 m so one fall cannot dominate a family mean


def ape_capped(traj_a, traj_b, cap_cm=CAP_CM) -> dict:
    """Translation APE (cm) between two (T,7) trajectories on the same clock, each frame's error saturated at cap_cm;
    `left_workspace` = some frame exceeded the cap (the raw max is reported alongside)."""
    n = min(len(traj_a), len(traj_b)); a, b = _resample(traj_a, n), _resample(traj_b, n)
    t = np.linalg.norm(a[:, :3] - b[:, :3], axis=1) * 100; tc = np.minimum(t, cap_cm)
    return dict(trans_cm=float(tc.mean()), trans_final_cm=float(tc[-1]), trans_max_cm=float(tc.max()), trans_max_raw_cm=float(t.max()), left_workspace=bool(t.max() > cap_cm), cap_cm=cap_cm)


# ---------------------------------------------------------------- trajectory metrics
def _resample(x, n):
    x = np.asarray(x, float)
    if len(x) == n: return x
    idx = np.linspace(0, len(x) - 1, n).round().astype(int); return x[idx]


def rpe(traj_a, traj_b, delta=5, symmetry=None, axis='z', *, time_s=None, valid=None, lag_s=.2) -> dict:
    """Base-axis displacement-increment error (cm), capped per interval at 100 cm.

    Inputs must already share a clock. With timestamps, select the nearest later
    state to t + lag_s (ties choose earlier); never stretch or compress a stream.
    Rotation retains the existing relative-rotation diagnostic definition.
    """
    a, b = np.asarray(traj_a, float), np.asarray(traj_b, float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 7 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('RPE requires paired finite T x 7 trajectories on the same clock')
    n = len(a)
    if type(delta) is not int or delta < 1: raise ValueError('RPE delta must be a positive integer')
    times = None
    if time_s is None:
        pairs = [(k, k + delta) for k in range(n - delta)]
    else:
        times = np.asarray(time_s, float)
        if times.shape != (n,) or not np.isfinite(times).all() or (np.diff(times) <= 0).any() or not np.isfinite(lag_s) or lag_s <= 0:
            raise ValueError('invalid RPE clock/lag')
        pairs = []
        for k in range(n - 1):
            target = times[k] + lag_s
            if target > times[-1] + 1e-10: continue
            j = int(np.searchsorted(times, target))
            candidates = [i for i in (j-1, j) if k < i < n]
            if candidates: pairs.append((k, min(candidates, key=lambda i: (round(abs(times[i]-target), 12), i))))
    if valid is not None:
        mask = np.asarray(valid, bool)
        if mask.shape != (n,): raise ValueError('invalid RPE validity mask')
        pairs = [(k, j) for k, j in pairs if mask[k:j+1].all()]
    raw, rotations = [], []
    for k, j in pairs:
        raw.append(float(np.linalg.norm((a[j, :3]-a[k, :3]) - (b[j, :3]-b[k, :3])) * 100))
        Da = np.linalg.inv(T_of(a[k])) @ T_of(a[j]); Db = np.linalg.inv(T_of(b[k])) @ T_of(b[j])
        qa = R.from_matrix(Da[:3, :3]).as_quat(); qb = R.from_matrix(Db[:3, :3]).as_quat()
        rotations.append(sym_rot_err(np.r_[qa[3], qa[:3]], np.r_[qb[3], qb[:3]], symmetry, axis))
    raw = np.asarray(raw); capped = np.minimum(raw, CAP_CM); rr = np.asarray(rotations)
    avg = lambda x: float(np.mean(x)) if len(x) else None
    p90 = lambda x: float(np.percentile(x, 90)) if len(x) else None
    lags = [float(times[j]-times[k]) for k,j in pairs] if times is not None else None
    strides = sorted({j-k for k,j in pairs})
    return dict(protocol='base-displacement-rpe/1', coordinate_frame='common fixed base axes',
                rotation_protocol='legacy relative-rotation diagnostic; unchanged', cap_cm=CAP_CM,
                trans_rpe_cm=avg(capped), trans_rpe_p90_cm=p90(capped),
                trans_rpe_raw_cm=avg(raw), trans_rpe_raw_p90_cm=p90(raw),
                trans_per_interval_cm=capped.tolist(), trans_raw_per_interval_cm=raw.tolist(),
                rot_rpe_deg=avg(rr), rot_rpe_p90_deg=p90(rr), rot_per_interval_deg=rr.tolist(),
                n=len(pairs), delta=(strides[0] if len(strides)==1 else None),
                start_frame_indices=[k for k,j in pairs], end_frame_indices=[j for k,j in pairs],
                requested_delta_seconds=lag_s if times is not None else None,
                delta_seconds=avg(lags) if lags is not None else None, interval_seconds=lags)


# ---------------------------------------------------------------- deformable
def fscore(P, Q, tau) -> dict:
    """3-D F-score at tau (m): precision = frac of P within tau of Q, recall = frac of Q within tau of P."""
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    if len(P) == 0 or len(Q) == 0: return dict(precision=float('nan'), recall=float('nan'), f=float('nan'))
    dp = cKDTree(Q).query(P, k=1)[0]; dq = cKDTree(P).query(Q, k=1)[0]
    pr, rc = float((dp <= tau).mean()), float((dq <= tau).mean()); f = 2 * pr * rc / max(pr + rc, 1e-9)
    return dict(precision=pr, recall=rc, f=f)


def footprint_iou(P, Q, res=0.01, pad=0.05) -> dict:
    """Top-view (xy) footprint IoU of two point sets rasterised at `res`, + areas."""
    P, Q = np.asarray(P, float)[:, :2], np.asarray(Q, float)[:, :2]
    if len(P) == 0 or len(Q) == 0: return dict(iou=float('nan'))
    lo = np.minimum(P.min(0), Q.min(0)) - pad; hi = np.maximum(P.max(0), Q.max(0)) + pad; W, H = (np.ceil((hi - lo) / res)).astype(int) + 1
    def rast(X):
        g = np.zeros((H, W), bool); ij = ((X - lo) / res).astype(int); g[ij[:, 1], ij[:, 0]] = True
        import cv2; return cv2.morphologyEx(g.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    a, b = rast(P), rast(Q); inter, uni = (a & b).sum(), (a | b).sum()
    return dict(iou=float(inter / max(uni, 1)), area_a_m2=float(a.sum() * res * res), area_b_m2=float(b.sum() * res * res))


# ---------------------------------------------------------------- task progress
def _first(cond):
    idx = np.flatnonzero(np.asarray(cond, bool)); return int(idx[0]) if len(idx) else None


def task_progress(kind: str, d: dict) -> dict:
    """Stage decomposition from simulator state. `d` carries what the family's rollout recorded.
    Returns stages [{name, done, frame}], progress in [0,1], order_ok. A stage counts only if achieved AND not earlier than
    the previous achieved stage (order check); progress = achieved-in-order / total."""
    if kind.startswith('rigid_'):
        from v2w.metrics.task import task_progress as ordered_progress
        return ordered_progress(kind, d)
    st = []
    if kind in ('drawer_close', 'drawer_open', 'press'):
        q = np.asarray(d['q'], float); T = len(q); q0 = float(q[0])
        if kind == 'drawer_close':
            mv = _first(q <= q0 - 0.01); st.append(dict(name='contact_move', done=mv is not None, frame=mv))
            half = _first(q <= 0.5 * q0); st.append(dict(name='half_closed', done=half is not None, frame=half))
            fin = bool(d.get('phi_success')); st.append(dict(name='closed_settled', done=fin, frame=T - 1 if fin else None))
        elif kind == 'drawer_open':
            hk = d.get('hook_frame'); mv = _first(q >= q0 + 0.01)
            st.append(dict(name='hook', done=(hk is not None) or (mv is not None), frame=hk if hk is not None else mv))
            st.append(dict(name='pull_move', done=mv is not None, frame=mv))
            half = _first(q >= q0 + 0.5 * float(d.get('real_final_q', 0.2))); st.append(dict(name='pulled_half', done=half is not None, frame=half))
            fin = bool(d.get('phi_success')); st.append(dict(name='open_settled', done=fin, frame=T - 1 if fin else None))
        else:
            tv = float(d.get('travel', 0.02)); c = _first(q >= 0.2 * tv); st.append(dict(name='contact', done=c is not None, frame=c))
            p = _first(q >= 0.7 * tv); st.append(dict(name='pressed', done=p is not None, frame=p))
            fin = bool(d.get('phi_success')); st.append(dict(name='latched', done=fin, frame=T - 1 if fin else None))
    elif kind == 'towel':
        cen = d.get('centroid_traj')
        if cen is not None:
            cen = np.asarray(cen, float); disp = np.linalg.norm(cen[:, :2] - cen[0, :2], axis=1)
            mv = _first(disp >= 0.03); st.append(dict(name='pinch_move', done=mv is not None, frame=mv))
            far = _first(disp >= 0.5 * float(d.get('real_disp', 0.06))); st.append(dict(name='fold_half', done=far is not None, frame=far))
        else:
            st.append(dict(name='pinch_move', done=bool(d.get('moved')), frame=None)); st.append(dict(name='fold_half', done=bool(d.get('moved')), frame=None))
        fin = bool(d.get('phi_success')); st.append(dict(name='final_config', done=fin, frame=None))
    else:
        raise ValueError(kind)
    # order check + progress (count achieved stages in order; stop at the first not-achieved)
    n_ok, last, order_ok = 0, -1, True; TOL = 5   # frames: stages that legitimately coincide (release vs settle-into-socket) are not 'out of order'
    for s in st:
        if not s['done']: break
        f = s['frame']
        if f is not None and f < last - TOL: order_ok = False; break
        if f is not None: last = f
        n_ok += 1
    return dict(kind=kind, stages=st, achieved=n_ok, total=len(st), progress=n_ok / max(len(st), 1), order_ok=order_ok)


# ---------------------------------------------------------------- Scene CD v2
SCENE_DENSITY = 1.0e4      # surface samples per m^2 (1 per cm^2): Chamfer becomes a surface integral, not a per-entity vote
SCENE_N_MIN, SCENE_N_MAX = 300, 40000


def _canonical_scene_mesh(mesh):
    """Canonical triangle order for Scene surface sampling only.

    Ignore vertex/face indexing, winding and material grouping. Keep the exact
    triangle coordinates and triangle multiplicity; do not quantize geometry,
    change tessellation or alter other metrics' mesh/canonical-frame semantics.
    """
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    if not np.isfinite(triangles).all():
        raise ValueError('non-finite scene mesh')
    if not len(triangles):
        return trimesh.Trimesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=int), process=False)
    order = np.lexsort((triangles[:, :, 2], triangles[:, :, 1], triangles[:, :, 0]), axis=1)
    triangles = np.take_along_axis(triangles, order[:, :, None], axis=1)
    flat = triangles.reshape(-1, 9)
    order = np.lexsort(tuple(flat[:, k] for k in range(8, -1, -1)))
    vertices = triangles[order].reshape(-1, 3)
    return trimesh.Trimesh(vertices=vertices, faces=np.arange(len(vertices)).reshape(-1, 3), process=False)


def _scene_samples(scene, pkg_dir, density, nmin, nmax, seed=0, canonical_surface=False):
    import trimesh
    from scipy.spatial.transform import Rotation as R
    pts = []
    for e in (scene.get('objects') or []) + (scene.get('props') or []):
        m = entity_mesh(e, pkg_dir)
        if canonical_surface: m = _canonical_scene_mesh(m)
        n = int(np.clip(m.area * density, nmin, nmax))
        p, _ = trimesh.sample.sample_surface(m, n, seed=seed); q = np.asarray(e.get('quat', (1, 0, 0, 0)), float)
        pts.append(p @ R.from_quat(np.r_[q[1:], q[0]]).as_matrix().T + np.asarray(e['pos'], float))
    if scene.get('support'):
        sp = scene['support']; cx, cy = sp.get('center', (0.3, 0.0)); lx, ly = sp.get('size', (1.0, 1.0)); n = int(np.clip(lx * ly * density, nmin, nmax))
        xy = np.random.default_rng(seed).uniform([-lx / 2, -ly / 2], [lx / 2, ly / 2], (n, 2)); pts.append(np.c_[xy + [cx, cy], np.full(n, float(sp['z']))])
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def _scene_triangles(scene, pkg_dir):
    """All triangles of the scene in the base frame, (N,3,3)."""
    from scipy.spatial.transform import Rotation as R
    tris = []
    for e in (scene.get('objects') or []) + (scene.get('props') or []):
        m = entity_mesh(e, pkg_dir); q = np.asarray(e.get('quat', (1, 0, 0, 0)), float); Rm = R.from_quat(np.r_[q[1:], q[0]]).as_matrix()
        V = np.asarray(m.vertices, float) @ Rm.T + np.asarray(e['pos'], float); tris.append(V[np.asarray(m.faces)])
    if scene.get('support'):
        sp = scene['support']; cx, cy = sp.get('center', (0.3, 0.0)); lx, ly = sp.get('size', (1.0, 1.0)); z = float(sp['z'])
        a, b, c, d = [cx - lx / 2, cy - ly / 2, z], [cx + lx / 2, cy - ly / 2, z], [cx + lx / 2, cy + ly / 2, z], [cx - lx / 2, cy + ly / 2, z]
        tris.append(np.array([[a, b, c], [a, c, d]], float))
    return np.concatenate(tris) if tris else np.zeros((0, 3, 3))


def depth_image(scene, pkg_dir, cam, img_hw=(448, 448)):
    """Software z-buffer of the scene through the hidden camera (perspective-correct, nearest surface per pixel; +inf = empty).
    Triangles smaller than a pixel are splatted at their vertices / centroid; larger ones are filled with edge functions."""
    K = np.asarray(cam['intrinsics'], float); T = np.linalg.inv(np.asarray(cam['extrinsics_base_cam'], float)); H, W = img_hw
    tri = _scene_triangles(scene, pkg_dir); n = len(tri); Z = np.full((H, W), np.inf)
    if n == 0: return Z
    Pc = tri.reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]; z = Pc[:, 2]; uv = (Pc @ K.T)[:, :2] / np.maximum(z[:, None], 1e-6)
    uv, z = uv.reshape(n, 3, 2), z.reshape(n, 3); ok = (z > 0.05).all(1); uv, z = uv[ok], z[ok]
    lo = np.floor(uv.min(1)).astype(int); hi = np.ceil(uv.max(1)).astype(int); span = (hi - lo)
    small = (span[:, 0] <= 2) & (span[:, 1] <= 2)
    # splat small triangles (vertices + centroid)
    pts = np.concatenate([uv[small].reshape(-1, 2), uv[small].mean(1)]); zz = np.concatenate([z[small].reshape(-1), z[small].mean(1)])
    px = np.floor(pts).astype(int); m = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
    np.minimum.at(Z, (px[m, 1], px[m, 0]), zz[m])
    # fill the rest
    for k in np.flatnonzero(~small):
        (x0, y0), (x1, y1), (x2, y2) = uv[k]; xa, xb = max(lo[k, 0], 0), min(hi[k, 0], W - 1); ya, yb = max(lo[k, 1], 0), min(hi[k, 1], H - 1)
        if xa > xb or ya > yb: continue
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-9: continue
        gx, gy = np.meshgrid(np.arange(xa, xb + 1) + 0.5, np.arange(ya, yb + 1) + 0.5)
        w0 = ((x1 - gx) * (y2 - gy) - (x2 - gx) * (y1 - gy)) / area; w1 = ((x2 - gx) * (y0 - gy) - (x0 - gx) * (y2 - gy)) / area; w2 = 1 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any(): continue
        inv = w0 / z[k, 0] + w1 / z[k, 1] + w2 / z[k, 2]; d = 1.0 / np.maximum(inv, 1e-9)
        sub = Z[ya:yb + 1, xa:xb + 1]; np.minimum(sub, np.where(inside, d, np.inf), out=sub)
    return Z


def scene_points_v2(scene: dict, pkg_dir, cam: dict | None = None, roi=0.9, density=SCENE_DENSITY, img_hw=(448, 448), rows=(98, 350), ztol=0.01, Z=None, nmax=None, canonical_surface=False):
    """Area-proportional surface samples of every entity (+ the support plane) at the initial pose, restricted to the workspace
    (r <= roi from the robot base) AND to what the hidden camera actually sees: inside the letterboxed image content rows and
    not occluded (a rasterized z-buffer of the same scene; a sample survives if its depth is within 1 cm + the local depth
    gradient of the surface the camera sees at that pixel, so the underside of a table box, the back of a lamp base, anything
    behind another part is dropped). Fixes two things in the upstream sampler: 4000 points per entity whatever its size (a 1.3 m
    table gets the same weight as a 3 cm tag, and a correct table modelled as a box instead of a plane scores 0.4-1.3 cm), and
    scoring table area no video frame shows."""
    P = _scene_samples(scene, pkg_dir, density, SCENE_N_MIN, nmax or SCENE_N_MAX, canonical_surface=canonical_surface)
    if not len(P): return P
    keep = np.linalg.norm(P[:, :2], axis=1) <= roi
    if cam is not None:
        K = np.asarray(cam['intrinsics'], float); T = np.linalg.inv(np.asarray(cam['extrinsics_base_cam'], float))
        Pc = P @ T[:3, :3].T + T[:3, 3]; z = Pc[:, 2]; uv = (Pc @ K.T)[:, :2] / np.maximum(z[:, None], 1e-6)
        keep &= (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < img_hw[1]) & (uv[:, 1] >= rows[0]) & (uv[:, 1] < rows[1])
        Z = depth_image(scene, pkg_dir, cam, img_hw) if Z is None else Z; Zf = np.where(np.isfinite(Z), Z, np.nan)
        from scipy.ndimage import maximum_filter, minimum_filter
        Zmax = maximum_filter(np.nan_to_num(Zf, nan=-np.inf), 3); Zmin = minimum_filter(np.nan_to_num(Zf, nan=np.inf), 3)
        idx = np.flatnonzero(keep); px = np.floor(uv[idx]).astype(int); zs = Z[px[:, 1], px[:, 0]]
        grad = np.where(np.isfinite(Zmax[px[:, 1], px[:, 0]]) & np.isfinite(Zmin[px[:, 1], px[:, 0]]), Zmax[px[:, 1], px[:, 0]] - Zmin[px[:, 1], px[:, 0]], 0.0)
        keep[idx] = ~np.isfinite(zs) | (z[idx] <= zs + ztol + grad)
    return P[keep]


SCENE_DENSE_MULT = 64   # the distance target is the other scene's surface sampled 64x denser (1.25 mm spacing): point-to-surface
                        # distance with a ~0.6 mm floor instead of the ~0.5 cm floor of two equally sparse clouds


def scene_cd_v2(cand_scene: dict, cand_pkg, gt_scene: dict, gt_dir, cam: dict | None) -> dict:
    """Scene CD v2 (cm) = mean_{a in visible surface of the candidate} dist(a, visible GT surface) + the same the other way.
    Surfaces are sampled uniformly by area (1 / cm^2 on the measured side, 64 / cm^2 on the distance-target side), restricted to
    the workspace (r <= 0.9 m) and to what the hidden camera sees (frustum, letterbox rows, rasterized z-buffer occlusion)."""
    Zc = depth_image(cand_scene, cand_pkg, cam) if cam is not None else None; Zg = depth_image(gt_scene, gt_dir, cam) if cam is not None else None
    A1 = scene_points_v2(cand_scene, cand_pkg, cam, Z=Zc); B1 = scene_points_v2(gt_scene, gt_dir, cam, Z=Zg)
    A64 = scene_points_v2(cand_scene, cand_pkg, cam, density=SCENE_DENSITY * SCENE_DENSE_MULT, Z=Zc, nmax=SCENE_N_MAX * SCENE_DENSE_MULT)
    B64 = scene_points_v2(gt_scene, gt_dir, cam, density=SCENE_DENSITY * SCENE_DENSE_MULT, Z=Zg, nmax=SCENE_N_MAX * SCENE_DENSE_MULT)
    if not (len(A1) and len(B1) and len(A64) and len(B64)): return dict(scene_cd_cm=float('nan'), a_to_b_cm=float('nan'), b_to_a_cm=float('nan'), n_cand=int(len(A1)), n_gt=int(len(B1)), version='v2')
    dab = cKDTree(B64).query(A1)[0]; dba = cKDTree(A64).query(B1)[0]
    return dict(scene_cd_cm=100 * float(dab.mean() + dba.mean()), a_to_b_cm=100 * float(dab.mean()), b_to_a_cm=100 * float(dba.mean()), a_to_b_p95_cm=100 * float(np.percentile(dab, 95)), b_to_a_p95_cm=100 * float(np.percentile(dba, 95)),
                n_cand=int(len(A1)), n_gt=int(len(B1)), version='v2: area-proportional sampling, camera-visible (frustum + rasterized z-buffer) workspace, point-to-dense-surface distance')

# ----------------------------------------------------------------------------------------------------------------------------------
# Scene CD v3: the v2 number on the hand track was 91-97 % the GT SUPPORT
# PLANE (10k visible samples of table area vs 1-2k of every object together), i.e. a table-extent / depth metric, not scene fidelity.
# v3 changes four things, all reported separately so the old number stays comparable (`scene_cd_v2_cm`):
#   1. the table is scored on its own (`table_depth_err_cm`: candidate table surface vs the GT support plane) and excluded from the CD;
#   2. per-ENTITY aggregation with role weights (manipulated object 2, receiving target 2, other props 1) instead of one area-weighted cloud;
#   3. robust statistics: each entity contributes its MEDIAN point-to-surface distance; a GT entity whose median exceeds MISSING_CM (or that
#      no candidate surface covers) is counted in `missing_fraction` (and a candidate entity far from every GT surface in `extra_fraction`);
#   4. a trimmed-ICP similarity fit (rotation + translation + uniform scale, fitted on every visible surface incl. the table) is applied to the
#      candidate before the CD; the fitted transform is reported as `alignment` (trans_cm / rot_deg / scale) = the monocular frame error, and
#      `scene_cd_unaligned_cm` keeps the un-aligned value.
# The camera image size and letterbox rows come from the sample's camera.json (width / height, `letterbox.pad_y` or the "letterbox pad N" note),
# so every track is measured through its own picture (v2 hard-coded the FurnitureBench 448x448 / rows 98-350, wrong for the 640x640 DROID view).
ROLE_WEIGHTS = dict(object=2.0, target=2.0, other=1.0)
MISSING_CM = 5.0
TABLE_KEYS = ('table', 'desk', 'tabletop', 'support', 'floor', 'ground', 'counter', 'placemat')
ICP_TRIM, ICP_ITERS = 0.7, 30
SCALE_RANGE = (0.7, 1.4)   # similarity scale bounds of the alignment fit


def image_rows(cam, img_hw=None):
    """(img_hw, rows) of the hidden camera picture: from camera.json width/height and its letterbox (explicit `letterbox.pad_y` or the
    'letterbox pad N' phrase of the calibration note); defaults = the FurnitureBench 448x448 / rows 98-350 when the json says nothing."""
    import re
    if cam is None: return (img_hw or (448, 448)), (98, 350)
    H, W = int(cam.get('height', img_hw[0] if img_hw else 448)), int(cam.get('width', img_hw[1] if img_hw else 448)); hw = (H, W)
    lb = cam.get('letterbox') or {}
    if lb: pad = int(lb.get('pad_y', 0)); return hw, (pad, H - pad)
    m = re.search(r'letterbox pad (\d+)', str(cam.get('note', '')) + ' ' + str(cam.get('source', '')))
    if m: pad = int(m.group(1)); return hw, (pad, H - pad)
    return hw, (0, H)


TABLE_MIN_SPAN_M, TABLE_MAX_THICK_M = 0.45, 0.12


def _is_table(e, pkg_dir=None, roles=None, geometric=True):
    """Is this entity the table / support surface? (the first cut matched 'table' as a substring, which swallowed
    'food_vegetables' and the furniture 'table_leg' / 'table_top' -- the manipulated part and the receiving part -- out of the CD.)
    Rule: an entity carrying a GT role (object / target) is never a table; otherwise the name must contain a table TOKEN (split on
    non-alphanumerics: 'table', 'desk', 'tabletop', ...; 'table_top' / 'table_leg' are furniture parts, not tables) AND, when the
    mesh is available, the geometry must be a broad thin slab (two extents >= TABLE_MIN_SPAN_M, the third <= TABLE_MAX_THICK_M)."""
    import re
    n = str(e.get('name', '')).lower()
    if roles and n in (str(roles.get('object', '')).lower(), str(roles.get('target', '')).lower()): return False
    toks = [t for t in re.split(r'[^a-z0-9]+', n) if t]
    if not any(t in TABLE_KEYS for t in toks) or any(t in ('top', 'leg', 'legs', 'lamp', 'cloth', 'tag') for t in toks): return False
    if pkg_dir is None or not geometric: return True
    try:
        m = entity_mesh(e, pkg_dir); ext = np.sort(np.asarray(m.bounding_box.extents, float))
        return bool(ext[1] >= TABLE_MIN_SPAN_M and ext[0] <= TABLE_MAX_THICK_M)
    except Exception: return True


def _entity_points(scene, pkg_dir, e, cam, hw, rows, Z, density=SCENE_DENSITY, nmax=SCENE_N_MAX, canonical_surface=False):
    sub = dict(support=None, props=[e], objects=[])
    return scene_points_v2(sub, pkg_dir, cam, density=density, img_hw=hw, rows=rows, Z=Z, nmax=nmax, canonical_surface=canonical_surface)


def _support_points(scene, pkg_dir, cam, hw, rows, Z, density=SCENE_DENSITY, nmax=SCENE_N_MAX, canonical_surface=False):
    if not scene.get('support'): return np.zeros((0, 3))
    sub = dict(support=scene['support'], props=[], objects=[])
    return scene_points_v2(sub, pkg_dir, cam, density=density, img_hw=hw, rows=rows, Z=Z, nmax=nmax, canonical_surface=canonical_surface)


def similarity_fit(A, B):
    """Umeyama: s, R, t minimising |s R a + t - b| over paired rows."""
    ma, mb = A.mean(0), B.mean(0); Ac, Bc = A - ma, B - mb
    H = Ac.T @ Bc / len(A); U, S, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(U @ Vt)); D = np.diag([1, 1, d])
    R = (U @ D @ Vt).T; var_a = (Ac ** 2).sum() / len(A); s = float((S * np.diag(D)).sum() / max(var_a, 1e-12)); t = mb - s * R @ ma
    return s, R, t


def trimmed_icp_similarity(A, Btree, trim=ICP_TRIM, iters=ICP_ITERS, scale_range=None):
    """similarity transform (s, R, t) aligning point set A onto the surface behind KD-tree Btree: nearest-neighbour pairs, keep the closest
    `trim` fraction, Umeyama, repeat. Deterministic; scale clamped to [0.5, 2]."""
    scale_range = scale_range or SCALE_RANGE; s, R, t = 1.0, np.eye(3), np.zeros(3); X = A.copy()
    for _ in range(iters):
        d, j = Btree.query(X); k = max(int(len(X) * trim), 10); sel = np.argsort(d)[:k]
        s_, R_, t_ = similarity_fit(X[sel], Btree.data[j[sel]])
        s_ = float(np.clip(s_, scale_range[0] / max(s, 1e-6), scale_range[1] / max(s, 1e-6)))   # a badly wrong scene otherwise collapses onto the GT cloud
        X = s_ * X @ R_.T + t_; s, R, t = s * s_, R_ @ R, s_ * R_ @ t + t_
        if abs(s_ - 1) < 1e-4 and np.linalg.norm(t_) < 1e-4 and abs(np.trace(R_) - 3) < 1e-6: break
    return s, R, t


def scene_cd_v3(cand_scene: dict, cand_pkg, gt_scene: dict, gt_dir, cam: dict | None, roles: dict | None = None, img_hw=None, rows=None) -> dict:
    """Scene CD v3, see the block comment above. `roles` = {'object': gt manipulated entity name, 'target': gt receiving entity name}
    (names of the GT scene; the candidate side is unweighted). Returns a dict whose `scene_cd_cm` is the headline number."""
    hw, rws = image_rows(cam, img_hw)
    if img_hw is not None: hw = tuple(img_hw)
    if rows is not None: rws = tuple(rows)
    roles = roles or {}
    Zc = depth_image(cand_scene, cand_pkg, cam, hw) if cam is not None else None; Zg = depth_image(gt_scene, gt_dir, cam, hw) if cam is not None else None
    dense = dict(density=SCENE_DENSITY * SCENE_DENSE_MULT, nmax=SCENE_N_MAX * SCENE_DENSE_MULT)
    # a candidate's table is whatever it NAMES a table (agents deliver thin boxes of any size); the GT side is role-aware and geometric
    c_ents = [(e, 'table' if _is_table(e, cand_pkg, geometric=False) else 'entity') for e in (cand_scene.get('objects') or []) + (cand_scene.get('props') or [])]
    g_ents = [(e, 'table' if _is_table(e, gt_dir, roles) else 'entity') for e in (gt_scene.get('objects') or []) + (gt_scene.get('props') or [])]
    c_tab = {e['name'] for e, k in c_ents if k == 'table'}; g_tab = {e['name'] for e, k in g_ents if k == 'table'}
    # visible samples per entity (measured side, 1 / cm^2) and the dense surfaces (distance targets, 64 / cm^2)
    C = {e['name']: _entity_points(cand_scene, cand_pkg, e, cam, hw, rws, Zc) for e, _ in c_ents}
    G = {e['name']: _entity_points(gt_scene, gt_dir, e, cam, hw, rws, Zg) for e, _ in g_ents}
    Csup = _support_points(cand_scene, cand_pkg, cam, hw, rws, Zc); Gsup = _support_points(gt_scene, gt_dir, cam, hw, rws, Zg)
    C64 = np.concatenate([_entity_points(cand_scene, cand_pkg, e, cam, hw, rws, Zc, **dense) for e, k in c_ents if k == 'entity'] or [np.zeros((0, 3))])
    G64 = np.concatenate([_entity_points(gt_scene, gt_dir, e, cam, hw, rws, Zg, **dense) for e, k in g_ents if k == 'entity'] or [np.zeros((0, 3))])
    Ctab64 = np.concatenate([_entity_points(cand_scene, cand_pkg, e, cam, hw, rws, Zc, **dense) for e, k in c_ents if k == 'table'] + [_support_points(cand_scene, cand_pkg, cam, hw, rws, Zc, **dense)])
    Gtab64 = np.concatenate([_entity_points(gt_scene, gt_dir, e, cam, hw, rws, Zg, **dense) for e, k in g_ents if k == 'table'] + [_support_points(gt_scene, gt_dir, cam, hw, rws, Zg, **dense)])
    out = dict(version='v3: per-entity median, role-weighted, table separate, trimmed-ICP similarity alignment; camera image from camera.json', img_hw=hw, rows=rws, roles=roles,
               n_cand_entities=len(c_ents), n_gt_entities=len(g_ents))
    if not len(G64) or not len(C64):
        out.update(scene_cd_cm=float('nan'), error='no visible entity samples on one side'); return out
    # 4. alignment on EVERY visible surface (entities + table), trimmed
    Aall = np.concatenate([p for p in C.values() if len(p)] + ([Csup] if len(Csup) else [])); Ball64 = np.concatenate([G64, Gtab64]) if len(Gtab64) else G64
    s, R, t = trimmed_icp_similarity(Aall, cKDTree(Ball64))
    cen = Aall.mean(0); shift = s * R @ cen + t - cen
    from scipy.spatial.transform import Rotation as _R
    out['alignment'] = dict(scale=round(float(s), 4), rot_deg=round(float(np.degrees(np.linalg.norm(_R.from_matrix(R).as_rotvec()))), 2), trans_cm=round(float(np.linalg.norm(shift)) * 100, 2), trans_vec_cm=(shift * 100).round(2).tolist(), note='similarity fit of the candidate visible surfaces onto the GT (trimmed ICP, 70 %): the monocular frame error the delivery carries; trans = displacement of the candidate scene centroid')
    xf = lambda P: (s * P @ R.T + t) if len(P) else P
    def per_entity(src, tgt_tree, transform):
        rows_ = {}
        for name, P in src.items():
            if not len(P): rows_[name] = dict(n=0, median_cm=None, mean_cm=None); continue
            d = tgt_tree.query(transform(P))[0]; rows_[name] = dict(n=int(len(P)), median_cm=round(100 * float(np.median(d)), 3), mean_cm=round(100 * float(d.mean()), 3), p95_cm=round(100 * float(np.percentile(d, 95)), 3))
        return rows_
    Gt64 = cKDTree(G64); Ct64_al = cKDTree(xf(C64)); Ct64_raw = cKDTree(C64)
    Cent = {n: P for n, P in C.items() if n not in c_tab}; Gent = {n: P for n, P in G.items() if n not in g_tab}
    c2g = per_entity(Cent, Gt64, xf); g2c = per_entity(Gent, Ct64_al, lambda P: P)
    c2g_raw = per_entity(Cent, Gt64, lambda P: P); g2c_raw = per_entity(Gent, Ct64_raw, lambda P: P)
    def role(name):
        if name == roles.get('object'): return 'object'
        if roles.get('target') and (name == roles['target'] or str(roles['target']) in name): return 'target'
        return 'other'
    def agg(rows_, weighted):
        """role-weighted mean of the per-entity medians over MATCHED entities (median <= MISSING_CM); an unmatched entity is a missing / extra
        entity (reported as a fraction), not a distance -- the GT scene lists only the annotated objects, so a real thing the candidate modelled
        that the GT omits would otherwise dominate the number (oak_micro_6: four such props, 9 cm)"""
        vals, ws, all_v, all_w = [], [], [], []
        for name, r_ in rows_.items():
            if r_['median_cm'] is None: continue
            all_v.append(r_['median_cm']); all_w.append(ROLE_WEIGHTS[role(name)] if weighted else 1.0)
            if r_['median_cm'] > MISSING_CM: continue
            vals.append(r_['median_cm']); ws.append(all_w[-1])
        if vals: return float(np.average(vals, weights=ws))
        return None if not all_v else float(np.average(all_v, weights=all_w))   # nothing matched on this side: the (uncapped) distances say how far off it is
    g2c_w, c2g_u = agg(g2c, True), agg(c2g, False)
    out['gt_to_cand_cm'] = g2c_w; out['cand_to_gt_cm'] = c2g_u; out['gt_to_cand_unweighted_cm'] = agg(g2c, False)
    out['scene_cd_cm'] = (g2c_w or 0) + (c2g_u or 0) if (g2c_w is not None and c2g_u is not None) else float('nan')   # NaN = no matched entity on one side (see missing_fraction / extra_fraction)
    g2c_wr, c2g_ur = agg(g2c_raw, True), agg(c2g_raw, False)
    out['scene_cd_unaligned_cm'] = (g2c_wr or 0) + (c2g_ur or 0) if (g2c_wr is not None and c2g_ur is not None) else float('nan')
    out['per_entity'] = dict(gt=g2c, cand=c2g, gt_unaligned=g2c_raw, cand_unaligned=c2g_raw, gt_roles={n: role(n) for n in Gent}, gt_tables=sorted(g_tab), cand_tables=sorted(c_tab))
    vis_g = [n for n, r_ in g2c.items() if r_['n'] > 0]; miss = [n for n in vis_g if g2c[n]['median_cm'] > MISSING_CM]
    vis_c = [n for n, r_ in c2g.items() if r_['n'] > 0]; extra = [n for n in vis_c if c2g[n]['median_cm'] > MISSING_CM]
    out['missing_fraction'] = (len(miss) / len(vis_g)) if vis_g else None; out['missing'] = miss; out['extra_fraction'] = (len(extra) / len(vis_c)) if vis_c else None; out['extra'] = extra
    out['n_gt_visible_entities'] = len(vis_g); out['n_cand_visible_entities'] = len(vis_c)
    # 1. table: candidate table surface (support field and/or table-named props) vs the GT support plane / GT table entities
    tab = dict(gt_has_support=bool(gt_scene.get('support')), cand_has_table=bool(len(Ctab64)))
    Ctab1 = np.concatenate([C[e['name']] for e, k in c_ents if k == 'table'] + ([Csup] if len(Csup) else []) or [np.zeros((0, 3))])
    if gt_scene.get('support') and len(Ctab1):
        zg = float(gt_scene['support']['z']); dz = Ctab1[:, 2] - zg
        tab.update(table_depth_err_cm=round(100 * float(np.abs(dz).mean()), 3), table_depth_signed_cm=round(100 * float(dz.mean()), 3), table_depth_err_aligned_cm=round(100 * float(np.abs(xf(Ctab1)[:, 2] - zg).mean()), 3), n=int(len(Ctab1)), rule='mean |z - z_gt| of the candidate table surface samples (base frame, GT support plane z)')
    elif len(Gtab64) and len(Ctab1):
        d = cKDTree(Gtab64).query(Ctab1)[0]; tab.update(table_depth_err_cm=round(100 * float(d.mean()), 3), n=int(len(Ctab1)), rule='mean distance of the candidate table samples to the GT table entity surface (no support plane declared)')
    elif len(Gsup) or len(Gtab64): tab.update(table_depth_err_cm=None, rule='candidate delivered no table')
    out['table'] = tab; out['table_depth_err_cm'] = tab.get('table_depth_err_cm')
    return out


# v4: geometry, not candidate entity names/counts, defines the measured surface.
SCENE_V4_RES_M = 0.004


def _v4_voxels(points, resolution=SCENE_V4_RES_M):
    points = np.asarray(points, float).reshape(-1, 3)
    if not np.isfinite(points).all():
        raise ValueError('non-finite scene surface')
    # Cell centres, not a centroid weighted by duplicate observations.
    return np.unique(np.round(points / resolution).astype(np.int64), axis=0).astype(float) * resolution


def _v4_surface(scene, pkg_dir, cam, hw, rows, resolution=SCENE_V4_RES_M, workspace_only=True):
    if cam is not None:
        Z = depth_image(scene, pkg_dir, cam, hw)
        yy, xx = np.nonzero(np.isfinite(Z))
        keep = (yy >= rows[0]) & (yy < rows[1]); yy, xx = yy[keep], xx[keep]
        rays = np.c_[xx + .5, yy + .5, np.ones(len(xx))] @ np.linalg.inv(np.asarray(cam['intrinsics'], float)).T
        P = rays * Z[yy, xx, None]
        T = np.asarray(cam['extrinsics_base_cam'], float)
        P = P @ T[:3, :3].T + T[:3, 3]
    else:
        # Deterministic triangle lattice. Independent of entity names/order/count;
        # re-triangulation changes at most the spatial discretization tolerance.
        triangles = _scene_triangles(scene, pkg_dir)
        if not np.isfinite(triangles).all(): raise ValueError('non-finite scene triangles')
        if not len(triangles): return np.zeros((0, 3))
        canonical = []
        for tri in triangles:
            canonical.append(tri[np.lexsort(tri.T[::-1])].reshape(-1))
        triangles = np.unique(np.round(canonical, 10), axis=0).reshape(-1, 3, 3)
        parts = []
        for tri in triangles:
            n = max(1, int(np.ceil(max(np.linalg.norm(tri[i] - tri[j]) for i, j in ((0, 1), (1, 2), (2, 0))) / (resolution / 2))))
            if n > 2000: raise ValueError('scene triangle exceeds geometry sampling budget')
            for i in range(n + 1):
                j = np.arange(n + 1 - i)
                parts.append(tri[0] + (i / n) * (tri[1] - tri[0]) + (j[:, None] / n) * (tri[2] - tri[0]))
        P = np.concatenate(parts) if parts else np.zeros((0, 3))
    if workspace_only: P = P[np.linalg.norm(P[:, :2], axis=1) <= .9]
    return _v4_voxels(P, resolution)


def scene_cd_v4(cand_scene, cand_pkg, gt_scene, gt_dir, cam, roles=None, img_hw=None, rows=None):
    """Fixed-GT-scope visible surface distance, cm; no fitted transform in headline.

    GT owns roles and scope. Candidate names, part counts and declared support roles
    have no scoring authority. All candidate geometry is one deduplicated surface.
    Per-GT-role recall is a mean (including every missing point), capped at CAP_CM.
    Reverse error is integrated over unique candidate voxels in the annotation's
    bounding box plus MISSING_CM margin, divided by the fixed GT voxel count.
    Unannotated background outside this scope is diagnostic only. Table is a
    separate fixed GT surface, never a candidate-name-based exclusion.
    Existing v3 remains available explicitly as a legacy diagnostic.
    """
    roles = roles or {}; hw, rws = image_rows(cam, img_hw)
    if img_hw is not None: hw = tuple(img_hw)
    if rows is not None: rws = tuple(rows)
    resolution = SCENE_V4_RES_M
    C = _v4_surface(cand_scene, cand_pkg, cam, hw, rws, resolution)
    Gall = _v4_surface(gt_scene, gt_dir, cam, hw, rws, resolution)
    ctree = cKDTree(C) if len(C) else None
    # Visibility is determined by GT, independent of what a candidate occludes.
    full_gt_tree = cKDTree(Gall) if len(Gall) else None
    def visible(sub):
        P = _v4_surface(sub, gt_dir, cam, hw, rws, resolution)
        if cam is not None and len(P) and full_gt_tree is not None:
            P = P[full_gt_tree.query(P)[0] <= resolution * 1.75]
        return P
    entities = []; table_points = []
    for idx, e in enumerate((gt_scene.get('objects') or []) + (gt_scene.get('props') or [])):
        P = visible(dict(objects=[e], props=[]))
        # GT role overrides GT names too: a manipulated 'table_leg' is an object.
        name = e['name']; role = 'object' if name == roles.get('object') else 'target' if name == roles.get('target') else 'other'
        if role in ('object', 'target') and not len(P):
            # Required annotated geometry cannot disappear from the denominator
            # because of occlusion/frustum. Its known full GT surface is retained.
            P = _v4_surface(dict(objects=[e], props=[]), gt_dir, None, hw, rws, resolution, workspace_only=False)
        if role == 'other' and _is_table(e): table_points.append(P)
        else: entities.append((f'{idx}:{name}', role, P))
    if gt_scene.get('support'):
        table_points.append(visible(dict(objects=[], props=[], support=gt_scene['support'])))
    def distances(P):
        return np.minimum(ctree.query(P)[0] * 100, CAP_CM) if ctree is not None else np.full(len(P), CAP_CM)
    records = {}; weights = []; recalls = []; covered = []; gparts = []; invalid_required = []
    for key, role, P in entities:
        if not len(P):
            if role in ('object', 'target'):
                records[key] = dict(role=role, n=0, applicable=True, mean_cm=CAP_CM, missing_fraction=1., reason='required GT entity has no samples in fixed workspace')
                weights.append(ROLE_WEIGHTS[role]); recalls.append(CAP_CM); covered.append(1.); invalid_required.append(key)
            else:
                records[key] = dict(role=role, n=0, applicable=False, reason='no GT-visible surface')
            continue
        d = distances(P); w = ROLE_WEIGHTS[role]
        records[key] = dict(role=role, n=len(P), applicable=True, mean_cm=float(d.mean()), missing_fraction=float(np.mean(d > MISSING_CM)))
        weights.append(w); recalls.append(float(d.mean())); covered.append(float(np.mean(d > MISSING_CM))); gparts.append(P)
    G = _v4_voxels(np.concatenate(gparts), resolution) if gparts else np.zeros((0, 3))
    # No candidate surface means no extras, while recall records all missing GT.
    # Undefined GT scope remains invalid rather than inventing applicability.
    reverse = 0.; scoped_n = 0; extra_fraction = 0. if len(G) else None
    if len(G) and len(C):
        # Fixed reference annotation scope, not candidate count or bounding box.
        scope = np.zeros(len(C), bool)
        for P in gparts:
            scope |= np.all((C >= P.min(0) - MISSING_CM / 100) & (C <= P.max(0) + MISSING_CM / 100), axis=1)
        d = np.minimum(cKDTree(G).query(C)[0] * 100, CAP_CM)
        reverse = float(np.minimum(d[scope].sum() / len(G), CAP_CM))
        scoped_n = int(scope.sum()); extra_fraction = float(np.mean(d > MISSING_CM))
    recall = float(np.average(recalls, weights=weights)) if weights else CAP_CM
    out = dict(version='v4: fixed GT scope, visible voxel union, fixed GT denominator, no headline ICP',
               scene_cd_cm=recall + reverse, gt_to_cand_cm=recall, cand_to_gt_cm=reverse,
               scene_cd_unaligned_cm=recall + reverse, cap_cm=CAP_CM, resolution_m=resolution,
               metric_valid=bool(weights) and not invalid_required, invalid_reason=('required GT entity has empty geometry: ' + ', '.join(invalid_required)) if invalid_required else (None if weights else 'no GT-visible annotated entity'),
               missing_fraction=float(np.average(covered, weights=weights)) if weights else 1.,
               extra_fraction=extra_fraction, n_cand_surface=len(C), n_gt_surface=len(G), n_cand_in_scope=scoped_n,
               per_entity=dict(gt=records), img_hw=hw, rows=rws, roles=roles,
               alignment=dict(scale=1., rot_deg=0., trans_cm=0., applied=False, note='fixed delivered frame; v3 alignment is a separate legacy diagnostic'),
               reverse_scope='GT entity AABBs plus 5 cm; outside is unannotated background, diagnostic only')
    T = _v4_voxels(np.concatenate(table_points), resolution) if table_points else np.zeros((0, 3))
    # Missing required table is finite failure, not N/A. No GT table means N/A.
    out['table_depth_err_cm'] = float(distances(T).mean()) if len(T) else None
    out['table'] = dict(gt_has_table=bool(len(T)), n_gt=len(T), rule='GT-visible table surface to candidate union, capped mean 3-D distance')
    return out


# ----------------------------------------------------------------------------------------------------------------------------------
# Scene CD v5: the merge of v3 and v4.
#   from v4: the CANDIDATE surface is geometry, not names -- the candidate depth image back-projected and voxelised at 4 mm, whatever the
#            agent called its parts and however many it delivered; the table is decided by geometry: GT side = support plane + GT 'other' props that are
#            broad thin slabs; candidate side (rev 2) = every delivered part that is a broad thin slab within TABLE_NEAR_M of the GT table plane and not
#            sitting on a GT entity, plus its `support`, whatever it is called (a table split in three: three tables); the GT decides
#            the denominator (a GT entity with a role stays in the forward term even when the candidate has nothing there).
#   from v3: the table is scored on its own (table_depth_err_cm) and kept out of the CD; a trimmed-ICP similarity alignment is applied first
#            (scene_align_cm; absolute placement is scored by Obj pos / APE in the delivered frame, so the CD does not score it twice);
#            per-entity MEDIAN distances, role-weighted (object 2 / target 2 / other 1); missing / extra fractions reported.
#   from v3b: a missing GT entity is not dropped from the headline but counted at the cap (CAP5_CM = 10 cm).
#   dropped from v4: the "GT bounding box + 5 cm" reverse scope (it counted the candidate's own table as extra surface: a perfect scene read
#            4-6 cm and lowering the table lowered the score) and the 100 cm capped means.
# headline scene_cd_cm = forward (GT entities -> aligned candidate non-table voxels, per-entity median capped at 10 cm, role-weighted)
#                        + reverse (candidate non-table voxels -> GT surface incl. the table, median capped at 10 cm; rev 2).
CAP5_CM = 10.0
SCENE5_REVISION = "scene_cd_v5/rev8-assembly-object-alignment"
TABLE5_CAP_CM = 10.0
CONTEXT_D0_M, CONTEXT_D1_M = 0.05, 0.25   # rev 4: a non-interacted GT entity's weight = clip((D1 - d) / (D1 - D0), 0, 1), d = its
                                          # distance to the demonstrated motion (TCP path + the manipulated object's swept surface); the interacted pair keeps 2 / 2
_CONTEXT_CACHE = {}


def context_weights(sd, gt_scene, roles, d0=CONTEXT_D0_M, d1=CONTEXT_D1_M):
    """per-entity weights for scene_cd_v5 (rev 4): manipulated object and receiving part 2 each; every other entity by how close its GT
    surface comes to the demonstrated motion (hidden/trajectory.npz tcp_pose + the manipulated object's hidden/objects_6d.npz path swept with
    its mesh): within d0 -> 1 (an obstacle the robot / object passes), beyond d1 -> 0 (background clutter). Tables are not weighted here.
    Samples without a hidden motion (twins without a TCP path) fall back to weight 1 for every other entity."""
    from scipy.spatial.transform import Rotation as _R
    sd = Path(sd); key = (str(sd), roles.get('object'), roles.get('target'))
    if key in _CONTEXT_CACHE: return dict(_CONTEXT_CACHE[key])
    def role(n):
        if n == roles.get('object'): return 'object'
        if roles.get('target') and (n == roles['target'] or str(roles['target']) in n): return 'target'
        return 'other'
    ents = (gt_scene.get('objects') or []) + (gt_scene.get('props') or []); W = {e['name']: 2.0 for e in ents if role(e['name']) != 'other'}
    others = [e for e in ents if role(e['name']) == 'other']; gt_dir = sd / 'gt_pkg'; path = []; note = None
    try:
        z = np.load(sd / 'hidden/objects_6d.npz'); src = roles.get('object')
        if src in z.files:
            o = next((e for e in ents if e['name'] == src), None); T = np.asarray(z[src], float)
            if o is not None:
                V = np.unique(np.asarray(entity_mesh(o, gt_dir).vertices, float), axis=0)
                if len(V) > 300: V = V[np.random.default_rng(0).choice(len(V), 300, replace=False)]
                for k in range(0, len(T), max(1, len(T) // 15)):
                    q = T[k, 3:7]; path.append(V @ _R.from_quat(np.r_[q[1:], q[0]]).as_matrix().T + T[k, :3])
        t = np.load(sd / 'hidden/trajectory.npz')
        if 'tcp_pose' in t.files:
            # Missing source-hand observations are not demonstrated positions.
            # Keep the finite GT motion and the object's swept surface; never
            # fill missing observations or filter candidate scene geometry here.
            tcp = np.asarray(t['tcp_pose'], float)[:, :3]
            valid = np.isfinite(tcp).all(axis=1)
            if not valid.all():
                W['@tcp_nonfinite_rows'] = int((~valid).sum())
                note = f'ignored {int((~valid).sum())} non-finite GT TCP observations'
            if valid.any(): path.append(tcp[valid])
    except Exception as e: note = f'{type(e).__name__}: {e}'
    if path:
        tree = cKDTree(np.concatenate(path))
        for e in others:
            try:
                V = np.unique(np.asarray(entity_mesh(e, gt_dir).vertices, float), axis=0)
                if len(V) > 1500: V = V[np.random.default_rng(0).choice(len(V), 1500, replace=False)]
                q = np.asarray(e.get('quat', (1, 0, 0, 0)), float); V = V @ _R.from_quat(np.r_[q[1:], q[0]]).as_matrix().T + np.asarray(e['pos'], float)
                d = float(tree.query(V)[0].min()); W[e['name']] = round(float(np.clip((d1 - d) / (d1 - d0), 0.0, 1.0)), 3); W[e['name'] + '@d_cm'] = round(100 * d, 1)
            except Exception: W[e['name']] = 1.0
    else:
        for e in others: W[e['name']] = 1.0
        W['@note'] = note or 'no demonstrated motion found: every other entity weighted 1'
    _CONTEXT_CACHE[key] = W; return dict(W)


ALIGN5_SCALE = (0.95, 1.05)   # rev 3: the alignment corrects a FRAME error -- translation is free,
ALIGN5_ROT_DEG = 15.0         # but a rescaled (0.70 on 5 of 84 agent rows) or turned (up to 30 deg) scene is the agent's error and stays in the score


def _icp_bounded(A, Btree, trim=ICP_TRIM):
    """trimmed-ICP similarity fit with v5's bounds: scale within ALIGN5_SCALE; if the fitted rotation exceeds ALIGN5_ROT_DEG its angle is
    clamped (same axis) and the translation is re-fitted for that rotation (translation-only trimmed iterations)"""
    from scipy.spatial.transform import Rotation as _R
    s, R, t = trimmed_icp_similarity(A, Btree, trim=trim, scale_range=ALIGN5_SCALE)
    rv = _R.from_matrix(R).as_rotvec(); ang = float(np.degrees(np.linalg.norm(rv)))
    if ang > ALIGN5_ROT_DEG:
        R = _R.from_rotvec(rv / ang * ALIGN5_ROT_DEG).as_matrix(); X0 = s * A @ R.T; t = np.zeros(3)
        for _ in range(ICP_ITERS):
            d, j = Btree.query(X0 + t); k = max(int(len(X0) * trim), 10); sel = np.argsort(d)[:k]
            dt = (Btree.data[j[sel]] - (X0[sel] + t)).mean(0); t = t + dt
            if np.linalg.norm(dt) < 1e-4: break
    return s, R, t


VOXEL5_M = 0.004   # candidate voxel size (cell centres of the union of the visible samples of every delivered part)


def _voxels(points, resolution=VOXEL5_M):
    """unique voxel-cell centres of a point set (duplicate observations count once)"""
    P = np.asarray(points, float).reshape(-1, 3)
    if not len(P): return np.zeros((0, 3))
    if not np.isfinite(P).all(): raise ValueError('non-finite scene surface')
    return np.unique(np.round(P / resolution).astype(np.int64), axis=0).astype(float) * resolution


TABLE_NEAR_M, ENTITY_WINS_M = 0.10, 0.03   # candidate part classification: a broad thin slab within TABLE_NEAR_M of the GT table plane is a table unless >= 50% of its samples lie within ENTITY_WINS_M of a GT entity


def table_surface_result(raw_cm, gt_has_table, cand_has_table):
    """Missing delivery gets the same cap as any measured table; GT absence is N/A."""
    out = dict(cap_cm=TABLE5_CAP_CM, gt_has_table=bool(gt_has_table), table_present=bool(cand_has_table))
    if not gt_has_table:
        out.update(table_depth_err_cm=None, table_depth_err_raw_cm=None, status='na',
                   reason='GT has no annotated table', raw_na_reason='table metric is not applicable')
    elif not cand_has_table:
        out.update(table_depth_err_cm=TABLE5_CAP_CM, table_depth_err_raw_cm=None, status='ok',
                   penalty='candidate delivered no table-like surface', raw_na_reason='no candidate table to measure')
    else:
        if isinstance(raw_cm, (bool, np.bool_)) or not isinstance(raw_cm, (int, float, np.number)) or not np.isfinite(raw_cm) or raw_cm < 0:
            raise ValueError('invalid measured table surface distance')
        out.update(table_depth_err_cm=round(min(float(raw_cm), TABLE5_CAP_CM), 3),
                   table_depth_err_raw_cm=round(float(raw_cm), 3), status='ok', capped=bool(raw_cm > TABLE5_CAP_CM))
    return out


def scene_cd_v5(cand_scene: dict, cand_pkg, gt_scene: dict, gt_dir, cam: dict | None, roles: dict | None = None, img_hw=None, rows=None, weights: dict | None = None) -> dict:
    # Reject invalid transforms before visibility filtering can turn NaNs into an empty delivery.
    for label, scene in (('GT', gt_scene), ('candidate', cand_scene)):
        for entity in (scene.get('objects') or []) + (scene.get('props') or []):
            position = np.asarray(entity['pos'], float); quaternion = np.asarray(entity.get('quat', [1, 0, 0, 0]), float)
            if position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(position).all() or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) == 0:
                raise ValueError(label + ' entity has invalid pose')
        if scene.get('support'):
            support = scene['support']; size = np.asarray(support.get('size', [1., 1.]), float)
            center = np.asarray(support.get('center', [.3, 0.]), float)
            if size.shape != (2,) or center.shape != (2,) or not np.isfinite(np.r_[size, center, float(support['z'])]).all() or (size <= 0).any():
                raise ValueError(label + ' support has invalid geometry')
    if cam is not None:
        for key, shape in (('intrinsics', (3, 3)), ('extrinsics_base_cam', (4, 4))):
            value = np.asarray(cam[key], float)
            if value.shape != shape or not np.isfinite(value).all(): raise ValueError('invalid camera ' + key)
    hw, rws = image_rows(cam, img_hw)
    if img_hw is not None: hw = tuple(img_hw)
    if rows is not None: rws = tuple(rows)
    roles = roles or {}; dense = dict(density=SCENE_DENSITY * SCENE_DENSE_MULT, nmax=SCENE_N_MAX * SCENE_DENSE_MULT, canonical_surface=True)
    W = weights if weights is not None else context_weights(Path(gt_dir).parent, gt_scene, roles)   # rev 4: entity weights from the demonstrated motion
    wt = lambda n: float(W.get(n, 1.0))
    out = dict(surface_sampling='canonical-triangles/1', revision=SCENE5_REVISION, version='v5 rev 8: assembly-aware scope; table-independent object visibility and deterministic object-only alignment; canonical triangle-order surface sampling; select minimum headline objective including identity; defined empty-scene and capped table penalties; v3 x v4 merge (candidate = delivered parts by geometry, table by slab test against the GT plane, GT-decided denominator, aligned with free translation / bounded scale and rotation, per-entity medians, missing counted at the cap, non-interacted entities weighted by their distance to the demonstrated motion)', img_hw=hw, rows=rws, roles=roles, cap_cm=CAP5_CM)
    # --- GT side: entities with roles, table by role + geometry, visible samples per entity (v3 sampler), dense targets
    Zg = depth_image(gt_scene, gt_dir, cam, hw) if cam is not None else None
    g_ents = [(e, 'table' if _is_table(e, gt_dir, roles) else 'entity') for e in sorted((gt_scene.get('objects') or []) + (gt_scene.get('props') or []), key=lambda e: e['name'])]
    def role(name):
        if name == roles.get('object'): return 'object'
        if roles.get('target') and (name == roles['target'] or str(roles['target']) in name): return 'target'
        return 'other'
    G = {}
    for e, k in g_ents:
        if k != 'entity': continue
        P = _entity_points(gt_scene, gt_dir, e, cam, hw, rws, Zg, canonical_surface=True)
        if not len(P) and role(e['name']) in ('object', 'target'):   # a required entity outside the frustum keeps its full surface (v4)
            P = scene_points_v2(dict(support=None, props=[e], objects=[]), gt_dir, None, canonical_surface=True)
        G[e['name']] = P
    _g64 = [(e['name'], _entity_points(gt_scene, gt_dir, e, cam, hw, rws, Zg, **dense)) for e, k in g_ents if k == 'entity']
    G64 = np.concatenate([P for _, P in _g64] or [np.zeros((0, 3))]); G64w = np.concatenate([np.full(len(P), wt(n)) for n, P in _g64] or [np.zeros(0)])
    out['weights'] = {k: v for k, v in W.items()}
    Gtab64 = np.concatenate([_entity_points(gt_scene, gt_dir, e, cam, hw, rws, Zg, **dense) for e, k in g_ents if k == 'table'] + [_support_points(gt_scene, gt_dir, cam, hw, rws, Zg, **dense)])
    out['gt_tables'] = [e['name'] for e, k in g_ents if k == 'table'] + (['<support>'] if gt_scene.get('support') else [])
    if not len(G64) or not any(len(P) for P in G.values()):
        out.update(scene_cd_cm=float('nan'), error='no visible GT entity surface'); return out
    if out['gt_tables'] and not len(Gtab64):
        out.update(scene_cd_cm=float('nan'), error='GT table annotation has no measurable surface'); return out
    # --- candidate side: the delivered PARTS by their geometry, names ignored (rev 2: a purely voxel-level table rule read a misplaced
    # flat object as "the table under where it should be"). A delivered part is a table when it is broad (visible footprint spans
    # >= TABLE_MIN_SPAN_M), the median of its samples lies within TABLE_NEAR_M of the GT table plane (a table 5 cm too low is still a table;
    # a big box on the table is not), and it is NOT sitting on a GT entity (< half of its samples within ENTITY_WINS_M of a GT entity
    # surface). So: a table split into three boxes = three tables; a table renamed 'floor' = a table; a spoon lying on the table = an entity
    # (too small); a furniture panel delivered where the GT panel is = an entity (matched); the same panel delivered elsewhere on the table
    # = a table, and the GT panel reads missing. The `support` field is always a table. The tests are run in the ALIGNED frame (a scene
    # delivered 20 cm off still has a table): a first classification in the delivered frame gives the alignment, the classification is
    # repeated on the aligned samples, and the alignment is redone once if it changed.
    # Explicit support identification comes from scope geometry, never a preferred
    # condition or a fitted score. It cannot occlude the object metric's input.
    support_names = set(cand_scene.get('_scene_table_names', []))
    object_scene = dict(cand_scene, support=None,
        objects=[e for e in cand_scene.get('objects', []) if e['name'] not in support_names],
        props=[e for e in cand_scene.get('props', []) if e['name'] not in support_names])
    Zc = depth_image(object_scene, cand_pkg, cam, hw) if cam is not None else None
    Ztable = depth_image(cand_scene, cand_pkg, cam, hw) if cam is not None else None
    zt = float(gt_scene['support']['z']) if gt_scene.get('support') else None
    gtab_tree = cKDTree(Gtab64) if len(Gtab64) else None; gent_tree = cKDTree(G64)
    Ball = np.concatenate([G64, Gtab64]) if len(Gtab64) else G64; ball_tree = cKDTree(Ball)
    parts = [(e['name'], P) for e in (cand_scene.get('objects') or []) + (cand_scene.get('props') or []) for P in [_entity_points(cand_scene, cand_pkg, e, cam, hw, rws, Ztable if e['name'] in support_names else Zc, canonical_surface=True)] if len(P)]
    Csup = _support_points(cand_scene, cand_pkg, cam, hw, rws, Ztable, canonical_surface=True)
    def classify(xf):
        names = []
        for name, P in parts:
            if name in support_names:
                names.append(name); continue
            if cand_scene.get('_scene_explicit_assemblies'):
                continue
            Q = xf(P); broad = float(np.ptp(Q[:, :2], axis=0).max()) >= TABLE_MIN_SPAN_M   # (no thickness test: a table's visible legs are a minority of its samples, the median test handles them)
            near_plane = (float(np.median(np.abs(Q[:, 2] - zt))) <= TABLE_NEAR_M) if zt is not None else (gtab_tree is not None and float(np.median(gtab_tree.query(Q)[0])) <= TABLE_NEAR_M)
            matched = float(np.mean(gent_tree.query(Q)[0] <= ENTITY_WINS_M)) >= 0.5
            if broad and near_plane and not matched: names.append(name)
        return names
    def build(tab_names):
        c_tab = [P for name, P in parts if name in tab_names] + ([Csup] if len(Csup) else []); c_ent = [P for name, P in parts if name not in tab_names]
        Ctab = _voxels(np.concatenate(c_tab)) if c_tab else np.zeros((0, 3)); Cent = _voxels(np.concatenate(c_ent)) if c_ent else np.zeros((0, 3))
        return Ctab, Cent, np.concatenate([Ctab, Cent])
    def terms(Cn):
        """forward (GT entities -> candidate non-table voxels) and reverse (those voxels -> GT entity surface) for a given candidate point set"""
        ct = cKDTree(Cn) if len(Cn) else None; per = {}; vals = []; ws = []; miss = []
        for name, P in G.items():
            r_ = role(name)
            if not len(P): per[name] = dict(role=r_, weight=wt(name), n=0, median_cm=None, note='no GT-visible surface'); continue
            med = 100 * float(np.median(ct.query(P)[0])) if ct is not None else None
            counted = min(med, CAP5_CM) if med is not None else CAP5_CM
            missing_entity = med is None or med > MISSING_CM
            per[name] = dict(role=r_, weight=wt(name), n=int(len(P)), median_cm=round(med, 3) if med is not None else None,
                             counted_cm=round(counted, 3), missing=bool(missing_entity))
            if med is None: per[name]['penalty'] = 'no candidate entity surface; distance undefined, count cap'
            vals.append(counted); ws.append(wt(name)); miss.append(missing_entity)
        fwd = float(np.average(vals, weights=ws)) if vals and sum(ws) > 0 else (float(np.mean(vals)) if vals else None)
        # The weighted mean of capped terms can exceed the cap by a few ULPs.
        # Bound the term itself so aligned/unaligned diagnostics and sums agree.
        if fwd is not None: fwd = min(fwd, CAP5_CM)
        if len(Cn):
            # reverse: the candidate's non-table surface against EVERYTHING the GT has (entities + table): flat clutter lying on the table that
            # the GT does not model (a placemat, a tray) reads its height above the table, a wrong tall object reads up to the cap
            d = 100 * ball_tree.query(Cn)[0]; extra = float(np.mean(d > MISSING_CM))
            # rev 4: each candidate voxel inherits the weight of the GT entity nearest to it (background built or not does not move the score)
            vw = G64w[gent_tree.query(Cn)[1]]
            if vw.sum() > 0:
                o_ = np.argsort(d); cw = np.cumsum(vw[o_]); rev = float(min(d[o_][min(int(np.searchsorted(cw, 0.5 * cw[-1])), len(d) - 1)], CAP5_CM))
            else: rev = float(min(np.median(d), CAP5_CM))
        else: rev, extra = CAP5_CM, 0.0   # explicit empty-delivery penalty; Extra has an empty-denominator convention
        return fwd, rev, per, (float(np.average(miss, weights=ws)) if miss and sum(ws) > 0 else (float(np.mean(miss)) if miss else None)), extra
    def align(Ctab, Cent, C):
        """Candidate-only, deterministic bounded search; tables never anchor objects.

        Select and refine the actual reported objective. No counterpart candidate,
        condition label or reference preferred outcome participates in the search.
        This is a finite multi-start search, not a global-optimality claim.
        """
        from scipy.optimize import minimize
        objective = lambda Q: sum(v if v is not None else CAP5_CM for v in terms(Q)[:2])
        best = (objective(Cent), 'identity_fallback', 1., np.eye(3), np.zeros(3))
        history = [dict(hypothesis=best[1], objective_cm=best[0])]
        if not len(Cent): return best[1:]
        seeds = [('identity', np.zeros(3)),
                 ('centroid', G64.mean(0) - Cent.mean(0)),
                 ('median', np.median(G64, axis=0) - np.median(Cent, axis=0)),
                 ('bounds', (G64.min(0)+G64.max(0)-Cent.min(0)-Cent.max(0))/2)]
        # Matched assemblies provide additional seeds without introducing any
        # dependence on the other refinement branch or its alignment transform.
        for name, P in parts:
            if name in G and len(G[name]) and name not in support_names:
                seeds.append(('role/'+name, G[name].mean(0)-P.mean(0)))
        candidates = [best]
        for label, t0 in seeds:
            s_, R_, t_ = _icp_bounded(Cent + t0, gent_tree)
            t_ = s_ * R_ @ t0 + t_
            fit = objective(s_ * Cent @ R_.T + t_)
            candidates.append((fit, 'objects/'+label, s_, R_, t_))
            history.append(dict(hypothesis='objects/'+label, objective_cm=fit))
        # Translation refinement is bounded around each solution for a finite,
        # reproducible budget; rotation/scale retain their original constraints.
        for fit, label, s_, R_, t_ in sorted(candidates, key=lambda v:v[0])[:2]:
            fixed = s_ * Cent @ R_.T
            result = minimize(lambda t: objective(fixed+t), t_, method='Powell',
                bounds=[(v-.1,v+.1) for v in t_],
                options=dict(maxfev=180, xtol=1e-5, ftol=1e-6))
            value = objective(fixed+result.x)
            history.append(dict(hypothesis=label+'/translation', objective_cm=value,
                                evaluations=int(result.nfev)))
            candidates.append((value, label+'/translation', s_, R_, result.x))
        best = min(candidates, key=lambda v:v[0])
        out.setdefault('alignment_objective_history', []).append(history)
        return best[1:]
    tab_names = classify(lambda P: P); Ctab, Cent, C = build(tab_names)
    empty_surface = not len(C)
    # A successfully sampled empty candidate is a defined delivery failure, not a failed computation.
    hyp, s, R, t = ('empty_candidate', 1., np.eye(3), np.zeros(3)) if empty_surface else align(Ctab, Cent, C)
    passes = 1
    tab_names2 = classify(lambda P: s * P @ R.T + t)
    if set(tab_names2) != set(tab_names):
        tab_names = tab_names2; Ctab, Cent, C = build(tab_names); hyp, s, R, t = align(Ctab, Cent, C); passes = 2
    cen = Cent.mean(0) if len(Cent) else np.zeros(3); shift = s * R @ cen + t - cen
    tab = dict(gt_has_table=bool(len(Gtab64)), cand_tables=tab_names + (['<support>'] if len(Csup) else []), cand_table_voxels=int(len(Ctab)), classification_passes=passes,
               rule=f'a delivered part is a table when, in the aligned frame, it is broad (span >= {TABLE_MIN_SPAN_M} m), its median sample lies within {TABLE_NEAR_M * 100:.0f} cm of the GT table plane and it is not sitting on a GT entity (< 50% of its samples within {ENTITY_WINS_M * 100:.0f} cm of one); names ignored; table_depth_err = min(10 cm, median |z - z_gt| of the aligned table samples); raw measurement retained')
    raw_table_cm = None
    if len(Gtab64) and len(Ctab):
        Ctab_al = s * Ctab @ R.T + t
        raw_table_cm = 100 * float(np.median(np.abs(Ctab_al[:, 2] - zt))) if zt is not None else 100 * float(np.median(gtab_tree.query(Ctab_al)[0]))
    tab.update(table_surface_result(raw_table_cm, bool(len(Gtab64)), bool(len(Ctab))))
    out['table'] = tab; out['table_depth_err_cm'] = tab.get('table_depth_err_cm')
    from scipy.spatial.transform import Rotation as _R
    out['alignment'] = dict(hypothesis=hyp, transform=dict(s=float(s), R=np.asarray(R).round(6).tolist(), t=np.asarray(t).round(6).tolist()), scale=round(float(s), 4), rot_deg=round(float(np.degrees(np.linalg.norm(_R.from_matrix(R).as_rotvec()))), 2), trans_cm=round(float(np.linalg.norm(shift)) * 100, 2), trans_vec_cm=(shift * 100).round(2).tolist(), note='deterministic object-only multi-start ICP and headline-objective translation refinement; table excluded from anchors, translation free, scale within ALIGN5_SCALE, rotation within ALIGN5_ROT_DEG; absolute placement is scored by Obj pos / APE, not here')
    if empty_surface:
        out['alignment'] = dict(status='na', reason='no candidate visible surface; alignment not performed')
    else:
        out['alignment']['status'] = 'ok'
    xf = lambda P: (s * P @ R.T + t) if len(P) else P
    fwd, rev, per, missing, extra = terms(xf(Cent)); fwd0, rev0, per0, _, _ = terms(Cent)
    out.update(gt_to_cand_cm=fwd, cand_to_gt_cm=rev, scene_cd_cm=(fwd if fwd is not None else CAP5_CM) + (rev if rev is not None else CAP5_CM),
               scene_cd_unaligned_cm=(fwd0 if fwd0 is not None else CAP5_CM) + (rev0 if rev0 is not None else CAP5_CM), gt_to_cand_unaligned_cm=fwd0, cand_to_gt_unaligned_cm=rev0,
               missing_fraction=missing, missing=[n for n, r_ in per.items() if r_.get('missing')], extra_fraction=extra,
               per_entity=dict(gt=per, gt_unaligned=per0, gt_roles={n: role(n) for n in G}), n_cand_voxels=int(len(C)), n_cand_table_voxels=int(len(Ctab)), n_cand_entity_voxels=int(len(Cent)), n_gt_entities=len(G))
    out['metric_valid'] = True
    out['extra_denominator'] = int(len(Cent))
    if not len(Cent):
        out.update(empty_candidate_entities=True, extra_rule='empty candidate entity denominator: Extra is defined as 0',
                   penalty='no candidate entity surface: forward and reverse each count the 10 cm cap')
    if empty_surface: out['empty_candidate_surface'] = True
    pass
    apply_coverage(out,cand_scene,cand_pkg,gt_dir)
    return out


# ---------------------------------------------------------------- support surfaces
# Large upward planar support surfaces from delivered geometry.
#
# The area/span rule follows the scene scorer's broad-surface convention. It does
# not require both table dimensions to exceed 40 cm or require a box primitive.
SUPPORT_VERSION='delivered-planar-support/1'
def support_planes(scene,pkg):
 out=[];s=scene.get('support')
 if s:
  h=np.asarray(s.get('size',[1,1]),float)/2;c=np.asarray(s.get('center',[0,0]),float);T=np.eye(4);T[:3,3]=np.r_[c,s['z']];out.append(dict(name='<support>',T=T,bounds=np.array([-h,h]),hull=None,area=float(np.prod(2*h))))
 for e in scene.get('props',[]):
  mesh=entity_mesh(e,Path(pkg));pose=T_of(np.r_[e['pos'],e.get('quat',[1,0,0,0])]);verts=np.asarray(mesh.vertices);normals=np.asarray(mesh.face_normals);centers=np.asarray(mesh.triangles_center);groups={}
  # Coplanar faces, including disconnected triangles forming the same plane.
  for j,(n,c) in enumerate(zip(normals,centers)):
   if (pose[:3,:3]@n)[2]<.8:continue
   key=tuple(np.round(n,5))+ (round(float(n@c),5),);groups.setdefault(key,[]).append(j)
  for ids in groups.values():
   area=float(mesh.area_faces[ids].sum())
   if area<.04:continue
   n=normals[ids[0]];u=np.cross(n,[1,0,0] if abs(n[0])<.9 else [0,1,0]);u/=np.linalg.norm(u);v=np.cross(n,u);Q=verts[np.unique(mesh.faces[ids])];origin=Q.mean(0);xy=(Q-origin)@np.array([u,v]).T;ext=np.ptp(xy,axis=0)
   if max(ext)<.45:continue
   H=ConvexHull(xy);T=np.eye(4);T[:3,:3]=pose[:3,:3]@np.column_stack([u,v,n]);T[:3,3]=pose[:3,:3]@origin+pose[:3,3]
   out.append(dict(name=e['name'],T=T,bounds=np.array([xy.min(0),xy.max(0)]),hull=H.equations,area=area))
 return out


# ---------------------------------------------------------------- source-observed table boundary
# Source-observed boundary coverage without invented 3D ground truth.
#
# Report image geometry separately from physical surface CD. Source-visible edge
# intervals are directly annotated in pixels; use the candidate's delivered camera
# and no GT-fitted alignment. Unobserved intervals do not become negative labels.
OBSERVATION_VERSION='source-observed-table-boundary/1'
OBSERVATION_REGISTRY=Path(__file__).parent/'rules/droid_scene_observations.json'

def observation(gt_dir,registry=None):
 path=Path(registry or OBSERVATION_REGISTRY)
 if not path.is_file():raise FileNotFoundError('required scene observation registry: '+str(path))
 reg=json.loads(path.read_text())
 if reg.get('protocol')!=OBSERVATION_VERSION:raise ValueError('unrecognized observation registry version')
 row=reg['samples'].get(Path(gt_dir).parent.name)
 if row is None:return None
 if row['status']=='not_annotated':return row
 if row['status']!='observed_partial_boundary':raise ValueError('invalid observation status')
 for xy in row['pixel_polylines']:
  if len(xy)<2 or np.asarray(xy).shape!=(len(xy),2) or not np.isfinite(xy).all():raise ValueError('invalid observed boundary')
 return row

def _plane_polygon(p):
 if p['hull'] is None:
  lo,hi=p['bounds'];xy=np.array([[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]])
 else:
  H=np.asarray(p['hull']);v=[]
  for a,b in itertools.combinations(H,2):
   A=np.array([a[:2],b[:2]])
   if abs(np.linalg.det(A))<1e-10:continue
   q=np.linalg.solve(A,-np.array([a[2],b[2]]))
   if np.all(H[:,:2]@q+H[:,2]<=1e-7):v.append(q)
  xy=np.unique(np.round(v,12),axis=0)
  if len(xy)<3:raise ValueError('degenerate support boundary')
  xy=xy[ConvexHull(xy).vertices]
 T=p['T'];return np.c_[xy,np.zeros(len(xy))]@T[:3,:3].T+T[:3,3]

def boundary_segments(scene,pkg):
 pass
 allowed=set(scene.get('_scene_table_names',[]))|({'<support>'} if scene.get('support') else set())
 planes=[p for p in support_planes(scene,pkg) if p['name'] in allowed]
 if not planes:return np.empty((0,2,3))
 # For a delivered slab use its largest upward face; its underside is not a
 # second tabletop. Part boundaries on different physical tables stay separate.
 chosen={}
 for p in planes:
  if p['name'] not in chosen or p['area']>chosen[p['name']]['area']:chosen[p['name']]=p
 segments=[]
 for p in chosen.values():
  poly=_plane_polygon(p);segments.extend(np.stack([poly,np.roll(poly,-1,axis=0)],axis=1))
 return np.asarray(segments)

def distances(points,segments):
 if not len(segments):return np.full(len(points),np.inf)
 a=segments[:,0];d=segments[:,1]-a;den=np.sum(d*d,axis=1)
 t=np.clip(np.sum((points[:,None]-a)*d,axis=2)/np.maximum(den,1e-20),0,1)
 return np.linalg.norm(points[:,None]-(a+t[:,:,None]*d),axis=2).min(axis=1)

def apply_coverage(out,scene,pkg,gt_dir,registry=None):
 row=observation(gt_dir,registry)
 out['coverage_protocol']=OBSERVATION_VERSION
 if row is None or row['status']=='not_annotated':
  out['observed_table_boundary']=dict(status='na',reason=row['reason'] if row else 'no source-verified boundary annotation for this sample',proxy_support_is_not_boundary_gt=True)
  return out
 # Read the already frame-normalized candidate camera; do not substitute hidden
 # intrinsics, and do not apply the object ICP transform to image evidence.
 path=Path(pkg)/'protocol.json'
 if not path.is_file():
  out['observed_table_boundary']=dict(status='error',reason='candidate protocol missing');return out
 cams=json.loads(path.read_text()).get('cameras',[])
 if not cams:
  out['observed_table_boundary']=dict(status='error',reason='candidate camera missing');return out
 cam=cams[0];K=np.asarray(cam['intrinsics'],float);T=np.linalg.inv(np.asarray(cam['extrinsics_base_cam'],float))
 if K.shape!=(3,3) or not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0:raise ValueError('invalid candidate camera')
 segments=boundary_segments(scene,pkg)
 if len(segments):
  p=segments@T[:3,:3].T+T[:3,3];p=p[(p[:,:,2]>.05).all(1)]
  uv=p@K.T;segments=uv[:,:,:2]/uv[:,:,2:]
  segments*=np.array([640/float(cam['width']),640/float(cam['height'])])
 samples=[];sample_weights=[]
 for poly in row['pixel_polylines']:
  for a,b in zip(poly[:-1],poly[1:]):
   a,b=np.asarray(a),np.asarray(b);length=np.linalg.norm(b-a)
   if length<=0:raise ValueError('zero-length source edge')
   n=max(2,int(np.ceil(length)));u=(np.arange(n)+.5)/n
   samples.extend(a+u[:,None]*(b-a));sample_weights.extend([length/n]*n)
 raw=float(np.average(distances(np.asarray(samples),segments),weights=sample_weights)) if len(segments) else None
 counted=min(raw,640.) if raw is not None else 640.
 out['observed_table_boundary']=dict(status='ok',protocol=OBSERVATION_VERSION,mean_distance_px=raw,counted_distance_px=counted,n_observed_samples=len(samples),n_candidate_segments=len(segments),missing_candidate_boundary=not len(segments),annotation_case=row['case'],annotation_uncertainty_px=row['annotation_uncertainty_px'],source_image=row['source_image'],source_image_sha256=row['source_image_sha256'],rule='one-way length-weighted mean on source-visible boundary intervals; delivered camera; no post-hoc alignment; missing boundary counts 640px; no inferred 3D or occluded edges',proxy_support_is_not_boundary_gt=True,candidate_focal_px=[float(K[0,0]),float(K[1,1])])
 return out
