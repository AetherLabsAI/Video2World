"""Cloth metrics against the segmented visible-surface GT (cloth-surface/1).

The GT is the cloth surface the two external cameras saw at each frame, so the predicate metrics are one-sided
(observed -> simulated surface): every observed point must be explained, unobserved cloth is not charged. A footprint
term (convex-hull area in the table plane) closes the gap a one-sided distance leaves open.

Success: terminal observed->sim median <= OBS_TOL_M, footprint change ratio (sim terminal/initial over observed
terminal/initial) within 1 +- AREA_BAND, and the initial state does not already satisfy it. The table-geometry and
motion columns reuse the rigid evaluator's functions on the cloth; the deformable columns reuse the rope/toy
definitions on the simulated surface culled to what the GT cameras could see and voxelised like the GT.
"""
import numpy as np
import trimesh
from scipy.spatial import cKDTree, ConvexHull

from v2w.metrics import geometry as M
from v2w.metrics.trajectory import demo_clock_metrics
from v2w.tracks.twins.task import cd as twin_cd, trajectory as twin_trajectory

OBS_TOL_M = 0.03
AREA_BAND = 0.25
VERSION = 'cloth-surface/1'


def gt_frames(surface):
    P, cnt = surface['particles'], surface['count']
    return [P[i, :int(cnt[i])].astype(float) for i in range(len(cnt))]


DENSIFY = 6


def densify(V, grid, k=DENSIFY):
    """The simulated cloth SURFACE, not its vertices. Distances to a 3 cm vertex grid have a floor of ~1.1 cm for a perfect
    simulation (measured: the measured flat towel against the exactly-placed sheet read 1.26 cm before a single step), which
    would score the grid spacing rather than the fold. Each quad of the x-major (nx, ny) flexcomp grid is sampled bilinearly
    at k x k points."""
    nx, ny = grid; G = np.asarray(V, float).reshape(nx, ny, 3)
    u = np.linspace(0, 1, k, endpoint=False)
    a, b = np.meshgrid(u, u, indexing='ij'); a = a[None, None, :, :, None]; b = b[None, None, :, :, None]
    P00, P10, P01, P11 = G[:-1, :-1, None, None], G[1:, :-1, None, None], G[:-1, 1:, None, None], G[1:, 1:, None, None]
    Q = (1 - a) * (1 - b) * P00 + a * (1 - b) * P10 + (1 - a) * b * P01 + a * b * P11
    return np.concatenate([Q.reshape(-1, 3), G[-1].reshape(-1, 3), G[:, -1].reshape(-1, 3)])


def obs_to_sim(obs, sim, grid):
    if len(obs) == 0: return None
    return cKDTree(densify(sim, grid)).query(obs)[0]


def footprint_area(P, table, above=0.10):
    """Area of the convex hull of the points projected INTO THE TABLE PLANE, keeping only points within `above` of it."""
    pt = np.asarray(table.get('point', [0, 0, table['z']]), float); n = np.asarray(table.get('normal', [0, 0, 1]), float)
    n = n / np.linalg.norm(n); h = (P - pt) @ n; Q = P[h < above]
    if len(Q) < 3: return 0.0
    e1 = np.cross(n, [1.0, 0, 0]); e1 = e1 if np.linalg.norm(e1) > 1e-6 else np.cross(n, [0, 1.0, 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1); uv = (Q - pt) @ np.c_[e1, e2]
    try: return float(ConvexHull(uv).volume)
    except Exception: return 0.0


ZTOL = 0.01          # a simulated point is visible if within 1 cm of the nearest cloth surface in its camera cell
LIFT_M = 0.03        # progress: the rigid families' lift threshold


def observable(P, vis, i):
    """The part of the simulated cloth surface P the GT cameras could have observed at GT frame i: in each camera, in the image,
    not behind the robot (the same FK occluder the GT removed), and front-most in its cell (the cloth hiding itself). Union of the
    two cameras. This makes a two-sided Chamfer against the partially observed GT fair: both sides are the VISIBLE surface."""
    keep = np.zeros(len(P), bool); w, h, dn = int(vis['width']), int(vis['height']), int(vis['down']); W, Hh = w // dn, h // dn
    for c in [str(x) for x in vis['cams']]:
        K = vis['K_' + c]; T = vis['T_base_cam_' + c]; mask = np.unpackbits(vis['robot_mask_' + c][i], axis=-1)[:, :W].astype(bool)
        Pc = (P - T[:3, 3]) @ T[:3, :3]; z = Pc[:, 2]; ok = z > 0.05
        u = np.where(ok, K[0, 0] * Pc[:, 0] / np.where(ok, z, 1) + K[0, 2], -1); v = np.where(ok, K[1, 1] * Pc[:, 1] / np.where(ok, z, 1) + K[1, 2], -1)
        ok &= (u >= 0) & (u < W * dn) & (v >= 0) & (v < Hh * dn)
        cu = (u[ok] // dn).astype(int); cv = (v[ok] // dn).astype(int); idx = np.nonzero(ok)[0]
        free = ~mask[cv, cu]; idx, cu, cv = idx[free], cu[free], cv[free]
        cell = cv * W + cu; zmin = np.full(W * Hh, np.inf); np.minimum.at(zmin, cell, z[idx])
        keep[idx[z[idx] <= zmin[cell] + ZTOL]] = True
    return P[keep]


def voxel(P, s):
    """Same voxel averaging as the GT construction, so the simulated and observed clouds are sampled alike."""
    if len(P) == 0: return P
    key = np.floor(P / s).astype(np.int64); _, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel()
    out = np.zeros((inv.max() + 1, 3)); cnt = np.bincount(inv)
    for d in range(3): out[:, d] = np.bincount(inv, P[:, d]) / cnt
    return out


def as_observed(P, vis, i, vs):
    return voxel(observable(P, vis, i), vs)


def _rigid_columns(surface, executed, vis, G, FR, grid, sim, sim_end):
    """The main table's geometry and motion columns, computed with the rigid evaluator's OWN functions on the cloth.
      Shape CD / Center / Size   v2w.metrics.geometry.shape_align + chamfer on the INITIAL simulated sheet vs the INITIAL observed
                                 surface (the flat towel is fully visible) -- reconstruction geometry, as for a rigid object;
                                 size = sorted full lengths of independently fitted oriented bounding boxes
      Trans. APE / RPE           v2w.metrics.trajectory.demo_clock_metrics on the visible-surface CENTROID sequences, each
                                 re-based on its own start (the rigid protocol). Cloth has no orientation: the quaternions are
                                 identity placeholders, so every rotation output is removed rather than reported as 0.
      Rel. position              terminal displacement error, (sim end - sim start) - (observed end - observed start), of the
                                 visible-surface centroid: a fold has no receiver, so the anchor is the cloth's own start."""
    A, B = sim[0], G[0]
    sh, Rs, _ = M.shape_align(A, B)
    cb = B.mean(0); _, _, Vb = np.linalg.svd(B - cb)
    Aal = (A - A.mean(0)) @ Rs.T
    ec = np.ptp(Aal @ Vb.T, axis=0); eg = np.ptp((B - cb) @ Vb.T, axis=0)
    _, oc = trimesh.bounds.oriented_bounds(A,angle_digits=5); _, og = trimesh.bounds.oriented_bounds(B,angle_digits=5)
    oc,og=np.sort(oc),np.sort(og)
    obj = dict(size_protocol='intrinsic-obb-full-extents/2', size_protocol_note='independent OBB full lengths of visibility-matched initial point sets; ascending dimensions',
               legacy_size_alignment_rotation=Rs.tolist(),
               shape_cd_cm=100 * sh['chamfer_m'], shape_a_to_b_cm=100 * sh['a_to_b_m'], shape_b_to_a_cm=100 * sh['b_to_a_m'],
               center_err_cm=100 * float(np.linalg.norm(A.mean(0) - B.mean(0))), size_err_cm=100 * float(np.linalg.norm(oc - og)),
               size_cand_cm=(100 * oc).tolist(), size_gt_cm=(100 * og).tolist(), frame='initial',
               legacy_size_err_in_plane_cm=100 * float(np.linalg.norm(ec[:2] - eg[:2])), legacy_size_err_out_of_plane_cm=100 * float(abs(ec[2] - eg[2])),
               size_note='Same independent OBB definition as rigid geometry. Partial visibility and depth-noise limits remain. Legacy diagnostic third axis is the observed surface '
                         'normal: a flat simulated sheet against a real towel with wrinkles and depth noise, and full extents are '
                         'sensitive to a few outlying points -- read the in-plane / out-of-plane split before interpreting it.')
    q = np.tile([1.0, 0, 0, 0], (len(G), 1))
    seqS = np.c_[np.stack([x.mean(0) for x in sim]), q]; seqG = np.c_[np.stack([g.mean(0) for g in G]), q]
    tt = np.asarray(surface['time_s'], float); dt = float(np.median(np.diff(tt)))
    own = demo_clock_metrics(seqS, dt, seqG, dt)
    for blk in (own.get('ape') or {}, own.get('rpe') or {}, (own.get('duration_normalized') or {}).get('ape') or {}, (own.get('duration_normalized') or {}).get('rpe') or {}):
        for k in [k for k in blk if k.startswith('rot_')]: blk[k] = None   # numeric rotation outputs only; NOT 'protocol', which also contains 'rot'
    own['rotation'] = 'not defined for cloth (identity placeholders); all rotation outputs removed'
    own['source'] = 'visible-surface centroid, simulated surface culled to the GT cameras and voxelised like the GT'
    dS = sim_end.mean(0) - sim[0].mean(0); dG = G[-1].mean(0) - G[0].mean(0)
    rel = dict(rel_trans_err_cm=100 * float(np.linalg.norm(dS - dG)), anchor='initial_object', protocol='cloth-terminal/1',
               frame='robot base', endpoint_rule='simulated final state after execution vs the last observed frame',
               note='a fold has no receiver; the terminal displacement of the visible cloth is compared with the demonstration')
    return dict(object=obj, own_trajectory=own, relative=rel)


def progress_record(executed, V, table, success, grid):
    pt = np.asarray(table.get('point', [0, 0, table['z']]), float); n = np.asarray(table.get('normal', [0, 0, 1]), float); n = n / np.linalg.norm(n)
    g, r = executed.get('grasp_frame'), executed.get('release_frame')
    grasp = bool(executed.get('captured', 0) > 0 and g is not None)
    lift = bool(grasp and max(float(((V[k] - pt) @ n).max()) for k in range(g, (r if r is not None else len(V) - 1) + 1)) >= LIFT_M)
    release = bool(lift and r is not None and r > g)
    done = [grasp, lift, release, bool(success)]
    achieved = 4 if success else next((j for j, x in enumerate(done) if not x), 3)
    return dict(protocol='cloth-progress/1', kind='cloth_fold', source='own_actions',
                stages=[dict(name=nm, done=d) for nm, d in zip(['grasp', 'lift', 'release', 'folded'], done)],
                achieved=achieved, total=4, progress=achieved / 4,
                rule='grasp = the closing pads captured cloth; lift = cloth rose >= 3 cm above the table while held; release = the '
                     'gripper opened after the lift; folded = the terminal predicate. Ordered prefix; terminal success is completion.')


def _deformable(surface, executed, vis, G, FR, grid):
    """The rope/toy family's deformable columns, same cd() and trajectory() functions, on the visibility-culled simulated surface."""
    if vis is None: return {}
    V = executed['vertices']; tgt = np.asarray(surface['time_s'], float); vs = float(surface['voxel'])
    sim = [as_observed(densify(V[min(k, len(V) - 1)], grid), vis, i, vs) for i, k in enumerate(FR)]
    sim_end = as_observed(densify(executed['final'], grid), vis, len(FR) - 1, vs)
    s0, g0 = sim[0], G[0]
    d = dict(initial_surface_cd_cm=twin_cd(s0, g0),
             surface_center_err_cm=float(np.linalg.norm(s0.mean(0) - g0.mean(0)) * 100),
             world_aabb_size_err_cm=float(np.linalg.norm(np.sort(np.ptp(s0, axis=0)) - np.sort(np.ptp(g0, axis=0))) * 100),
             terminal_surface_cd_cm=twin_cd(sim_end, G[-1]),
             trajectory=twin_trajectory(sim, tgt, G, tgt),
             culled_points=dict(first=len(s0), last=len(sim_end), per_frame=[len(x) for x in sim]),
             definition='twin cd / trajectory (two-sided, the rope/toy definitions), with the simulated '
                        'surface culled to what the GT cameras could observe (robot occluder + cloth self-occlusion) and voxelised '
                        'like the GT, because the GT is the visible surface only')
    return dict(deformable=d, **_rigid_columns(surface, executed, vis, G, FR, grid, sim, sim_end))


def score(surface, executed, table, visibility=None):
    """surface: surface.npz dict-like (frames, time_s, particles, count). executed: executor result with 'vertices' (T,Nv,3)
    indexed by demo frame and 'final' (Nv,3). Returns the metric record."""
    FR = [int(k) for k in surface['frames']]; G = gt_frames(surface); V = executed['vertices']; grid = executed['grid']
    per = []
    for i, k in enumerate(FR):
        d = obs_to_sim(G[i], V[min(k, len(V) - 1)], grid)
        per.append(None if d is None else dict(frame=k, n_obs=len(G[i]), median_cm=float(np.median(d) * 100),
                                               mean_cm=float(d.mean() * 100), p90_cm=float(np.percentile(d, 90) * 100),
                                               within_tol=float(np.mean(d <= OBS_TOL_M))))
    valid = [p for p in per if p is not None]
    g0, gT = G[0], G[-1]
    d0 = obs_to_sim(g0, V[FR[0]], grid); dT = obs_to_sim(gT, executed['final'], grid)
    a_obs0, a_obsT = footprint_area(g0, table), footprint_area(gT, table)
    a_sim0, a_simT = footprint_area(V[FR[0]], table), footprint_area(executed['final'], table)
    obs_change = a_obsT / a_obs0 if a_obs0 > 0 else float('nan')
    ratioT = (a_simT / a_sim0) / obs_change if a_sim0 > 0 and obs_change > 0 else float('inf')
    # the unfolded start judged as if it were the end: no change in the simulation against the observed halving
    ratio0 = 1.0 / obs_change if obs_change > 0 else float('inf')
    ok_T = bool(np.median(dT) <= OBS_TOL_M and abs(ratioT - 1) <= AREA_BAND)
    ok_0 = bool(np.median(d0) <= OBS_TOL_M and abs(ratio0 - 1) <= AREA_BAND)
    return dict(
        protocol=VERSION,
        metric_definitions=dict(
            observed_to_sim=('one-sided: for each OBSERVED cloth point, distance to the simulated cloth SURFACE (bilinear %dx%d per '
                             'quad) (cm). Unobserved cloth is not scored. Not the twin family two-sided Chamfer.' % (DENSIFY, DENSIFY)),
            footprint='convex-hull area of cloth points projected into the TABLE PLANE, keeping those within 10 cm of it (m^2)'),
        initial=dict(observed_to_sim_median_cm=float(np.median(d0) * 100), observed_to_sim_p90_cm=float(np.percentile(d0, 90) * 100),
                     footprint_sim_m2=a_sim0, footprint_obs_m2=a_obs0),
        trajectory=dict(observed_to_sim_median_cm=float(np.median([p['median_cm'] for p in valid])),
                        observed_to_sim_mean_cm=float(np.mean([p['mean_cm'] for p in valid])),
                        frames_scored=len(valid), frames_total=len(per), per_frame=per),
        terminal=dict(observed_to_sim_median_cm=float(np.median(dT) * 100), observed_to_sim_p90_cm=float(np.percentile(dT, 90) * 100),
                      within_tol=float(np.mean(dT <= OBS_TOL_M)), footprint_sim_m2=a_simT, footprint_obs_m2=a_obsT,
                      footprint_change_sim=a_simT / a_sim0 if a_sim0 > 0 else None, footprint_change_obs=obs_change,
                      footprint_change_ratio=ratioT, footprint_abs_ratio=a_simT / a_obsT if a_obsT > 0 else None, n_obs=len(gT)),
        predicate=dict(obs_tol_m=OBS_TOL_M, area_band=AREA_BAND, initial_satisfied=ok_0, terminal_satisfied=ok_T,
                       area_term='footprint change ratio (terminal/initial, sim vs observed)', initial_change_ratio=ratio0),
        task_success=bool(ok_T and not ok_0),
        **_deformable(surface, executed, visibility, G, FR, grid),
        progress=progress_record(executed, V, table, bool(ok_T and not ok_0), grid),
        grasp=dict(captured_vertices=executed['captured'], grasp_frame=executed['grasp_frame'], release_frame=executed['release_frame']))
