"""Benchmark metrics. Pure numpy/torch; no simulator import here.

    chamfer(A, B)             symmetric Chamfer distance between point clouds (m)
    lpips_batch(a, b)         mean LPIPS(alex) over paired uint8 images
    pose_ape(traj_a, traj_b)  translation / rotation APE over aligned trajectories
    sym_rot_err(qa, qb, sym)  symmetry-aware geodesic rotation error (deg)
    match_objects(...)        agent<->reference object correspondence by initial pose
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R


# ----------------------------------------------------------------- geometry
def chamfer(A: np.ndarray, B: np.ndarray) -> dict:
    """Bidirectional Chamfer distance in metres: mean over A of nearest B plus
    mean over B of nearest A (the usual sum form), plus each direction."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if len(A) == 0 or len(B) == 0:
        return {"chamfer_m": float("nan"), "a_to_b_m": float("nan"), "b_to_a_m": float("nan")}
    dab = cKDTree(B).query(A, k=1)[0]
    dba = cKDTree(A).query(B, k=1)[0]
    return {"chamfer_m": float(dab.mean() + dba.mean()),
            "a_to_b_m": float(dab.mean()), "b_to_a_m": float(dba.mean()),
            "a_to_b_p95_m": float(np.percentile(dab, 95)),
            "b_to_a_p95_m": float(np.percentile(dba, 95))}


def quat_wxyz_to_R(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return R.from_quat(np.r_[q[1:], q[0]]).as_matrix()


def sym_rot_err(qa, qb, symmetry: str | None = None, axis: str = "z") -> float:
    """Geodesic angle (deg) between two orientations, minimised over the
    object's symmetry group. `symmetry`: None | "axis" (continuous, about
    `axis`) | "box" (24 cube rotations) | "box2" (rectangular box: 4 rotations
    about each axis is too permissive; use the 8 that keep the box's long
    axis) — for the benchmark we use None / axis / box."""
    Ra, Rb = quat_wxyz_to_R(qa), quat_wxyz_to_R(qb)
    D = Ra.T @ Rb
    if symmetry is None:
        return float(np.degrees(np.linalg.norm(R.from_matrix(D).as_rotvec())))
    if symmetry == "axis":
        i = "xyz".index(axis)
        # residual rotation after removing the component about the axis: the
        # angle between the two symmetry axes
        a_axis, b_axis = Ra[:, i], Rb[:, i]
        c = float(np.clip(np.dot(a_axis, b_axis), -1.0, 1.0))
        return float(np.degrees(np.arccos(c)))
    if symmetry == "box":
        best = 1e9
        for g in _cube_group():
            best = min(best, np.linalg.norm(R.from_matrix(D @ g).as_rotvec()))
        return float(np.degrees(best))
    raise ValueError(symmetry)


_CUBE = None


def _cube_group():
    global _CUBE
    if _CUBE is None:
        mats = []
        for p in ([0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]):
            for s in np.array(np.meshgrid([1, -1], [1, -1], [1, -1])).T.reshape(-1, 3):
                M = np.zeros((3, 3))
                for r, c in enumerate(p):
                    M[r, c] = s[r]
                if np.isclose(np.linalg.det(M), 1.0):
                    mats.append(M)
        _CUBE = mats
    return _CUBE


def pose_ape(traj_a: np.ndarray, traj_b: np.ndarray, symmetry: str | None = None,
             axis: str = "z") -> dict:
    """APE between two (T, 7) [x y z qw qx qy qz] trajectories on the same
    clock (resampled to the shorter one). Translation in cm, rotation in deg."""
    n = min(len(traj_a), len(traj_b))
    if n == 0:
        return {"trans_ape_cm": float("nan"), "rot_ape_deg": float("nan"), "n": 0}
    a = _resample(np.asarray(traj_a), n)
    b = _resample(np.asarray(traj_b), n)
    t = np.linalg.norm(a[:, :3] - b[:, :3], axis=1) * 100.0
    r = np.array([sym_rot_err(a[k, 3:], b[k, 3:], symmetry, axis) for k in range(n)])
    return {"trans_ape_cm": float(t.mean()), "trans_final_cm": float(t[-1]),
            "trans_max_cm": float(t.max()),
            "rot_ape_deg": float(r.mean()), "rot_final_deg": float(r[-1]), "n": int(n)}


def _resample(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == n:
        return x
    idx = np.linspace(0, len(x) - 1, n).round().astype(int)
    return x[idx]


def match_objects(agent_init: dict[str, np.ndarray], ref_init: dict[str, np.ndarray],
                  max_dist: float = 0.25) -> dict[str, str | None]:
    """Reference object name -> agent object name.

    1. Name evidence first: a unique agent name that contains the reference
       name (or vice versa) is paired outright — packages name their objects
       informatively ("green_block" for "green").
    2. The rest: optimal assignment (Hungarian) on initial-position distance,
       pairs beyond `max_dist` dropped.
    3. A single-object episode is paired with the nearest agent object
       regardless of distance — there is nothing else it could be.
    No ICP, no size normalisation: pose error stays in the metrics.
    """
    out: dict[str, str | None] = {rn: None for rn in ref_init}
    free_a = dict(agent_init)
    for rn in sorted(ref_init):
        cands = [an for an in free_a if rn.lower() in an.lower() or an.lower() in rn.lower()]
        if len(cands) == 1:
            out[rn] = cands[0]
            free_a.pop(cands[0])
    rest_r = [rn for rn in sorted(ref_init) if out[rn] is None]
    if rest_r and free_a:
        from scipy.optimize import linear_sum_assignment
        names_a = list(free_a)
        D = np.array([[np.linalg.norm(np.asarray(ref_init[rn])[:3] - np.asarray(free_a[an])[:3])
                       for an in names_a] for rn in rest_r])
        ri, ai = linear_sum_assignment(D)
        for i, j in zip(ri, ai):
            if D[i, j] <= max_dist:
                out[rest_r[i]] = names_a[j]
    if len(ref_init) == 1 and next(iter(out.values())) is None and agent_init:
        rn = next(iter(ref_init))
        names_a = list(agent_init)
        d = [np.linalg.norm(np.asarray(agent_init[an])[:3] - np.asarray(ref_init[rn])[:3]) for an in names_a]
        out[rn] = names_a[int(np.argmin(d))]
    return out


# ------------------------------------------------------------------- images
_LPIPS = None


def lpips_batch(a: np.ndarray, b: np.ndarray, net: str = "alex") -> np.ndarray:
    """Per-pair LPIPS for uint8 HxWx3 arrays (N, H, W, 3). Runs on CPU (tiny
    batches; keeps the GPU free for the renderer)."""
    global _LPIPS
    import torch
    import lpips
    if _LPIPS is None:
        _LPIPS = lpips.LPIPS(net=net, verbose=False).eval()
    a = torch.from_numpy(np.asarray(a)).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    b = torch.from_numpy(np.asarray(b)).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    with torch.no_grad():
        d = _LPIPS(a, b)
    return d.flatten().cpu().numpy()


def resize_like_bridge(img: np.ndarray, size: int = 256) -> np.ndarray:
    """Bridge frames are 640x480 captures resized (non-uniformly) to 256x256;
    apply the same to a rendered 640x480 frame before comparing."""
    from PIL import Image
    return np.asarray(Image.fromarray(np.asarray(img).astype(np.uint8)).resize((size, size), Image.BILINEAR))
