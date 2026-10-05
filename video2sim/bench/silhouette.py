"""Fast analytic silhouettes: project a mesh through a camera and rasterise
its triangles into a binary mask (cv2.fillPoly). Used by the model-based
pose fitting that produces the benchmark's quasi ground truth.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


class Projector:
    def __init__(self, cam: dict):
        self.K = np.asarray(cam["intrinsics"], dtype=np.float64)
        T_base_cam = np.asarray(cam["extrinsics_base_cam"], dtype=np.float64)
        self.T_cam_base = np.linalg.inv(T_base_cam)
        self.w, self.h = int(cam["width"]), int(cam["height"])

    def project(self, pts_base: np.ndarray):
        p = pts_base @ self.T_cam_base[:3, :3].T + self.T_cam_base[:3, 3]
        z = p[:, 2]
        uv = (p[:, :2] / np.maximum(z[:, None], 1e-6)) * np.array([self.K[0, 0], self.K[1, 1]]) + np.array([self.K[0, 2], self.K[1, 2]])
        return uv, z

    def silhouette(self, verts_obj: np.ndarray, faces: np.ndarray, pos, quat_wxyz) -> np.ndarray:
        import cv2
        q = np.asarray(quat_wxyz, dtype=np.float64)
        Rm = R.from_quat(np.r_[q[1:], q[0]]).as_matrix()
        v = verts_obj @ Rm.T + np.asarray(pos, dtype=np.float64)
        uv, z = self.project(v)
        mask = np.zeros((self.h, self.w), np.uint8)
        ok = z > 0.02
        tri = uv[faces]                       # (F, 3, 2)
        keep = ok[faces].all(1)
        tri = tri[keep]
        if len(tri) == 0:
            return mask
        tri = np.clip(tri, -1e4, 1e4).astype(np.int32)
        cv2.fillPoly(mask, list(tri), 1)
        return mask


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def load_mesh(entity: dict):
    """trimesh for a bench entity (library / mesh / box / cylinder), object frame."""
    import trimesh
    from .bridge_env import library_model_dir, _visual_file
    k = entity["kind"]
    if k == "library":
        d = library_model_dir(entity["library_id"])
        try:
            m = trimesh.load(_visual_file(d), force="mesh")
        except Exception:
            m = trimesh.load(str(d / "collision.obj"), force="mesh")
    elif k == "mesh":
        m = trimesh.load(entity.get("mesh_path") or entity["collision_path"], force="mesh")
    elif k == "box":
        m = trimesh.creation.box(extents=2 * np.asarray(entity["half_size"], float))
    elif k == "cylinder":
        m = trimesh.creation.cylinder(radius=float(entity["half_size"][0]), height=2 * float(entity["half_size"][2]))
    elif k == "sphere":
        m = trimesh.creation.icosphere(subdivisions=2, radius=float(entity["half_size"][0]))
    elif k == "container":
        hx, hy, hh = (float(v) for v in entity["half_size"]); t = float(entity.get("wall", 0.004))
        parts = []
        for off, half in [((0, 0, -t / 2), (hx + t, hy + t, t / 2)), ((hx + t / 2, 0, hh), (t / 2, hy + t, hh)),
                          ((-hx - t / 2, 0, hh), (t / 2, hy + t, hh)), ((0, hy + t / 2, hh), (hx + t, t / 2, hh)),
                          ((0, -hy - t / 2, hh), (hx + t, t / 2, hh))]:
            b = trimesh.creation.box(extents=2 * np.asarray(half)); b.apply_translation(off); parts.append(b)
        m = trimesh.util.concatenate(parts)
    else:
        raise ValueError(k)
    m.apply_scale(float(entity.get("scale", 1.0)))
    return m


def yaw_quat(yaw: float, base_quat=(1, 0, 0, 0)) -> np.ndarray:
    """quat (wxyz) = Rz(yaw) * base_quat."""
    q = np.asarray(base_quat, float)
    r = R.from_euler("z", yaw) * R.from_quat(np.r_[q[1:], q[0]])
    s = r.as_quat()
    return np.array([s[3], s[0], s[1], s[2]])
