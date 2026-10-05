"""Deterministic geometry metrics in metres; public output in centimetres."""
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def transform(points, pose):
    pose = np.asarray(pose)
    return np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]


def pose_matrix(pos, quat):
    p = np.eye(4)
    p[:3, :3] = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()
    p[:3, 3] = pos
    return p


def sample_mesh(mesh, n=4000, seed=0):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all() or mesh.area <= 0:
        raise ValueError('Geometry must have finite, nonzero surface area')
    rng = np.random.default_rng(seed)
    triangles = mesh.triangles[rng.choice(len(mesh.faces), n, p=mesh.area_faces / mesh.area)]
    uv = rng.random((n, 2)); flip = uv.sum(axis=1) > 1; uv[flip] = 1 - uv[flip]
    return triangles[:, 0] + uv[:, :1] * (triangles[:, 1] - triangles[:, 0]) + uv[:, 1:] * (triangles[:, 2] - triangles[:, 0])


def distances(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if min(len(a), len(b)) < 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Empty/nonfinite point cloud')
    return cKDTree(b).query(a)[0], cKDTree(a).query(b)[0]


def chamfer_cm(a, b):
    x, y = distances(a, b)
    return float((x.mean() + y.mean()) * 100)  # existing benchmark sum convention


def geometry_metrics(candidate, reference):
    # Position/size are cloud bounds diagnostics, not pose GT / rigid APE.
    a, b = np.asarray(candidate), np.asarray(reference)
    return {'chamfer_cm': chamfer_cm(a, b),
            'surface_bbox_center_err_cm': float(np.linalg.norm((a.max(0)+a.min(0)-b.max(0)-b.min(0))/2)*100),
            'surface_bbox_size_err_cm': float(np.linalg.norm(np.sort(np.ptp(a, axis=0))-np.sort(np.ptp(b, axis=0)))*100),
            'center_definition': 'surface_AABB_midpoint', 'size_definition': 'sorted_world_AABB_extents'}
