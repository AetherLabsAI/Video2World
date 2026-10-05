"""SE(3) helpers: quaternion math, pose interpolation, frame changes.

Quaternions are wxyz throughout (matching SAPIEN).
Poses are 7-vectors [x y z qw qx qy qz].
"""
from __future__ import annotations

import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).copy()
    q[..., 1:] *= -1
    return q


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) v by quaternion(s) q."""
    v = np.asarray(v, dtype=np.float64)
    qv = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], axis=-1)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[..., 1:]


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return quat_normalize(np.array([w, x, y, z]))


def pose_to_mat(p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(p[3:])
    T[:3, 3] = p[:3]
    return T


def mat_to_pose(T: np.ndarray) -> np.ndarray:
    return np.concatenate([T[:3, 3], mat_to_quat(T[:3, :3])])


def pose_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose poses: a ∘ b."""
    return mat_to_pose(pose_to_mat(a) @ pose_to_mat(b))


def pose_inv(p: np.ndarray) -> np.ndarray:
    T = pose_to_mat(p)
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return mat_to_pose(Ti)


def quat_slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    q0, q1 = quat_normalize(q0), quat_normalize(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        return quat_normalize(q0 + u * (q1 - q0))
    th = np.arccos(np.clip(d, -1, 1))
    return (np.sin((1 - u) * th) * q0 + np.sin(u * th) * q1) / np.sin(th)


def quat_angle(q0: np.ndarray, q1: np.ndarray) -> float:
    """Geodesic angle (rad) between two orientations."""
    d = abs(float(np.dot(quat_normalize(q0), quat_normalize(q1))))
    return 2 * np.arccos(np.clip(d, -1, 1))


def interp_pose_traj(times: np.ndarray, poses: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Piecewise-linear position + slerp orientation interpolation."""
    times = np.asarray(times, dtype=np.float64)
    poses = np.asarray(poses, dtype=np.float64)
    out = np.zeros((len(query), 7))
    for k, t in enumerate(np.asarray(query, dtype=np.float64)):
        i = int(np.clip(np.searchsorted(times, t) - 1, 0, len(times) - 2))
        t0, t1 = times[i], times[i + 1]
        u = 0.0 if t1 <= t0 else float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))
        out[k, :3] = (1 - u) * poses[i, :3] + u * poses[i + 1, :3]
        out[k, 3:] = quat_slerp(poses[i, 3:], poses[i + 1, 3:], u)
    return out


def look_at_quat(approach: np.ndarray, closing: np.ndarray) -> np.ndarray:
    """Gripper orientation from an approach direction (tool z-axis) and a
    closing direction (tool y-axis, finger closing line). Panda TCP convention:
    z points out of the hand, fingers close along y."""
    z = np.asarray(approach, dtype=np.float64)
    z = z / (np.linalg.norm(z) + 1e-12)
    y = np.asarray(closing, dtype=np.float64)
    y = y - z * np.dot(y, z)
    n = np.linalg.norm(y)
    if n < 1e-8:  # pick any perpendicular
        y = np.cross(z, [1.0, 0.0, 0.0])
        if np.linalg.norm(y) < 1e-8:
            y = np.cross(z, [0.0, 1.0, 0.0])
        n = np.linalg.norm(y)
    y = y / n
    x = np.cross(y, z)
    return mat_to_quat(np.stack([x, y, z], axis=1))
