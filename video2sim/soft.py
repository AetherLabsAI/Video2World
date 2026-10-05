"""Soft-body backend (protocol physics_profile "soft_warp").

PhysTwin-style particle cloth (spring/distance constraints) simulated with
NVIDIA Warp on GPU — no SAPIEN, no Vulkan, safe on the shared machine.

Model
-----
- cloth_grid: nx*ny particles, structural + shear + bend distance
  constraints, solved with ping-pong Jacobi PBD (order-independent =>
  deterministic across runs on the same wheel/GPU).
- Robot: KINEMATIC — the recorded ``pd_joint_pos`` action stream (T, 8) is
  assumed tracked perfectly; FK gives the TCP frame, finger pads and palm
  become moving collision boxes. Grasp = pinch attachment: when the gripper
  command closes below ``attach_width`` the particles inside the pinch box
  are rigidly attached to the TCP frame; opening releases them.
- Collisions: table plane + kinematic boxes, projection with tangential
  velocity damping (friction proxy).

The cloth can only move through gravity, constraints, contact and the pinch
attachment — the same "no set_pose after init" principle as the rigid env.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

GRAVITY = wp.vec3(0.0, 0.0, -9.81)

# Panda "rest" keyframe (ManiSkill), arm 7 + fingers 2
PANDA_REST = np.array([0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, np.pi / 2, np.pi / 4, 0.04, 0.04])


# ------------------------------------------------------------------ kernels
@wp.kernel
def k_predict(
    p: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    p_prev: wp.array(dtype=wp.vec3),
    dt: float,
    damping: float,
):
    i = wp.tid()
    vel = (v[i] + GRAVITY * dt) * damping
    v[i] = vel
    p_prev[i] = p[i]
    p[i] = p[i] + vel * dt


@wp.kernel
def k_jacobi(
    p_in: wp.array(dtype=wp.vec3),
    p_out: wp.array(dtype=wp.vec3),
    offsets: wp.array(dtype=int),
    neighbors: wp.array(dtype=int),
    rest: wp.array(dtype=float),
    stiff: wp.array(dtype=float),
    omega: float,
):
    i = wp.tid()
    xi = p_in[i]
    corr = wp.vec3(0.0, 0.0, 0.0)
    n = offsets[i + 1] - offsets[i]
    for k in range(offsets[i], offsets[i + 1]):
        j = neighbors[k]
        d = xi - p_in[j]
        dist = wp.length(d)
        if dist > 1.0e-9:
            c = dist - rest[k]
            corr = corr - d * (0.5 * stiff[k] * c / dist)
    if n > 0:
        p_out[i] = xi + corr * (omega / float(n))
    else:
        p_out[i] = xi


@wp.kernel
def k_selfcol(
    p_in: wp.array(dtype=wp.vec3),
    p_prev: wp.array(dtype=wp.vec3),
    p_out: wp.array(dtype=wp.vec3),
    offsets: wp.array(dtype=int),
    neighbors: wp.array(dtype=int),
    d_min: float,
    friction: float,
):
    # Brute-force all-pairs: wp.HashGrid queries silently miss neighbors on
    # this wheel (asymmetric, point-order dependent), and at cloth sizes
    # (<= a few thousand particles) O(n^2) is cheap and iteration order is
    # fixed, so the sum is deterministic.
    i = wp.tid()
    xi = p_in[i]
    corr = wp.vec3(0.0, 0.0, 0.0)
    cnt = int(0)
    for j in range(p_in.shape[0]):
        if j != i:
            skip = int(0)
            for k in range(offsets[i], offsets[i + 1]):
                if neighbors[k] == j:
                    skip = 1
            if skip == 0:
                d = xi - p_in[j]
                dist = wp.length(d)
                if dist < d_min and dist > 1.0e-9:
                    n = d / dist
                    corr = corr + n * (0.5 * (d_min - dist))
                    if friction > 0.0:
                        # PBD contact friction: damp the relative tangential
                        # displacement accumulated this substep, so stacked
                        # cloth layers do not slide freely over each other
                        rel = (xi - p_prev[i]) - (p_in[j] - p_prev[j])
                        tang = rel - n * wp.dot(rel, n)
                        corr = corr - tang * (0.5 * friction)
                    cnt = cnt + 1
    if cnt > 0:
        p_out[i] = xi + corr / float(cnt)
    else:
        p_out[i] = xi


@wp.kernel
def k_mesh_collide(
    mesh: wp.uint64,
    p: wp.array(dtype=wp.vec3),
    p_prev: wp.array(dtype=wp.vec3),
    thickness: float,
    attach_flag: wp.array(dtype=int),
):
    i = wp.tid()
    if attach_flag[i] == 1:
        return
    x = p[i]
    q = wp.mesh_query_point_sign_normal(mesh, x, 0.05)
    if q.result:
        cp = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        d = x - cp
        dist = wp.length(d)
        if q.sign < 0.0:
            # inside: exit through the closest surface point
            if dist > 1.0e-9:
                x = cp - d * (thickness / dist)
            else:
                x = cp + wp.vec3(0.0, 0.0, thickness)
            prev = p_prev[i]
            x = wp.vec3(x[0] - 0.5 * (x[0] - prev[0]),
                        x[1] - 0.5 * (x[1] - prev[1]), x[2])
            p[i] = x
        elif dist < thickness and dist > 1.0e-9:
            p[i] = cp + d * (thickness / dist)


@wp.kernel
def k_arm_collide(
    mesh_ids: wp.array(dtype=wp.uint64),
    link_tf: wp.array(dtype=wp.transformf),
    n_links: int,
    p: wp.array(dtype=wp.vec3),
    thickness: float,
    attach_flag: wp.array(dtype=int),
):
    i = wp.tid()
    if attach_flag[i] == 1:
        return
    x = p[i]
    for l in range(n_links):
        xl = wp.transform_point(wp.transform_inverse(link_tf[l]), x)
        q = wp.mesh_query_point_sign_normal(mesh_ids[l], xl, 0.04)
        if q.result:
            cp = wp.mesh_eval_position(mesh_ids[l], q.face, q.u, q.v)
            d = xl - cp
            dist = wp.length(d)
            if q.sign < 0.0 and dist > 1.0e-9:
                xl = cp - d * (thickness / dist)
                x = wp.transform_point(link_tf[l], xl)
            elif dist < thickness and dist > 1.0e-9:
                xl = cp + d * (thickness / dist)
                x = wp.transform_point(link_tf[l], xl)
    p[i] = x


@wp.kernel
def k_project(
    p: wp.array(dtype=wp.vec3),
    p_prev: wp.array(dtype=wp.vec3),
    table_z: float,
    thickness: float,
    boxes: wp.array(dtype=wp.transformf),
    box_half: wp.array(dtype=wp.vec3),
    n_boxes: int,
    attach_flag: wp.array(dtype=int),
    attach_local: wp.array(dtype=wp.vec3),
    hand_tf: wp.array(dtype=wp.transformf),
    stick_disp: float,
):
    i = wp.tid()
    if attach_flag[i] == 1:
        p[i] = wp.transform_point(hand_tf[0], attach_local[i])
        return
    x = p[i]
    # table plane with simple friction: kill tangential motion on contact;
    # below stick_disp (per-substep displacement) kill it entirely — stiction.
    # Without it, solver-asymmetry thrust makes settled stacks glide forever.
    if x[2] < table_z + thickness:
        x = wp.vec3(x[0], x[1], table_z + thickness)
        prev = p_prev[i]
        tx = x[0] - prev[0]
        ty = x[1] - prev[1]
        if wp.sqrt(tx * tx + ty * ty) < stick_disp:
            x = wp.vec3(prev[0], prev[1], x[2])
        else:
            x = wp.vec3(x[0] - 0.6 * tx, x[1] - 0.6 * ty, x[2])
    # kinematic boxes (finger pads, palm): push out along min-penetration axis
    for b in range(n_boxes):
        lp = wp.transform_point(wp.transform_inverse(boxes[b]), x)
        h = box_half[b]
        dx = h[0] + thickness - wp.abs(lp[0])
        dy = h[1] + thickness - wp.abs(lp[1])
        dz = h[2] + thickness - wp.abs(lp[2])
        if dx > 0.0 and dy > 0.0 and dz > 0.0:
            if dx <= dy and dx <= dz:
                lp = wp.vec3(wp.sign(lp[0]) * (h[0] + thickness), lp[1], lp[2])
            elif dy <= dz:
                lp = wp.vec3(lp[0], wp.sign(lp[1]) * (h[1] + thickness), lp[2])
            else:
                lp = wp.vec3(lp[0], lp[1], wp.sign(lp[2]) * (h[2] + thickness))
            x = wp.transform_point(boxes[b], lp)
    p[i] = x


@wp.kernel
def k_velocity(
    p: wp.array(dtype=wp.vec3),
    p_prev: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    inv_dt: float,
):
    i = wp.tid()
    v[i] = (p[i] - p_prev[i]) * inv_dt


# ------------------------------------------------------------------ FK / IK
class KinematicPanda:
    """FK/IK on the ManiSkill panda URDF without any simulator."""

    def __init__(self, urdf_path: str | None = None, base_pose: np.ndarray | None = None):
        import pytorch_kinematics as pk
        import torch

        if urdf_path is None:
            from mani_skill.agents.robots.panda.panda import Panda
            urdf_path = Panda.urdf_path
        with open(urdf_path, "rb") as f:
            self.chain = pk.build_serial_chain_from_urdf(f.read(), "panda_hand_tcp")
        # device pinned: importing genesis flips torch's default device, and an
        # unpinned chain then mixes CPU buffers with CUDA tensors (only
        # reproduces when both backends run in one process, e.g. the test suite)
        self.chain = self.chain.to(dtype=torch.float64, device="cpu")
        self.base_pose = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float64) \
            if base_pose is None else np.asarray(base_pose, dtype=np.float64)

    def fk(self, q7: np.ndarray) -> np.ndarray:
        """TCP pose7 in the WORLD frame."""
        import torch

        from .transforms import mat_to_pose, pose_mul
        m = self.chain.forward_kinematics(torch.as_tensor(q7, dtype=torch.float64, device="cpu")[None])
        return pose_mul(self.base_pose, mat_to_pose(m.get_matrix()[0].numpy()))

    def link_polyline(self, q7: np.ndarray) -> np.ndarray:
        """World positions of every serial-chain frame origin (arm skeleton)."""
        import torch

        from .transforms import quat_rotate
        frames = self.chain.forward_kinematics(
            torch.as_tensor(q7, dtype=torch.float64, device="cpu")[None], end_only=False)
        pts = [np.zeros(3)]
        for name in self.chain.get_frame_names(exclude_fixed=False):
            try:
                M = frames[name].get_matrix()[0].numpy()
            except KeyError:
                continue
            pts.append(M[:3, 3])
        pts = np.asarray(pts)
        return quat_rotate(self.base_pose[3:], pts) + self.base_pose[:3]

    def ik(self, q7: np.ndarray, target_world: np.ndarray,
           iters: int = 12, damping: float = 1e-3, max_dq: float = 0.2) -> np.ndarray:
        import torch

        from .transforms import pose_inv, pose_mul, quat_to_mat
        target = pose_mul(pose_inv(self.base_pose), target_world)
        q = q7.astype(np.float64).copy()
        for _ in range(iters):
            qt = torch.as_tensor(q, dtype=torch.float64, device="cpu")[None]
            J = self.chain.jacobian(qt)[0].numpy()
            m = self.chain.forward_kinematics(qt).get_matrix()[0].numpy()
            e_p = target[:3] - m[:3, 3]
            R_err = quat_to_mat(target[3:]) @ m[:3, :3].T
            w = 0.5 * np.array([R_err[2, 1] - R_err[1, 2],
                                R_err[0, 2] - R_err[2, 0],
                                R_err[1, 0] - R_err[0, 1]])
            e = np.concatenate([e_p, w])
            if np.linalg.norm(e_p) < 1e-4 and np.linalg.norm(w) < 1e-3:
                break
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
            q = q + np.clip(dq, -max_dq, max_dq)
        return q


def grip_norm_to_width(a: float) -> float:
    """Inverse of rollout._grip_cmd: normalized [-1,1] -> opening in m."""
    return 2 * ((a + 1) / 2 * 0.05 - 0.01)


# ------------------------------------------------------------------ cloth
@dataclass
class ClothSpec:
    name: str = "cloth"
    kind: str = "cloth_grid"
    nx: int = 16
    ny: int = 16
    spacing: float = 0.0125
    origin: tuple = (0.0, 0.0, 0.005)   # corner particle (0,0) position
    mass_total: float = 0.05
    iters: int = 20
    substeps: int = 8
    damping: float = 0.995
    thickness: float = 0.004
    attach_width: float = 0.02          # cmd width below which pinch attaches
    release_width: float = 0.035
    bend_stiff: float = 0.35
    yaw_deg: float = 0.0                # grid rotation about +z through origin
    self_collision: bool = True         # HashGrid particle repulsion
    self_dist: float = 0.6              # d_min = self_dist * spacing
    self_friction: float = 0.0          # tangential damping between colliding
                                        #   particles (cloth-on-cloth friction)
    table_stick_speed: float = 0.0      # m/s; below this, table contact is
                                        #   fully sticking (static friction)

    def as_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in self.__dict__.items()}

    @staticmethod
    def from_dict(d: dict) -> "ClothSpec":
        d = dict(d)
        d["origin"] = tuple(d.get("origin", (0, 0, 0.005)))
        return ClothSpec(**d)


def _grid(spec: ClothSpec):
    xs = np.arange(spec.nx) * spec.spacing
    ys = np.arange(spec.ny) * spec.spacing
    P = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
    th = np.radians(spec.yaw_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pts = np.concatenate([P @ R.T + np.asarray(spec.origin[:2]),
                          np.full((len(P), 1), spec.origin[2])], 1)
    idx = np.arange(spec.nx * spec.ny).reshape(spec.nx, spec.ny)
    edges = []
    for di, dj, k in [(1, 0, 1.0), (0, 1, 1.0), (1, 1, 0.7), (1, -1, 0.7),
                      (2, 0, None), (0, 2, None)]:
        kk = spec.bend_stiff if k is None else k
        for i in range(spec.nx):
            for j in range(spec.ny):
                i2, j2 = i + di, j + dj
                if 0 <= i2 < spec.nx and 0 <= j2 < spec.ny:
                    edges.append((idx[i, j], idx[i2, j2], kk))
    tris = []
    for i in range(spec.nx - 1):
        for j in range(spec.ny - 1):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            tris += [(a, b, c), (a, c, d)]
    return pts, edges, np.asarray(tris)


def build_prop_trimesh(props: list[dict]):
    """Static collider mesh from prop dicts (protocol scene.props, soft profile).

    Supported kinds: static_box {half_size,pos,quat?}, cylinder
    {radius, half_length, pos, quat?} (axis = local +z), mesh
    {collision_path, scale?, pos, quat?}. Returns one merged trimesh or None.
    """
    import trimesh

    from .transforms import quat_to_mat
    parts = []
    for p in props or []:
        kind = p.get("kind", "mesh" if p.get("collision_path") else None)
        if kind == "static_box":
            m = trimesh.creation.box(extents=2 * np.asarray(p["half_size"], float))
        elif kind == "cylinder":
            m = trimesh.creation.cylinder(radius=float(p["radius"]),
                                          height=2 * float(p["half_length"]))
        elif kind == "mesh":
            m = trimesh.load(p["collision_path"], force="mesh")
            m.apply_scale(float(p.get("scale", 1.0)))
        else:
            raise ValueError(f"prop kind not supported by the soft profile: {kind!r}")
        T = np.eye(4)
        T[:3, :3] = quat_to_mat(np.asarray(p.get("quat", (1, 0, 0, 0)), float))
        T[:3, 3] = np.asarray(p["pos"], float)
        m.apply_transform(T)
        parts.append(m)
    return trimesh.util.concatenate(parts) if parts else None


class SoftClothSim:
    """Cloth + kinematic panda gripper world, stepped at the control rate."""

    PAD_HALF = (0.011, 0.004, 0.023)
    PALM_HALF = (0.03, 0.045, 0.012)

    def __init__(self, spec: ClothSpec, table_z: float = 0.0,
                 urdf_path: str | None = None, base_pose: np.ndarray | None = None,
                 props: list[dict] | None = None,
                 device: str = "cuda:0"):
        wp.init()
        self.spec, self.table_z, self.device = spec, float(table_z), device
        self.panda = KinematicPanda(urdf_path, base_pose)
        pts, edges, self.tris = _grid(spec)
        self.n = len(pts)
        # CSR adjacency (both directions), deterministic order
        adj = [[] for _ in range(self.n)]
        for a, b, k in edges:
            L = float(np.linalg.norm(pts[a] - pts[b]))
            adj[a].append((b, L, k))
            adj[b].append((a, L, k))
        offs, nbr, rest, stf = [0], [], [], []
        for lst in adj:
            for j, L, k in sorted(lst):
                nbr.append(j); rest.append(L); stf.append(k)
            offs.append(len(nbr))
        dev = device
        self.p = wp.array(pts.astype(np.float32), dtype=wp.vec3, device=dev)
        self.p_tmp = wp.zeros(self.n, dtype=wp.vec3, device=dev)
        self.p_prev = wp.zeros(self.n, dtype=wp.vec3, device=dev)
        self.v = wp.zeros(self.n, dtype=wp.vec3, device=dev)
        self.offs = wp.array(np.asarray(offs, np.int32), dtype=int, device=dev)
        self.nbr = wp.array(np.asarray(nbr, np.int32), dtype=int, device=dev)
        self.rest = wp.array(np.asarray(rest, np.float32), dtype=float, device=dev)
        self.stf = wp.array(np.asarray(stf, np.float32), dtype=float, device=dev)
        self.attach_flag = wp.zeros(self.n, dtype=int, device=dev)
        self.attach_local = wp.zeros(self.n, dtype=wp.vec3, device=dev)
        self._attached = np.zeros(self.n, bool)
        self.boxes = wp.zeros(3, dtype=wp.transformf, device=dev)
        self.box_half = wp.zeros(3, dtype=wp.vec3, device=dev)
        self.hand_tf = wp.zeros(1, dtype=wp.transformf, device=dev)
        # static prop colliders as one BVH mesh
        self.prop_trimesh = build_prop_trimesh(props)
        if self.prop_trimesh is not None:
            self.prop_mesh = wp.Mesh(
                points=wp.array(np.asarray(self.prop_trimesh.vertices, np.float32),
                                dtype=wp.vec3, device=dev),
                indices=wp.array(np.asarray(self.prop_trimesh.faces, np.int32).flatten(),
                                 dtype=int, device=dev))
        else:
            self.prop_mesh = None
        self.use_selfcol = bool(spec.self_collision)
        self.d_min = spec.self_dist * spec.spacing
        # full-arm collision: one static BVH per link (collision STLs, link
        # frame), particles queried in link-local coords — no per-step rebuild
        from .soft_render import PandaModel
        self.arm_model = PandaModel(urdf_path, base_pose, collision=True)
        self.arm_links = []          # link names in fixed order
        arm_meshes = []
        for name, entries in self.arm_model.link_meshes.items():
            import trimesh as _tm
            parts = []
            for m, T0 in entries:
                mm = m.copy()
                mm.apply_transform(T0)
                parts.append(mm)
            merged = _tm.util.concatenate(parts)
            arm_meshes.append(wp.Mesh(
                points=wp.array(np.asarray(merged.vertices, np.float32), dtype=wp.vec3, device=dev),
                indices=wp.array(np.asarray(merged.faces, np.int32).flatten(), dtype=int, device=dev)))
            self.arm_links.append(name)
        self._arm_meshes = arm_meshes            # keep refs alive
        self.arm_mesh_ids = wp.array(np.array([m.id for m in arm_meshes], dtype=np.uint64),
                                     dtype=wp.uint64, device=dev)
        self.arm_tf = wp.zeros(len(arm_meshes), dtype=wp.transformf, device=dev)

    # ------------------------------------------------------------ helpers
    def _tf(self, pose7: np.ndarray) -> wp.transformf:
        p, q = pose7[:3], pose7[3:]           # wxyz -> warp xyzw
        return wp.transformf(wp.vec3(*map(float, p)),
                             wp.quatf(float(q[1]), float(q[2]), float(q[3]), float(q[0])))

    def _local(self, tcp: np.ndarray, pts: np.ndarray) -> np.ndarray:
        from .transforms import pose_inv, quat_to_mat
        inv = pose_inv(tcp)
        return pts @ quat_to_mat(inv[3:]).T + inv[:3]

    def step(self, arm_q: np.ndarray, grip_cmd_norm: float, dt: float = 0.05):
        """Advance one control step under the commanded joint targets."""
        from .transforms import quat_to_mat
        spec = self.spec
        tcp = self.panda.fk(arm_q)
        w = max(0.0, grip_norm_to_width(float(grip_cmd_norm)))
        R = quat_to_mat(tcp[3:])
        y_axis, z_off = R[:, 1], R[:, 2]
        pad = np.asarray(self.PAD_HALF)
        boxes = np.zeros((3, 7)); boxes[:, 3] = 1.0
        for s, row in ((1.0, 0), (-1.0, 1)):
            c = tcp[:3] + y_axis * s * (w / 2 + pad[1]) + z_off * (-0.020)
            boxes[row] = np.concatenate([c, tcp[3:]])
        boxes[2] = np.concatenate([tcp[:3] + z_off * (-0.055), tcp[3:]])
        self.boxes = wp.array([self._tf(b) for b in boxes], dtype=wp.transformf, device=self.device)
        self.box_half = wp.array([wp.vec3(*self.PAD_HALF), wp.vec3(*self.PAD_HALF),
                                  wp.vec3(*self.PALM_HALF)], dtype=wp.vec3, device=self.device)
        self.hand_tf = wp.array([self._tf(tcp)], dtype=wp.transformf, device=self.device)
        # pose the arm link BVHs for this control step
        from .transforms import mat_to_pose
        link_T = self.arm_model.link_world_T(arm_q, w)
        self.arm_tf = wp.array([self._tf(mat_to_pose(link_T[n])) for n in self.arm_links],
                               dtype=wp.transformf, device=self.device)

        # pinch attach / release (CPU, deterministic)
        if w < spec.attach_width and not self._attached.any():
            pts = self.p.numpy()
            lp = self._local(tcp, pts)
            m = (np.abs(lp[:, 0]) < 0.012) & (np.abs(lp[:, 1]) < w / 2 + 0.006) \
                & (lp[:, 2] > -0.045) & (lp[:, 2] < 0.006)
            if m.any():
                self._attached = m
                self.attach_flag = wp.array(m.astype(np.int32), dtype=int, device=self.device)
                self.attach_local = wp.array(lp.astype(np.float32), dtype=wp.vec3, device=self.device)
        elif w > spec.release_width and self._attached.any():
            self._attached[:] = False
            self.attach_flag.zero_()

        sub_dt = dt / spec.substeps
        for _ in range(spec.substeps):
            wp.launch(k_predict, dim=self.n,
                      inputs=[self.p, self.v, self.p_prev, sub_dt, spec.damping])
            for _ in range(spec.iters):
                wp.launch(k_jacobi, dim=self.n,
                          inputs=[self.p, self.p_tmp, self.offs, self.nbr,
                                  self.rest, self.stf, 1.2])
                self.p, self.p_tmp = self.p_tmp, self.p
            if self.use_selfcol:
                wp.launch(k_selfcol, dim=self.n,
                          inputs=[self.p, self.p_prev, self.p_tmp,
                                  self.offs, self.nbr, self.d_min,
                                  float(spec.self_friction)])
                self.p, self.p_tmp = self.p_tmp, self.p
            if self.prop_mesh is not None:
                wp.launch(k_mesh_collide, dim=self.n,
                          inputs=[self.prop_mesh.id, self.p, self.p_prev,
                                  spec.thickness, self.attach_flag])
            wp.launch(k_arm_collide, dim=self.n,
                      inputs=[self.arm_mesh_ids, self.arm_tf, len(self.arm_links),
                              self.p, spec.thickness, self.attach_flag])
            wp.launch(k_project, dim=self.n,
                      inputs=[self.p, self.p_prev, self.table_z, spec.thickness,
                              self.boxes, self.box_half, 3,
                              self.attach_flag, self.attach_local, self.hand_tf,
                              float(spec.table_stick_speed) * sub_dt])
            wp.launch(k_velocity, dim=self.n,
                      inputs=[self.p, self.p_prev, self.v, 1.0 / sub_dt])

    def particles(self) -> np.ndarray:
        return self.p.numpy().astype(np.float64)


def run_soft_actions(spec: ClothSpec, actions: np.ndarray, table_z: float = 0.0,
                     urdf_path: str | None = None, base_pose: np.ndarray | None = None,
                     props: list[dict] | None = None,
                     dt: float = 0.05, device: str = "cuda:0") -> np.ndarray:
    """Replay a (T, 8) action stream; returns particle trajectory (T, N, 3)."""
    sim = SoftClothSim(spec, table_z, urdf_path, base_pose, props, device)
    out = np.empty((len(actions), sim.n, 3))
    for k, a in enumerate(np.asarray(actions, dtype=np.float64)):
        sim.step(a[:7], a[7], dt)
        out[k] = sim.particles()
    return out


def _gripper_boxes(tcp: np.ndarray, width: float):
    """Corner sets of the two finger pads + palm (matches SoftClothSim.step)."""
    from .transforms import quat_to_mat
    R = quat_to_mat(tcp[3:])
    y_axis, z_off = R[:, 1], R[:, 2]
    pad = np.asarray(SoftClothSim.PAD_HALF)
    boxes = []
    for s in (1.0, -1.0):
        c = tcp[:3] + y_axis * s * (width / 2 + pad[1]) + z_off * (-0.020)
        boxes.append((c, SoftClothSim.PAD_HALF))
    boxes.append((tcp[:3] + z_off * (-0.055), SoftClothSim.PALM_HALF))
    out = []
    for c, half in boxes:
        h = np.asarray(half)
        corners = np.array([[sx * h[0], sy * h[1], sz * h[2]]
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        out.append(corners @ R.T + c)
    return out


def render_cloth_mp4(traj: np.ndarray, tris: np.ndarray, path: str | Path,
                     table_z: float = 0.0, fps: int = 20, stride: int = 2,
                     prop_trimesh=None, actions: np.ndarray | None = None,
                     panda: "KinematicPanda | None" = None):
    """Headless matplotlib render — no Vulkan involved. When ``actions`` and
    ``panda`` are given, draws the arm skeleton and gripper boxes too."""
    import matplotlib
    matplotlib.use("Agg")
    import imageio.v2 as iio
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    FACES = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
             [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    lo = traj.reshape(-1, 3).min(0) - 0.05
    hi = traj.reshape(-1, 3).max(0) + 0.05
    frames = []
    for k in range(0, len(traj), stride):
        fig = plt.figure(figsize=(5, 4), dpi=90)
        ax = fig.add_subplot(projection="3d")
        if prop_trimesh is not None:
            V, F = prop_trimesh.vertices, prop_trimesh.faces
            ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F,
                            color="#b0a08a", edgecolor="none", alpha=0.9, shade=True)
        P = traj[k]
        ax.plot_trisurf(P[:, 0], P[:, 1], P[:, 2], triangles=tris,
                        color="#4d7fbe", edgecolor="none", alpha=0.95, shade=True)
        if actions is not None and panda is not None:
            q7 = actions[k, :7]
            sk = panda.link_polyline(q7)
            sk = sk[np.linalg.norm(np.diff(sk, axis=0, prepend=sk[:1]), axis=1) < 0.6]
            ax.plot(sk[:, 0], sk[:, 1], sk[:, 2], "-", color="#444444",
                    lw=3.5, solid_capstyle="round", alpha=0.9)
            w = max(0.0, grip_norm_to_width(float(actions[k, 7])))
            for corners in _gripper_boxes(panda.fk(q7), w):
                polys = [[corners[i] for i in f] for f in FACES]
                ax.add_collection3d(Poly3DCollection(
                    polys, facecolor="#2a2a2a", edgecolor="#111111", alpha=0.9))
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(table_z, max(hi[2], table_z + 0.25))
        ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1],
                           max(hi[2], table_z + 0.25) - table_z))
        ax.view_init(elev=28, azim=-70)
        ax.set_title(f"t = {k * 0.05:.2f}s")
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(img.copy())
        plt.close(fig)
    iio.mimwrite(path, frames, fps=max(1, fps // stride))
