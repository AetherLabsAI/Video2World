"""Real rasterized renderer for the soft_warp profile (pyrender / EGL).

Replaces the matplotlib schematic: full panda visual meshes posed by FK,
cloth as a shaded double-sided mesh, props, a support surface, and — when the
package carries a camera block — the protocol camera (intrinsics +
T_base_cam, OpenCV convention), so soft renders are directly comparable with
the source video, like the rigid profile's SAPIEN renders.

EGL offscreen only (OpenGL) — never touches Vulkan, safe on the shared box.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _rpy_to_mat(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def _origin_T(el) -> np.ndarray:
    T = np.eye(4)
    if el is not None:
        xyz = [float(v) for v in (el.get("xyz") or "0 0 0").split()]
        rpy = [float(v) for v in (el.get("rpy") or "0 0 0").split()]
        T[:3, :3] = _rpy_to_mat(rpy)
        T[:3, 3] = xyz
    return T


class PandaModel:
    """URDF link meshes (visual or collision) + poses for every link,
    including the width-posed fingers that sit off the serial FK chain."""

    def __init__(self, urdf_path: str | None = None, base_pose: np.ndarray | None = None,
                 collision: bool = False):
        import trimesh

        from .soft import KinematicPanda
        if urdf_path is None:
            from mani_skill.agents.robots.panda.panda import Panda
            urdf_path = Panda.urdf_path
        self.urdf_dir = Path(urdf_path).parent
        self.kin = KinematicPanda(urdf_path, base_pose)
        root = ET.parse(urdf_path).getroot()

        tag = "collision" if collision else "visual"
        self.link_meshes: dict[str, list] = {}
        for link in root.findall("link"):
            entries = []
            for vis in link.findall(tag):
                geo = vis.find("geometry/mesh")
                if geo is None:
                    continue
                m = trimesh.load(self.urdf_dir / geo.get("filename"), force="mesh")
                sc = geo.get("scale")
                if sc:
                    m.apply_scale([float(v) for v in sc.split()][0])
                entries.append((m, _origin_T(vis.find("origin"))))
            if entries:
                self.link_meshes[link.get("name")] = entries

        self.finger_joints = []
        for j in root.findall("joint"):
            if j.get("name") in ("panda_finger_joint1", "panda_finger_joint2"):
                child = j.find("child").get("link")
                axis_el = j.find("axis")
                axis = np.array([float(v) for v in (axis_el.get("xyz") if axis_el is not None
                                                    else "0 1 0").split()])
                self.finger_joints.append((child, _origin_T(j.find("origin")), axis))

    # ------------------------------------------------------------------ poses
    def link_world_T(self, q7: np.ndarray, width: float) -> dict[str, np.ndarray]:
        """4x4 world transform per link that has meshes."""
        import torch

        from .transforms import quat_to_mat
        frames = self.kin.chain.forward_kinematics(
            torch.as_tensor(q7, dtype=torch.float64, device="cpu")[None], end_only=False)
        Tb = np.eye(4)
        Tb[:3, :3] = quat_to_mat(self.kin.base_pose[3:])
        Tb[:3, 3] = self.kin.base_pose[:3]
        out = {}
        for name in self.link_meshes:
            if name in frames:
                out[name] = Tb @ frames[name].get_matrix()[0].numpy()
        hand = out.get("panda_hand")
        if hand is not None:
            for child, T0, axis in self.finger_joints:
                if child in self.link_meshes:
                    Tq = np.eye(4)
                    Tq[:3, 3] = axis * max(0.0, width / 2)
                    out[child] = hand @ T0 @ Tq
        return out


def render_soft_mp4(traj: np.ndarray, tris: np.ndarray, path: str | Path,
                    actions: np.ndarray, urdf_path: str | None = None,
                    base_pose: np.ndarray | None = None, table_z: float = 0.0,
                    prop_trimesh=None, camera: dict | None = None,
                    width: int = 640, height: int = 480,
                    fps: int = 20, stride: int = 2):
    """Rasterized soft-episode render. camera = protocol camera block
    (intrinsics + extrinsics_base_cam, OpenCV) or None for a default orbit."""
    import imageio.v2 as iio
    import pyrender
    import trimesh

    from .soft import grip_norm_to_width
    from .transforms import quat_to_mat

    model = PandaModel(urdf_path, base_pose)
    scene = pyrender.Scene(ambient_light=[0.35] * 3, bg_color=[0.92, 0.92, 0.95])

    ext = float(np.abs(traj.reshape(-1, 3)[:, :2]).max() + 0.6)
    table = trimesh.creation.box(extents=[2 * ext, 2 * ext, 0.02])
    table.visual.face_colors = [205, 175, 135, 255]
    tn = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(table, smooth=False), matrix=np.eye(4))
    scene.add_node(tn)
    Tt = np.eye(4)
    Tt[2, 3] = table_z - 0.01
    scene.set_pose(tn, Tt)

    if prop_trimesh is not None:
        pm = prop_trimesh.copy()
        pm.visual.face_colors = [176, 160, 138, 255]
        scene.add(pyrender.Mesh.from_trimesh(pm, smooth=False))

    link_nodes = {}
    for name, entries in model.link_meshes.items():
        nodes = []
        for m, T0 in entries:
            node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(m, smooth=True),
                                 matrix=np.eye(4))
            scene.add_node(node)
            nodes.append((node, T0))
        link_nodes[name] = nodes

    cloth_mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.24, 0.42, 0.72, 1.0], metallicFactor=0.0,
        roughnessFactor=0.85, doubleSided=True)

    # camera ------------------------------------------------------------
    if camera:
        K = np.asarray(camera["intrinsics"], dtype=np.float64)
        width, height = int(camera["width"]), int(camera["height"])
        cam = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
                                        znear=0.01, zfar=10.0)
        bp = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float) if base_pose is None \
            else np.asarray(base_pose, dtype=float)
        Tb = np.eye(4)
        Tb[:3, :3] = quat_to_mat(bp[3:])
        Tb[:3, 3] = bp[:3]
        T_world_cv = Tb @ np.asarray(camera["extrinsics_base_cam"], dtype=np.float64)
        T_gl = T_world_cv @ np.diag([1.0, -1.0, -1.0, 1.0])   # OpenCV -> OpenGL
    else:
        cam = pyrender.PerspectiveCamera(yfov=1.0, znear=0.01, zfar=10.0)
        center = traj.reshape(-1, 3).mean(0) + np.array([0.0, 0.0, 0.12])
        eye = center + np.array([0.7, 0.7, 0.75])
        z = (eye - center) / np.linalg.norm(eye - center)
        x = np.cross([0, 0, 1.0], z)
        x /= np.linalg.norm(x)
        T_gl = np.eye(4)
        T_gl[:3, :3] = np.stack([x, np.cross(z, x), z], 1)
        T_gl[:3, 3] = eye
    scene.add(cam, pose=T_gl)
    scene.add(pyrender.DirectionalLight(intensity=4.0), pose=T_gl)
    key = np.eye(4)
    key[:3, 3] = [0.3, -0.3, 1.2]
    scene.add(pyrender.DirectionalLight(intensity=2.5), pose=key)

    r = pyrender.OffscreenRenderer(width, height)
    frames = []
    cloth_node = None
    for k in range(0, len(traj), stride):
        w = max(0.0, grip_norm_to_width(float(actions[k, 7])))
        for name, T in model.link_world_T(actions[k, :7], w).items():
            for node, T0 in link_nodes[name]:
                scene.set_pose(node, T @ T0)
        cm = trimesh.Trimesh(vertices=traj[k], faces=tris, process=False)
        cm.fix_normals()
        if cloth_node is not None:
            scene.remove_node(cloth_node)
        cloth_node = pyrender.Node(
            mesh=pyrender.Mesh.from_trimesh(cm, material=cloth_mat, smooth=True))
        scene.add_node(cloth_node)
        color, _ = r.render(scene)
        frames.append(color.copy())
    r.delete()
    iio.mimwrite(path, frames, fps=max(1, fps // stride))
