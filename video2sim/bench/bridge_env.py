"""Generic WidowX (SIMPLER rig) scene built from a manifest — the unified
embodiment / control specification of the Bridge real2sim benchmark.

Every simulator that enters the benchmark — the human-built one and every
agent-generated one — is instantiated through this class from a *scene spec*
dict, so they differ only in what the spec says (geometry, poses, materials),
never in robot, controller, physics settings or camera model.

Scene spec (all poses in the ROBOT BASE frame; world == base):

    {
      "support":   {"z": -0.002, "center": [0.3, 0.0], "size": [1.0, 1.0],
                    "thickness": 0.05, "color": [r, g, b, 1]} | null,
      "arena":     {"library": "bridge_table_1_v2"} | null,  # SIMPLER stage glb
      "props":     [ {STATIC ...}, ... ],
      "objects":   [ {DYNAMIC ...}, ... ]
    }

    entity (prop or object):
      {"name": "eggplant", "kind": "box|cylinder|mesh|library",
       "half_size": [..],                # box / cylinder(radius, -, half_h)
       "mesh_path", "collision_path", "scale",   # mesh (paths already resolved)
       "library_id": "eggplant",         # SIMPLER bridge model library
       "pos": [x, y, z], "quat": [w, x, y, z],
       "density": 500, "color": [r, g, b, a],
       "static_friction": 0.5, "dynamic_friction": 0.5}

Robot: `WidowX250SBridgeDatasetFlatTable` (real2sim-tuned gains, the same
controller SIMPLER uses: `arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos`).
The camera is whatever the evaluator mounts (OpenCV extrinsics in the base
frame + intrinsics); it is NOT part of the spec an agent competes on.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sapien
import torch
from sapien.physx import PhysxMaterial

from mani_skill import ASSET_DIR
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval.base_env import (
    WidowX250SBridgeDatasetFlatTable,
)
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig

LIBRARY_ROOT = ASSET_DIR / "tasks/bridge_v2_real2sim_dataset"
LIBRARY_INFO = LIBRARY_ROOT / "custom/info_bridge_custom_v0.json"

# SIMPLER's rest configuration ("measured values for bridge dataset")
REST_QPOS = np.array([-0.01840777, 0.0398835, 0.22242722, -0.00460194,
                      1.36524296, 0.00153398, 0.037, 0.037])
SIM_FREQ, CONTROL_FREQ = 500, 5

# SAPIEN camera frame (+X fwd, +Y left, +Z up) expressed in OpenCV (+Z fwd,
# +X right, +Y down): columns are the SAPIEN axes.
CV_TO_SAPIEN = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])

# SIMPLER's own stages, expressed in the base frame of the robot as SIMPLER
# places it. Recovered from base_env.py: arena pose = Pose(-scene_offset,
# q=[0.707,0.707,0,0]) in a world where the base sits at BASE_WORLD.
_SIMPLER_ARENA = {
    "bridge_table_1_v1": {"base_world": [0.147, 0.028, 0.870]},
    "bridge_table_1_v2": {"base_world": [0.127, 0.060, 0.850]},
}
_SCENE_OFFSET = np.array([-2.0634, -2.8313, 0.0])
_SCENE_QUAT = [0.707, 0.707, 0, 0]


def library_info() -> dict:
    return json.loads(LIBRARY_INFO.read_text())


def library_model_dir(model_id: str) -> Path:
    d = LIBRARY_ROOT / "custom/models" / model_id
    if not d.exists():
        raise FileNotFoundError(f"library model {model_id!r} not found under {d.parent}")
    return d


def _visual_file(d: Path) -> str:
    for n in ("textured.obj", "textured.dae", "textured.glb"):
        if (d / n).exists():
            return str(d / n)
    raise FileNotFoundError(f"no visual mesh in {d}")


def simpler_arena_pose_in_base(library: str) -> sapien.Pose:
    """Pose of a SIMPLER stage glb in the robot base frame (base at origin,
    identity orientation — SIMPLER's base quaternion [0,0,0,1] is a 180° yaw
    which we absorb into the arena pose instead)."""
    base_world = np.asarray(_SIMPLER_ARENA[library]["base_world"])
    # SIMPLER: base q = (w=0,x=0,y=0,z=1) → Rz(180°)
    T_wb = np.eye(4)
    T_wb[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    T_wb[:3, 3] = base_world
    arena_world = sapien.Pose(p=(-_SCENE_OFFSET).tolist())  # builder.initial_pose
    inner = sapien.Pose(q=_SCENE_QUAT)                       # add_visual pose
    T_wa = arena_world.to_transformation_matrix() @ inner.to_transformation_matrix()
    T_ba = np.linalg.inv(T_wb) @ T_wa
    return sapien.Pose(T_ba)


def simpler_world_to_base(library: str, p_world) -> np.ndarray:
    """Convert a point given in SIMPLER's world (as in its task files) to our
    base frame — handy for porting their object placements."""
    base_world = np.asarray(_SIMPLER_ARENA[library]["base_world"])
    d = np.asarray(p_world, dtype=np.float64) - base_world
    return np.array([-d[0], -d[1], d[2]])


def camera_config_from_cv(cam: dict, name: str = "eval_camera") -> CameraConfig:
    """OpenCV `extrinsics_base_cam` (T_base_cam) + K → SAPIEN CameraConfig."""
    T = np.asarray(cam["extrinsics_base_cam"], dtype=np.float64)
    Ts = T.copy()
    Ts[:3, :3] = T[:3, :3] @ CV_TO_SAPIEN
    return CameraConfig(
        name, pose=sapien.Pose(Ts),
        width=int(cam["width"]), height=int(cam["height"]),
        intrinsic=np.asarray(cam["intrinsics"], dtype=np.float32),
        near=0.01, far=10.0,
    )


@register_env("V2SBridge-v1", max_episode_steps=1_000_000)
class V2SBridgeEnv(BaseEnv):
    SUPPORTED_OBS_MODES = ["state", "rgb", "rgb+segmentation", "none"]
    SUPPORTED_REWARD_MODES = ["none"]

    def __init__(self, *args, scene_spec: dict, camera: dict | None = None,
                 enable_cameras: bool = True, **kwargs):
        self.scene_spec = scene_spec
        self.camera_spec = camera
        self.enable_cameras = enable_cameras
        self._info = library_info()
        self.objs: dict[str, object] = {}
        self.props: dict[str, object] = {}
        self.arena = None
        self.settled_poses: dict[str, np.ndarray] = {}
        super().__init__(*args, robot_uids=WidowX250SBridgeDatasetFlatTable, **kwargs)

    # ----------------------------------------------------------------- config
    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=SIM_FREQ, control_freq=CONTROL_FREQ, spacing=20)

    @property
    def _default_sensor_configs(self):
        if self.camera_spec and self.enable_cameras:
            return [camera_config_from_cv(self.camera_spec)]
        return []

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig("render_camera", pose=sapien.Pose([0.0, -0.16, 0.336],
                            [0.909182, -0.0819809, 0.347277, 0.214629]),
                            width=512, height=512, fov=1.0, near=0.01, far=100)

    def _load_lighting(self, options: dict):
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light([0, 0, -1], [2.2, 2.2, 2.2], shadow=False)
        self.scene.add_directional_light([-1, -0.5, -1], [0.7, 0.7, 0.7])
        self.scene.add_directional_light([1, 1, -1], [0.7, 0.7, 0.7])

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose())          # world == base

    # ------------------------------------------------------------------ build
    def _material(self, e: dict) -> PhysxMaterial:
        return PhysxMaterial(static_friction=float(e.get("static_friction", 0.5)),
                             dynamic_friction=float(e.get("dynamic_friction", 0.5)),
                             restitution=0.0)

    def _add_geometry(self, b, e: dict, visual: bool):
        kind = e["kind"]
        mat = self._material(e)
        color = tuple(e.get("color", (0.7, 0.7, 0.7, 1.0)))
        density = float(e.get("density", 500.0))
        if kind == "box":
            hs = [float(v) for v in e["half_size"]]
            b.add_box_collision(half_size=hs, material=mat, density=density)
            if visual:
                b.add_box_visual(half_size=hs, material=sapien.render.RenderMaterial(base_color=color))
        elif kind == "cylinder":
            r, hh = float(e["half_size"][0]), float(e["half_size"][2])
            # SAPIEN cylinders lie along local +x; stand them up (axis = +z)
            rot = sapien.Pose(q=[0.7071068, 0.0, 0.7071068, 0.0])
            b.add_cylinder_collision(radius=r, half_length=hh, pose=rot, material=mat, density=density)
            if visual:
                b.add_cylinder_visual(radius=r, half_length=hh, pose=rot,
                                      material=sapien.render.RenderMaterial(base_color=color))
        elif kind == "sphere":
            r = float(e["half_size"][0])
            b.add_sphere_collision(radius=r, material=mat, density=density)
            if visual:
                b.add_sphere_visual(radius=r, material=sapien.render.RenderMaterial(base_color=color))
        elif kind == "container":
            # open-top vessel: half_size = (inner_lx/2, inner_ly/2, wall_height/2),
            # `wall` thickness; origin at the INNER FLOOR centre
            hx, hy, hh = (float(v) for v in e["half_size"])
            t = float(e.get("wall", 0.004))
            parts = [((0, 0, -t / 2), (hx + t, hy + t, t / 2)),
                     ((hx + t / 2, 0, hh), (t / 2, hy + t, hh)), ((-hx - t / 2, 0, hh), (t / 2, hy + t, hh)),
                     ((0, hy + t / 2, hh), (hx + t, t / 2, hh)), ((0, -hy - t / 2, hh), (hx + t, t / 2, hh))]
            for off, half in parts:
                b.add_box_collision(pose=sapien.Pose(p=list(off)), half_size=list(half), material=mat, density=density)
                if visual:
                    b.add_box_visual(pose=sapien.Pose(p=list(off)), half_size=list(half),
                                     material=sapien.render.RenderMaterial(base_color=color))
        elif kind == "mesh":
            s = [float(e.get("scale", 1.0))] * 3
            b.add_multiple_convex_collisions_from_file(filename=e["collision_path"], scale=s,
                                                       material=mat, density=density)
            if visual:
                b.add_visual_from_file(filename=e.get("mesh_path") or e["collision_path"], scale=s)
        elif kind == "library":
            d = library_model_dir(e["library_id"])
            info = self._info.get(e["library_id"], {})
            s = [float(e.get("scale", 1.0))] * 3
            b.add_multiple_convex_collisions_from_file(
                filename=str(d / "collision.obj"), scale=s, material=mat,
                density=float(e.get("density", info.get("density", 1000))))
            if visual:
                b.add_visual_from_file(filename=_visual_file(d), scale=s)
        else:
            raise ValueError(f"unknown entity kind {kind!r}")

    def _build_entity(self, e: dict, static: bool):
        b = self.scene.create_actor_builder()
        self._add_geometry(b, e, visual=self.enable_cameras)
        pose = sapien.Pose(p=[float(v) for v in e["pos"]],
                           q=[float(v) for v in e.get("quat", (1, 0, 0, 0))])
        b.initial_pose = pose
        name = e["name"]
        return b.build_static(name=name) if static else b.build(name=name)

    def _load_scene(self, options: dict):
        sp = self.scene_spec
        if sp.get("arena"):
            lib = sp["arena"]["library"]
            b = self.scene.create_actor_builder()
            f = str(LIBRARY_ROOT / "stages" / f"{lib}.glb")
            pose = simpler_arena_pose_in_base(lib)
            b.add_nonconvex_collision_from_file(f, pose=pose)
            if self.enable_cameras:
                b.add_visual_from_file(f, pose=pose)
            b.initial_pose = sapien.Pose()
            self.arena = b.build_static(name="arena")
        if sp.get("support"):
            s = sp["support"]
            cx, cy = s.get("center", (0.3, 0.0))
            lx, ly = s.get("size", (1.0, 1.0))
            t = float(s.get("thickness", 0.05))
            zt = float(s["z"])
            b = self.scene.create_actor_builder()
            hs = [lx / 2, ly / 2, t / 2]
            b.add_box_collision(half_size=hs, material=self._material(s))
            if self.enable_cameras:
                b.add_box_visual(half_size=hs, material=sapien.render.RenderMaterial(
                    base_color=tuple(s.get("color", (0.55, 0.35, 0.2, 1.0)))))
            b.initial_pose = sapien.Pose(p=[cx, cy, zt - t / 2])
            self.support = b.build_static(name="support")
        for p in sp.get("props") or []:
            self.props[p["name"]] = self._build_entity(p, static=True)
        for o in sp.get("objects") or []:
            self.objs[o["name"]] = self._build_entity(o, static=False)

    # ---------------------------------------------------------------- episode
    def _settle(self, t: float):
        for _ in range(int(self.sim_freq * t)):
            self.scene.step()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            for o in self.scene_spec.get("objects") or []:
                a = self.objs[o["name"]]
                a.set_pose(Pose.create_from_pq(
                    p=torch.tensor([[float(v) for v in o["pos"]]]),
                    q=torch.tensor([[float(v) for v in o.get("quat", (1, 0, 0, 0))]])))
                a.set_linear_velocity(torch.zeros(1, 3))
                a.set_angular_velocity(torch.zeros(1, 3))
            qpos = np.asarray(options.get("qpos", REST_QPOS), dtype=np.float32)
            self.agent.reset(init_qpos=qpos)
            for j, v in zip(self.agent.robot.active_joints, qpos):
                j.set_drive_target(torch.tensor([float(v)]))
            self._settle(float(options.get("settle", 0.5)))
            v = sum(float(torch.linalg.norm(a.linear_velocity)) for a in self.objs.values())
            if v > 1e-3:
                self._settle(3.0)
            self.settled_poses = {n: self.obj_pose_np(n) for n in self.objs}

    # ----------------------------------------------------------------- access
    def obj_pose_np(self, name: str) -> np.ndarray:
        p = self.objs[name].pose
        return np.r_[p.p[0].cpu().numpy(), p.q[0].cpu().numpy()].astype(np.float64)

    def all_obj_poses_np(self) -> dict[str, np.ndarray]:
        return {n: self.obj_pose_np(n) for n in self.objs}

    def obj_velocity_np(self, name: str) -> float:
        a = self.objs[name]
        return float(torch.linalg.norm(a.linear_velocity)) + 0.1 * float(torch.linalg.norm(a.angular_velocity))

    def contact_force(self, a_name: str, b_name: str) -> float:
        ents = {**self.objs, **self.props}
        f = self.scene.get_pairwise_contact_forces(ents[a_name], ents[b_name])
        return float(torch.linalg.norm(f))

    def is_grasping_np(self, name: str) -> bool:
        return bool(self.agent.is_grasping(self.objs[name])[0])

    def evaluate(self):
        return {}

    def _get_obs_extra(self, info: dict):
        return {}
