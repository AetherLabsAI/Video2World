#!/usr/bin/env python3
"""Minimal Panda profile for the video2sim evaluator: `V2SRobotScene-v1`.

Same scene-spec contract as `V2SBridge-v1` (support / props / objects built from the package), but the robot is the ManiSkill Panda with its
two-finger gripper at the task's base pose, controlled in `pd_joint_pos` — so a hidden ManiSkill Panda trajectory can be replayed as-is
(no WidowX conversion). Everything else (entity building, settling, pose/grasp/contact accessors) is inherited from V2SBridgeEnv.
"""
from __future__ import annotations
import numpy as np, sapien, torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from video2sim.bench.bridge_env import V2SBridgeEnv
from . import panda_robotiq   # noqa: registers the `panda_robotiq` agent (DROID: Panda arm + Robotiq 2F-85)
from . import xarm7_pusher    # noqa: registers the `xarm7_pusher` agent (reconstructed-twins Push-T: xArm7 + 20 cm pusher rod, no gripper)

MS3_TABLE_ARENA = 'ms3_table_scene'   # profile-provided static scene: the task's own TableSceneBuilder (table asset + floor/background)

PANDA_REST_QPOS = np.array([0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, np.pi * 3 / 4, np.pi / 4, 0.04, 0.04], dtype=np.float32)
PANDA_BASE = (-0.615, 0.0, 0.0)          # ManiSkill tabletop tasks put the Panda base here (world == our base frame)
MAX_DEPEN_VEL = 10.0   # m/s, max depenetration velocity of DROID-rig (panda_robotiq) scene objects
SIM_FREQ, CONTROL_FREQ = 100, 20          # ManiSkill defaults, matching the demo's recording rate
REST_QPOS = {'panda': PANDA_REST_QPOS, 'panda_robotiq': PANDA_REST_QPOS, 'xarm7_pusher': xarm7_pusher.REST_QPOS}   # per-robot default reset qpos (hidden trajectories override it)
HAS_GRIPPER = {'panda': True, 'panda_robotiq': True, 'xarm7_pusher': False}


@register_env("V2SRobotScene-v1", max_episode_steps=1_000_000)
class V2SPandaEnv(V2SBridgeEnv):
    SUPPORTED_OBS_MODES = ["state", "rgb", "rgb+segmentation", "none"]
    SUPPORTED_REWARD_MODES = ["none"]

    def __init__(self, *args, scene_spec: dict, camera: dict | None = None, enable_cameras: bool = True,
                 robot_base_pose=None, robot_uid='panda', **kwargs):
        self.scene_spec = scene_spec
        self.camera_spec = camera
        self.enable_cameras = enable_cameras
        self.robot_base_pose = robot_base_pose or list(PANDA_BASE) + [1.0, 0.0, 0.0, 0.0]
        self.robot_uid = robot_uid
        from video2sim.bench.bridge_env import library_info
        self._info = library_info()
        self.objs, self.props, self.arena, self.settled_poses = {}, {}, None, {}
        kwargs.setdefault("control_mode", "pd_joint_pos")
        BaseEnv.__init__(self, *args, robot_uids=robot_uid, **kwargs)   # skip V2SBridgeEnv.__init__ (WidowX)

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=SIM_FREQ, control_freq=CONTROL_FREQ, spacing=20)

    def _load_lighting(self, options: dict):
        BaseEnv._load_lighting(self, options)          # ManiSkill tabletop default lighting (PickCube does not override it)

    def _load_scene(self, options: dict):
        sp = self.scene_spec
        if (sp.get('arena') or {}).get('library') == MS3_TABLE_ARENA:
            self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0.0); self.table_scene.build()
            for p_ in sp.get('props') or []:
                self.props[p_['name']] = self._build_entity(p_, static=True)
            for o in sp.get('objects') or []:
                self.objs[o['name']] = self._build_entity(o, static=False)
            if sp.get('support'):                      # a candidate may also declare its own support slab; build it as a static box on top of the table
                s_ = sp['support']; cx, cy = s_.get('center', (0.0, 0.0)); lx, ly = s_.get('size', (1.0, 1.0)); th = float(s_.get('thickness', 0.05))
                self.props['support'] = self._build_entity(dict(name='support', kind='box', half_size=[lx / 2, ly / 2, th / 2],
                                                                pos=[cx, cy, float(s_['z']) - th / 2], color=s_.get('color', (0.55, 0.35, 0.2, 1.0))), static=True)
        else:
            self.table_scene = None
            V2SBridgeEnv._load_scene(self, options)
        if self.robot_uid == 'panda_robotiq':
            # DROID rig only (FurnitureBench physics unchanged): cap how fast PhysX may push interpenetrating bodies apart. The
            # default is unlimited, so a held part pressed by the stiff PD arm into another part's thin-slab collision edge
            # (sample 7 cup on a bowl rim, sample 8 tape on a plate rim) was thrown metres away in one frame; 1 m/s is far above
            # any real pick-and-place speed; the value is a trade-off measured in METRIC_AUDIT 18.18 (10 m/s chosen by the user)
            import sapien as _sp
            for a_ in self.objs.values():
                for ent in a_._objs:
                    comp = ent.find_component_by_type(_sp.physx.PhysxRigidDynamicComponent)
                    if comp is not None: comp.set_max_depenetration_velocity(MAX_DEPEN_VEL)

    STATIC_NONCONVEX = True   # static props keep their concave geometry

    def _build_entity(self, e: dict, static: bool):
        """Static mesh props with their real (concave) geometry (STATIC_NONCONVEX).

        The shared builder gives every mesh a convex decomposition. That closes the holes of the FurnitureBench receivers:
        at the official assembled pose 12-24 % of a leg's surface points lie inside the desk / table top's collision, so a leg
        can never enter its hole and stands on the plate instead (2026-09-17 diagnosis). PhysX allows a triangle mesh for a
        STATIC actor, which keeps the holes; dynamic parts keep the convex decomposition.
        """
        if not (static and self.STATIC_NONCONVEX and e.get('kind') == 'mesh'): return V2SBridgeEnv._build_entity(self, e, static)
        import sapien as _sp
        b = self.scene.create_actor_builder(); s_ = [float(e.get('scale', 1.0))] * 3
        mat = _sp.physx.PhysxMaterial(static_friction=float(e.get('static_friction', 1.0)), dynamic_friction=float(e.get('dynamic_friction', 1.0)), restitution=0.0)
        b.add_nonconvex_collision_from_file(filename=e.get('mesh_path') or e['collision_path'], scale=s_, material=mat)
        if self.enable_cameras: b.add_visual_from_file(filename=e.get('mesh_path') or e['collision_path'], scale=s_)
        b.initial_pose = sapien.Pose(p=[float(v) for v in e['pos']], q=[float(v) for v in e.get('quat', (1, 0, 0, 0))])
        return b.build_static(name=e['name'])

    def _load_agent(self, options: dict):
        bp = self.robot_base_pose
        BaseEnv._load_agent(self, options, sapien.Pose(p=bp[:3], q=bp[3:7]))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            if getattr(self, 'table_scene', None) is not None:
                self.table_scene.initialize(env_idx)   # table pose + nominal robot pose/qpos (overridden below by the hidden qpos)
            for o in self.scene_spec.get("objects") or []:
                a = self.objs[o["name"]]
                a.set_pose(Pose.create_from_pq(p=torch.tensor([[float(v) for v in o["pos"]]]),
                                               q=torch.tensor([[float(v) for v in o.get("quat", (1, 0, 0, 0))]])))
                a.set_linear_velocity(torch.zeros(1, 3)); a.set_angular_velocity(torch.zeros(1, 3))
            qpos = np.asarray(options.get("qpos", REST_QPOS.get(self.robot_uid, PANDA_REST_QPOS)), dtype=np.float32)
            qpos = self.robot_qpos(qpos)
            self.agent.reset(init_qpos=qpos)
            for j, v in zip(self.agent.robot.active_joints, qpos):
                j.set_drive_target(torch.tensor([float(v)]))
            self._settle(float(options.get("settle", 0.5)))
            self.settled_poses = {n: self.obj_pose_np(n) for n in self.objs}

    # --- Panda state helpers (used by the replay) ---
    def robot_qpos(self, q):
        """Hidden trajectories store the Franka layout (7 arm + 2 finger joints, finger = half the opening). For the
        `panda_robotiq` agent the 6 Robotiq joints are derived from the opening width."""
        q = np.asarray(q, dtype=np.float32)
        if self.robot_uid == 'xarm7_pusher': return q[:7]          # no gripper joints: 7 arm joints only
        if self.robot_uid == 'panda_robotiq' and len(q) == 9:
            th = self.agent.theta_from_width(2.0 * float(q[7])); names = [j.name for j in self.agent.robot.active_joints][7:]; return np.r_[q[:7], panda_robotiq.gripper_qpos(th, names)].astype(np.float32)
        return q

    def gripper_cmd(self, width_m):
        """Gripper action for a recorded opening width: Robotiq -> outer-knuckle angle (rad); Franka -> normalised finger target."""
        if self.robot_uid == 'panda_robotiq': return float(self.agent.theta_from_width(width_m))
        if self.robot_uid == 'xarm7_pusher': return 0.0            # no gripper: the action vector has no gripper column
        return float(np.clip((np.clip(width_m / 2, 0, 0.04) + 0.01) / 0.05 * 2 - 1, -1, 1))

    def qpos_np(self) -> np.ndarray:
        return self.agent.robot.get_qpos()[0].cpu().numpy().astype(np.float64)

    def tcp_pose_np(self) -> np.ndarray:
        p = self.agent.tcp.pose
        return np.r_[p.p[0].cpu().numpy(), p.q[0].cpu().numpy()].astype(np.float64)
