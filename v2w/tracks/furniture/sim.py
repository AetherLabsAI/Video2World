"""ManiSkill rig of the furniture track: the `V2SPanda-v1` environment and its robots.

Robots: the ManiSkill Panda (FurnitureBench), `panda_robotiq` = Franka Panda + Robotiq 2F-85 (DROID) and
`xarm7_pusher` = UFACTORY xArm7 with a 20 cm pusher rod (Push-T, no gripper). The environment builds the package scene
with the shared `V2SBridge-v1` entity builder and places the robot at the sample's base pose, controlled in
`pd_joint_pos`.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import sapien
import torch
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig, PDJointPosMimicControllerConfig, deepcopy_dict
from mani_skill.agents.registration import register_agent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig

from v2w import paths
from video2sim.bench.bridge_env import V2SBridgeEnv


def local_urdf(path):
    """The URDF with ManiSkill asset locations of the producing host mapped to this installation."""
    import mani_skill
    path = Path(path); text = path.read_text()
    local = re.sub(r'filename="[^"]*/site-packages/mani_skill/assets/', f'filename="{mani_skill.PACKAGE_ASSET_DIR}/', text)
    local = re.sub(r'filename="[^"]*/\.maniskill/data/', f'filename="{mani_skill.ASSET_DIR}/', local)
    if local == text:
        return path
    out = Path.home() / '.cache/v2w' / path.name
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists() or out.read_text() != local:
        out.write_text(local)
    return out


# ---------------------------------------------------------------- Franka Panda + Robotiq 2F-85 (DROID)
PANDA_ROBOTIQ_URDF = local_urdf(paths.asset('maniskill', 'panda_robotiq.urdf'))
GRIPPER_JOINTS = ['left_outer_knuckle_joint', 'left_inner_knuckle_joint', 'left_inner_finger_joint', 'right_outer_knuckle_joint', 'right_inner_knuckle_joint', 'right_inner_finger_joint']
W_OPEN = 0.085


def gripper_qpos(theta, joint_names=GRIPPER_JOINTS):
    """Robotiq joint angles for an outer-knuckle angle theta in the given joint order (inner knuckle = theta, inner finger = -theta)."""
    return np.array([-theta if 'inner_finger' in n else theta for n in joint_names], np.float32)


@register_agent(asset_download_ids=['xarm6'])
class PandaRobotiq(BaseAgent):
    """Franka Panda arm + Robotiq 2F-85. qpos: 7 arm joints + 6 gripper joints; the gripper action is the outer-knuckle
    angle in radians (0 open, 0.81 closed). Gripper closure (4-bar linkage) and grasp test follow ManiSkill's XArm6Robotiq."""
    uid = 'panda_robotiq'
    urdf_path = str(PANDA_ROBOTIQ_URDF)
    urdf_config = dict(_materials=dict(gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)),
                       link=dict(left_inner_finger_pad=dict(material='gripper', patch_radius=0.1, min_patch_radius=0.1), right_inner_finger_pad=dict(material='gripper', patch_radius=0.1, min_patch_radius=0.1)))
    keyframes = dict(rest=Keyframe(qpos=np.r_[[0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, np.pi * 3 / 4, np.pi / 4], np.zeros(6)], pose=sapien.Pose()))
    arm_joint_names = [f'panda_joint{i}' for i in range(1, 8)]
    arm_stiffness = 1e3; arm_damping = 1e2; arm_force_limit = 100
    gripper_stiffness = 1e5; gripper_damping = 2000; gripper_force_limit = 20.0; gripper_friction = 1
    ee_link_name = 'eef'

    @property
    def _controller_configs(self):
        arm = PDJointPosControllerConfig(self.arm_joint_names, lower=None, upper=None, stiffness=self.arm_stiffness, damping=self.arm_damping, force_limit=self.arm_force_limit, normalize_action=False)
        # all six finger joints follow the right outer knuckle (parallelogram: inner knuckle = +theta, inner finger = -theta)
        mimic = dict(left_outer_knuckle_joint=dict(joint='right_outer_knuckle_joint', multiplier=1.0, offset=0.0),
                     left_inner_knuckle_joint=dict(joint='right_outer_knuckle_joint', multiplier=1.0, offset=0.0), right_inner_knuckle_joint=dict(joint='right_outer_knuckle_joint', multiplier=1.0, offset=0.0),
                     left_inner_finger_joint=dict(joint='right_outer_knuckle_joint', multiplier=-1.0, offset=0.0), right_inner_finger_joint=dict(joint='right_outer_knuckle_joint', multiplier=-1.0, offset=0.0))
        grip = PDJointPosMimicControllerConfig(['left_outer_knuckle_joint', 'right_outer_knuckle_joint', 'left_inner_knuckle_joint', 'right_inner_knuckle_joint', 'left_inner_finger_joint', 'right_inner_finger_joint'],
                                               lower=None, upper=None, stiffness=self.gripper_stiffness, damping=self.gripper_damping, force_limit=self.gripper_force_limit, friction=self.gripper_friction, normalize_action=False, mimic=mimic)
        return deepcopy_dict(dict(pd_joint_pos=dict(arm=arm, gripper_active=grip)))

    def _after_loading_articulation(self):
        # 4-bar closure of each finger (constants from ManiSkill's XArm6Robotiq)
        for side, p_f, p_p in (('right', [-1.6048949e-08, 3.7600022e-02, 4.3000020e-02], [1.3578170e-09, -1.7901104e-02, 6.5159947e-03]),
                               ('left', [-1.8080145e-08, 3.7600014e-02, 4.2999994e-02], [-1.4041154e-08, -1.7901093e-02, 6.5159872e-03])):
            pad = self.robot.active_joints_map[f'{side}_inner_finger_joint'].get_child_link(); lif = self.robot.active_joints_map[f'{side}_inner_knuckle_joint'].get_child_link()
            d = self.scene.create_drive(lif, sapien.Pose(p_f), pad, sapien.Pose(p_p)); d.set_limit_x(0, 0); d.set_limit_y(0, 0); d.set_limit_z(0, 0)
        for name in ['right_inner_knuckle', 'right_outer_knuckle', 'left_inner_knuckle', 'left_outer_knuckle', 'right_inner_finger_pad', 'left_inner_finger_pad', 'right_outer_finger', 'left_outer_finger',
                     'robotiq_arg2f_base_link', 'right_inner_finger', 'left_inner_finger', 'panda_link7', 'panda_link6']:
            self.robot.links_map[name].set_collision_group_bit(group=2, bit_idx=31, bit=1)

    def _after_init(self):
        self.finger1_link = sapien_utils.get_obj_by_name(self.robot.get_links(), 'left_inner_finger_pad')
        self.finger2_link = sapien_utils.get_obj_by_name(self.robot.get_links(), 'right_inner_finger_pad')
        self.tcp = sapien_utils.get_obj_by_name(self.robot.get_links(), self.ee_link_name)
        self._width_table = None

    def width_table(self):
        """Recorded opening width (m) <-> outer-knuckle angle, measured once on the loaded articulation."""
        if self._width_table is None:
            q0 = self.robot.get_qpos().clone(); thetas = np.linspace(0.0, 0.81, 41); widths = []; names = [j.name for j in self.robot.active_joints][7:]
            for th in thetas:
                q = q0.clone(); q[0, 7:13] = torch.tensor(gripper_qpos(th, names)); self.robot.set_qpos(q)
                widths.append(float(torch.linalg.norm(self.finger1_link.pose.p[0] - self.finger2_link.pose.p[0])))
            self.robot.set_qpos(q0); widths = np.array(widths); self._width_table = (thetas, widths - widths[-1] + 0.0)
            self._width_scale = W_OPEN / max(self._width_table[1][0], 1e-6)
        return self._width_table

    def theta_from_width(self, w):
        thetas, widths = self.width_table(); w = float(np.clip(w / self._width_scale, widths[-1], widths[0]))
        return float(np.interp(-w, -widths, thetas))

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        lf = self.scene.get_pairwise_contact_forces(self.finger1_link, object); rf = self.scene.get_pairwise_contact_forces(self.finger2_link, object)
        lforce, rforce = torch.linalg.norm(lf, axis=1), torch.linalg.norm(rf, axis=1)
        ld = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]; rd = self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        la, ra = common.compute_angle_between(ld, lf), common.compute_angle_between(rd, rf)
        return torch.logical_and(torch.logical_and(lforce >= min_force, torch.rad2deg(la) <= max_angle), torch.logical_and(rforce >= min_force, torch.rad2deg(ra) <= max_angle))

    def is_static(self, threshold: float = 0.2):
        return torch.max(torch.abs(self.robot.get_qvel()[..., :7]), 1)[0] <= threshold

    @property
    def tcp_pos(self): return self.tcp.pose.p

    @property
    def tcp_pose(self): return self.tcp.pose


# ---------------------------------------------------------------- xArm7 + pusher rod (Push-T)
XARM7_PUSHER_URDF = paths.asset('maniskill', 'xarm7_pusher', 'xarm7_pusher.urdf')
XARM7_REST_QPOS = np.deg2rad([0.0, -45.0, 0.0, 30.0, 0.0, 75.0, 0.0]).astype(np.float32)
XARM7_JOINT_LIMITS = np.array([[-6.283, 6.283], [-2.059, 2.0944], [-6.283, 6.283], [-0.19198, 3.927], [-6.283, 6.283], [-1.69297, 3.14159], [-6.283, 6.283]])


@register_agent()
class XArm7Pusher(BaseAgent):
    """xArm7 with a 20 cm pusher rod fused to link7; the TCP `link_tcp` is the rod tip (link7 + 0.215 m along +z).
    qpos: the 7 arm joints; nothing is ever grasped."""
    uid = 'xarm7_pusher'
    urdf_path = str(XARM7_PUSHER_URDF)
    urdf_config = dict(_materials=dict(pusher=dict(static_friction=0.6, dynamic_friction=0.6, restitution=0.0)),
                       link=dict(link7=dict(material='pusher', patch_radius=0.05, min_patch_radius=0.05)))
    keyframes = dict(rest=Keyframe(qpos=XARM7_REST_QPOS.copy(), pose=sapien.Pose()))
    arm_joint_names = [f'joint{i}' for i in range(1, 8)]
    arm_stiffness = 1e3; arm_damping = 1e2; arm_force_limit = 100
    ee_link_name = 'link_tcp'

    @property
    def _controller_configs(self):
        arm = PDJointPosControllerConfig(self.arm_joint_names, lower=None, upper=None, stiffness=self.arm_stiffness, damping=self.arm_damping, force_limit=self.arm_force_limit, normalize_action=False)
        return deepcopy_dict(dict(pd_joint_pos=dict(arm=arm)))

    def _after_init(self):
        self.tcp = sapien_utils.get_obj_by_name(self.robot.get_links(), self.ee_link_name)
        self.rod_link = sapien_utils.get_obj_by_name(self.robot.get_links(), 'link7')

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        return torch.zeros(1, dtype=torch.bool, device=self.device)

    def is_static(self, threshold: float = 0.2):
        return torch.max(torch.abs(self.robot.get_qvel()[..., :7]), 1)[0] <= threshold

    @property
    def tcp_pos(self): return self.tcp.pose.p

    @property
    def tcp_pose(self): return self.tcp.pose


# ---------------------------------------------------------------- environment
MS3_TABLE_ARENA = 'ms3_table_scene'   # profile-provided static scene: ManiSkill's TableSceneBuilder
PANDA_REST_QPOS = np.array([0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, np.pi * 3 / 4, np.pi / 4, 0.04, 0.04], dtype=np.float32)
PANDA_BASE = (-0.615, 0.0, 0.0)
MAX_DEPEN_VEL = 10.0   # m/s: cap on PhysX depenetration speed for scene objects of the DROID rig
SIM_FREQ, CONTROL_FREQ = 100, 20   # the rollout raises both for the DROID rig
REST_QPOS = {'panda': PANDA_REST_QPOS, 'panda_robotiq': PANDA_REST_QPOS, 'xarm7_pusher': XARM7_REST_QPOS}
HAS_GRIPPER = {'panda': True, 'panda_robotiq': True, 'xarm7_pusher': False}


@register_env("V2SPanda-v1", max_episode_steps=1_000_000)
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
        BaseEnv._load_lighting(self, options)

    def _load_scene(self, options: dict):
        sp = self.scene_spec
        if (sp.get('arena') or {}).get('library') == MS3_TABLE_ARENA:
            self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0.0); self.table_scene.build()
            for p_ in sp.get('props') or []:
                self.props[p_['name']] = self._build_entity(p_, static=True)
            for o in sp.get('objects') or []:
                self.objs[o['name']] = self._build_entity(o, static=False)
            if sp.get('support'):   # a declared support slab is built as a static box on top of the table
                s_ = sp['support']; cx, cy = s_.get('center', (0.0, 0.0)); lx, ly = s_.get('size', (1.0, 1.0)); th = float(s_.get('thickness', 0.05))
                self.props['support'] = self._build_entity(dict(name='support', kind='box', half_size=[lx / 2, ly / 2, th / 2],
                                                                pos=[cx, cy, float(s_['z']) - th / 2], color=s_.get('color', (0.55, 0.35, 0.2, 1.0))), static=True)
        else:
            self.table_scene = None
            V2SBridgeEnv._load_scene(self, options)
        if self.robot_uid == 'panda_robotiq':
            # cap how fast PhysX pushes interpenetrating bodies apart: a held part pressed into a thin slab is otherwise thrown away
            for a_ in self.objs.values():
                for ent in a_._objs:
                    comp = ent.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
                    if comp is not None: comp.set_max_depenetration_velocity(MAX_DEPEN_VEL)

    def _build_entity(self, e: dict, static: bool):
        """Static mesh props keep their concave triangle mesh (receiver holes stay open); dynamic parts use the shared
        convex decomposition."""
        if not (static and e.get('kind') == 'mesh'): return V2SBridgeEnv._build_entity(self, e, static)
        b = self.scene.create_actor_builder(); s_ = [float(e.get('scale', 1.0))] * 3
        mat = sapien.physx.PhysxMaterial(static_friction=float(e.get('static_friction', 1.0)), dynamic_friction=float(e.get('dynamic_friction', 1.0)), restitution=0.0)
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
                self.table_scene.initialize(env_idx)
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

    def robot_qpos(self, q):
        """Franka-layout qpos (7 arm + 2 fingers, finger = half the opening) mapped to the loaded robot."""
        q = np.asarray(q, dtype=np.float32)
        if self.robot_uid == 'xarm7_pusher': return q[:7]
        if self.robot_uid == 'panda_robotiq' and len(q) == 9:
            th = self.agent.theta_from_width(2.0 * float(q[7])); names = [j.name for j in self.agent.robot.active_joints][7:]; return np.r_[q[:7], gripper_qpos(th, names)].astype(np.float32)
        return q

    def gripper_cmd(self, width_m):
        """Gripper action for an opening width: Robotiq outer-knuckle angle (rad); Franka normalised finger target."""
        if self.robot_uid == 'panda_robotiq': return float(self.agent.theta_from_width(width_m))
        if self.robot_uid == 'xarm7_pusher': return 0.0
        return float(np.clip((np.clip(width_m / 2, 0, 0.04) + 0.01) / 0.05 * 2 - 1, -1, 1))

    def qpos_np(self) -> np.ndarray:
        return self.agent.robot.get_qpos()[0].cpu().numpy().astype(np.float64)

    def tcp_pose_np(self) -> np.ndarray:
        p = self.agent.tcp.pose
        return np.r_[p.p[0].cpu().numpy(), p.q[0].cpu().numpy()].astype(np.float64)
