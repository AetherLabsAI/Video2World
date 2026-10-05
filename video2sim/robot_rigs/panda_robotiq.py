"""ManiSkill agent `panda_robotiq`: Franka Panda arm + Robotiq 2F-85 gripper, the DROID robot (replay_fb.py R1 done properly:
the simulated fingertips are where the real ones were, so no tool offset is needed).

URDF: assets/panda_robotiq.urdf (build_panda_robotiq_urdf.py). Gripper closure (4-bar linkage) and grasp test follow
ManiSkill's XArm6Robotiq. qpos layout: 7 arm joints + 6 gripper joints (URDF order, see GRIPPER_JOINTS); the single
gripper action is the outer-knuckle angle in radians (0 = open, 0.81 = closed). `theta_from_width` converts a recorded
Robotiq opening width (m) into that angle using the URDF kinematics (table measured once on the loaded articulation)."""
from pathlib import Path
import numpy as np, sapien, torch
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig, PDJointPosMimicControllerConfig, deepcopy_dict
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor

URDF = Path(__file__).resolve().parent / 'assets/panda_robotiq.urdf'
GRIPPER_JOINTS = ['left_outer_knuckle_joint', 'left_inner_knuckle_joint', 'left_inner_finger_joint', 'right_outer_knuckle_joint', 'right_inner_knuckle_joint', 'right_inner_finger_joint']
W_OPEN = 0.085


def gripper_qpos(theta, joint_names=GRIPPER_JOINTS):
    """The Robotiq joint angles for an outer-knuckle angle theta, in the given joint order (closure: inner knuckle =
    theta, inner finger = -theta). The loaded articulation orders the active joints differently from the URDF."""
    return np.array([-theta if 'inner_finger' in n else theta for n in joint_names], np.float32)


@register_agent(asset_download_ids=['xarm6'])
class PandaRobotiq(BaseAgent):
    uid = 'panda_robotiq'
    urdf_path = str(URDF)
    urdf_config = dict(_materials=dict(gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)),
                       link=dict(left_inner_finger_pad=dict(material='gripper', patch_radius=0.1, min_patch_radius=0.1), right_inner_finger_pad=dict(material='gripper', patch_radius=0.1, min_patch_radius=0.1)))
    keyframes = dict(rest=Keyframe(qpos=np.r_[[0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, np.pi * 3 / 4, np.pi / 4], np.zeros(6)], pose=sapien.Pose()))
    arm_joint_names = [f'panda_joint{i}' for i in range(1, 8)]
    arm_stiffness = 1e3; arm_damping = 1e2; arm_force_limit = 100
    gripper_stiffness = 1e5; gripper_damping = 2000; gripper_force_limit = 20.0; gripper_friction = 1   # knuckle torque limit: enough to actually squeeze a part (xarm6_robotiq's 0.1 barely touches)
    ee_link_name = 'eef'

    @property
    def _controller_configs(self):
        arm = PDJointPosControllerConfig(self.arm_joint_names, lower=None, upper=None, stiffness=self.arm_stiffness, damping=self.arm_damping, force_limit=self.arm_force_limit, normalize_action=False)
        # all six finger joints follow the right outer knuckle (parallelogram: inner knuckle = +theta, inner finger = -theta).
        # Leaving the inner joints passive (ManiSkill's XArm6-Robotiq) only works with its 0.1 N m force limit: under a real
        # squeeze the soft loop-closure drive yields, the inner finger joint runs to its limit and the pads tilt 10-16 deg.
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

    # --- recorded Robotiq width (m) <-> outer-knuckle angle, measured on the loaded articulation once
    def width_table(self):
        if self._width_table is None:
            q0 = self.robot.get_qpos().clone(); thetas = np.linspace(0.0, 0.81, 41); widths = []; names = [j.name for j in self.robot.active_joints][7:]
            for th in thetas:
                q = q0.clone(); q[0, 7:13] = torch.tensor(gripper_qpos(th, names)); self.robot.set_qpos(q)
                widths.append(float(torch.linalg.norm(self.finger1_link.pose.p[0] - self.finger2_link.pose.p[0])))
            self.robot.set_qpos(q0); widths = np.array(widths); self._width_table = (thetas, widths - widths[-1] + 0.0)   # pad-centre distance; closed = 0 mm opening
            self._width_scale = W_OPEN / max(self._width_table[1][0], 1e-6)
        return self._width_table

    def theta_from_width(self, w):
        thetas, widths = self.width_table(); w = float(np.clip(w / self._width_scale, widths[-1], widths[0]))
        return float(np.interp(-w, -widths, thetas))   # widths decrease with theta

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
