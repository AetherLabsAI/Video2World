"""ManiSkill agent `xarm7_pusher`: UFACTORY xArm7 with the 20 cm PUSHER ROD of real2sim-eval's Push-T rig (no gripper).

Added 2026-09-12 for the reconstructed-twins Push-T track (intake/reconstructed_twins, 3 demos of `xarm7_pusht`): the
candidate chain (fb/rollout_fb.py, fb/replay_fb.py) drives this agent exactly like the Franka — numerical IK from an
absolute TCP action stream — but there is no grasp: the `grip` column is ignored, `is_grasping` is always False.

URDF: assets/xarm7_pusher/xarm7_pusher.urdf = the upstream `xarm7_with_pusher.urdf` (real2sim-eval 8bd7091, arm meshes +
`pusher_20cm.stl` mounted 15 mm below link7) with a `link_tcp` frame added at the rod TIP: link7 + 0.215 m along +z.
The dataset's recorded end-effector pose is link7 (real2sim-eval kinematics_utils eef_name='link7'); the benchmark TCP
is the tip (the point that actually touches the block). qpos layout: the 7 arm joints only (the loaded articulation has no
other active joints). Rest pose = upstream construct_scene_pusher qpos [0,-45,0,30,0,75,0] deg.
"""
from pathlib import Path
import numpy as np, sapien, torch
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig, deepcopy_dict
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.actor import Actor

URDF = Path(__file__).resolve().parent / 'assets/xarm7_pusher/xarm7_pusher.urdf'
REST_QPOS = np.deg2rad([0.0, -45.0, 0.0, 30.0, 0.0, 75.0, 0.0]).astype(np.float32)
TCP_FROM_LINK7_M = 0.215        # rod tip along link7 +z (0.015 mount + 0.200 rod)
JOINT_LIMITS = np.array([[-6.283, 6.283], [-2.059, 2.0944], [-6.283, 6.283], [-0.19198, 3.927], [-6.283, 6.283], [-1.69297, 3.14159], [-6.283, 6.283]])


@register_agent(override=True)
class XArm7Pusher(BaseAgent):
    uid = 'xarm7_pusher'
    urdf_path = str(URDF)
    urdf_config = dict(_materials=dict(pusher=dict(static_friction=0.6, dynamic_friction=0.6, restitution=0.0)),
                       link=dict(link7=dict(material='pusher', patch_radius=0.05, min_patch_radius=0.05)))   # the rod is fused into link7 (fixed joint)
    keyframes = dict(rest=Keyframe(qpos=REST_QPOS.copy(), pose=sapien.Pose()))
    arm_joint_names = [f'joint{i}' for i in range(1, 8)]
    arm_stiffness = 1e3; arm_damping = 1e2; arm_force_limit = [50, 50, 30, 30, 30, 20, 20]
    ee_link_name = 'link_tcp'

    @property
    def _controller_configs(self):
        arm = PDJointPosControllerConfig(self.arm_joint_names, lower=None, upper=None, stiffness=self.arm_stiffness, damping=self.arm_damping, force_limit=self.arm_force_limit, normalize_action=False)
        return deepcopy_dict(dict(pd_joint_pos=dict(arm=arm)))

    def _after_init(self):
        self.tcp = sapien_utils.get_obj_by_name(self.robot.get_links(), self.ee_link_name)
        self.rod_link = sapien_utils.get_obj_by_name(self.robot.get_links(), 'link7')

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        return torch.zeros(1, dtype=torch.bool, device=self.device)   # no gripper: nothing is ever held

    def is_static(self, threshold: float = 0.2):
        return torch.max(torch.abs(self.robot.get_qvel()[..., :7]), 1)[0] <= threshold

    @property
    def tcp_pos(self): return self.tcp.pose.p

    @property
    def tcp_pose(self): return self.tcp.pose
