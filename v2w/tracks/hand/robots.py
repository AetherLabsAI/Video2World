"""ManiSkill agents of the hand track's dexterous setting: the Wuji hand (wuji-technology/wuji-description, MIT) on a 6-DoF
floating base, single (`wuji_right_floating` / `wuji_left_floating`) and as a 52-DoF two-hand articulation
(`wuji_bimanual_floating`).

Palm frame (right hand, URDF): fingers along +z, thumb tip on +y, palm normal along x. Root chain R = Rx(a) Ry(b) Rz(c),
i.e. scipy 'XYZ' intrinsic Euler angles. Contract finger order: finger1 (thumb) joint1..4, ..., finger5 joint1..4 (URDF
names `{side}_finger{n}_joint{m}`); the articulation's own active-joint order differs and `contract_to_qpos` maps it.
"""
import json
from types import SimpleNamespace

import numpy as np
import sapien
import torch
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig, deepcopy_dict
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils

from v2w import paths
from .physics import collision_exclusions, joint_limits

ASSETS = paths.asset('hand')
ROOT_JOINTS = ['root_x_axis_joint', 'root_y_axis_joint', 'root_z_axis_joint', 'root_x_rot_joint', 'root_y_rot_joint', 'root_z_rot_joint']
SIDES = ('left', 'right')


def contract_joint_names(side):
    return [f'{side}_finger{n}_joint{m}' for n in range(1, 6) for m in range(1, 5)]


def open_fingers(side='right'):
    """The open pinch posture (20 contract-order joints) shipped with the hand model."""
    return np.asarray(json.loads((ASSETS / f'wuji_{side}_floating/pinch.json').read_text())['open'])


class _WujiFloating(BaseAgent):
    side = 'right'
    root_pos_stiffness = 3e4; root_pos_damping = 6e2; root_pos_force = 200.0     # prismatic root joints: track a 0.5 m/s wrist within a few mm
    root_rot_stiffness = 6e2; root_rot_damping = 2e1; root_rot_force = 100.0     # revolute root joints (hand inertia about the wrist ~5e-3 kg m^2)
    finger_stiffness = 30.0; finger_damping = 1.5
    urdf_config = dict(_materials=dict(finger=dict(static_friction=1.5, dynamic_friction=1.2, restitution=0.0)))

    def __init__(self, *args, **kwargs):
        self.finger_joint_names = contract_joint_names(self.side)
        self.ee_link_name = f'{self.side}_palm_link'
        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        root_pos = PDJointPosControllerConfig(ROOT_JOINTS[:3], lower=None, upper=None, stiffness=self.root_pos_stiffness, damping=self.root_pos_damping, force_limit=self.root_pos_force, normalize_action=False)
        root_rot = PDJointPosControllerConfig(ROOT_JOINTS[3:], lower=None, upper=None, stiffness=self.root_rot_stiffness, damping=self.root_rot_damping, force_limit=self.root_rot_force, normalize_action=False)
        efforts = joint_limits(self.urdf_path)
        fingers = PDJointPosControllerConfig(self.finger_joint_names, lower=None, upper=None, stiffness=self.finger_stiffness, damping=self.finger_damping, force_limit=[efforts[n]['effort'] for n in self.finger_joint_names], normalize_action=False)
        return deepcopy_dict(dict(pd_joint_pos=dict(root_pos=root_pos, root_rot=root_rot, fingers=fingers)))

    def _after_init(self):
        links = self.robot.get_links()
        self.palm = sapien_utils.get_obj_by_name(links, self.ee_link_name); self.tcp = self.palm
        self.tip_links = [sapien_utils.get_obj_by_name(links, f'{self.side}_finger{n}_tip_link') for n in range(1, 6)]
        self.finger_links = {n: [sapien_utils.get_obj_by_name(links, f'{self.side}_finger{n}_link{m}') for m in range(1, 5)] + [self.tip_links[n - 1]] for n in range(1, 6)}
        masks = collision_exclusions(self.urdf_path)
        for link in links:
            for native in link._objs:
                for shape in native.get_collision_shapes():
                    groups = list(shape.get_collision_groups()); groups[2] = masks[link.name]
                    shape.set_collision_groups(groups)
        self.self_collision_contract = dict(enabled=True, exclusion='URDF graph distance <= 2', ignore_masks=masks)
        self.active_names = [j.name for j in self.robot.active_joints]
        self._contract_idx = [self.active_names.index(n) for n in self.finger_joint_names]
        self._root_idx = [self.active_names.index(n) for n in ROOT_JOINTS]

    def contract_to_qpos(self, root6, fingers20):
        """(root xyz + XYZ euler, 20 contract-order joints) -> full qpos in the articulation's active-joint order"""
        q = np.zeros(len(self.active_names), np.float32)
        for k, i in enumerate(self._root_idx): q[i] = root6[k]
        for k, i in enumerate(self._contract_idx): q[i] = fingers20[k]
        return q

    def is_static(self, threshold: float = 0.2):
        return torch.max(torch.abs(self.robot.get_qvel()[..., :6]), 1)[0] <= threshold

    @property
    def tcp_pose(self): return self.palm.pose


@register_agent()
class WujiRightFloating(_WujiFloating):
    uid = 'wuji_right_floating'; side = 'right'; urdf_path = str(ASSETS / 'wuji_right_floating/wuji_right_floating.urdf')
    keyframes = dict(rest=Keyframe(qpos=np.zeros(26, np.float32), pose=sapien.Pose()))


@register_agent()
class WujiLeftFloating(_WujiFloating):
    uid = 'wuji_left_floating'; side = 'left'; urdf_path = str(ASSETS / 'wuji_left_floating/wuji_left_floating.urdf')
    keyframes = dict(rest=Keyframe(qpos=np.zeros(26, np.float32), pose=sapien.Pose()))


@register_agent()
class WujiBimanualFloating(BaseAgent):
    """Two independent, force-limited Wuji hands in one 52-DoF articulation; cross-hand collisions stay enabled."""
    uid = 'wuji_bimanual_floating'
    urdf_path = str(ASSETS / uid / (uid + '.urdf'))
    urdf_config = _WujiFloating.urdf_config
    keyframes = dict(rest=Keyframe(qpos=np.zeros(52, np.float32), pose=sapien.Pose()))

    @property
    def _controller_configs(self):
        config = {}; limits = joint_limits(self.urdf_path)
        for side in SIDES:
            root = [f'{side}_{n}' for n in ROOT_JOINTS]; names = contract_joint_names(side)
            for suffix, joints, stiffness, damping, force in (
                    ('root_pos', root[:3], 3e4, 6e2, 200.), ('root_rot', root[3:], 6e2, 2e1, 100.),
                    ('fingers', names, 30., 1.5, [limits[n]['effort'] for n in names])):
                config[f'{side}_{suffix}'] = PDJointPosControllerConfig(joints, lower=None, upper=None,
                                                                         stiffness=stiffness, damping=damping, force_limit=force, normalize_action=False)
        return deepcopy_dict(dict(pd_joint_pos=config))

    def _after_init(self):
        self.active_names = [j.name for j in self.robot.active_joints]
        links = {l.name: l for l in self.robot.links}; self.hands = {}
        for identifier, side in enumerate(SIDES, start=1):
            masks = collision_exclusions(ASSETS / f'wuji_{side}_floating' / f'wuji_{side}_floating.urdf')
            for original, mask in masks.items():
                name = f'{side}_{original}' if original.startswith('root') else original
                for native in links[name]._objs:
                    for shape in native.get_collision_shapes():
                        groups = list(shape.get_collision_groups()); groups[2] = mask
                        # SAPIEN ignores matching g2 bits only for equal low-16-bit g3 ids: separate ids keep cross-hand collisions.
                        groups[3] = (groups[3] & 0xffff0000) | identifier
                        shape.set_collision_groups(groups)
                        if list(shape.get_collision_groups()) != groups:
                            raise RuntimeError('bimanual collision filter readback mismatch')
            self.hands[side] = SimpleNamespace(side=side, palm=links[f'{side}_palm_link'],
                                               root_idx=[self.active_names.index(f'{side}_{n}') for n in ROOT_JOINTS],
                                               finger_idx=[self.active_names.index(n) for n in contract_joint_names(side)],
                                               finger_links={n: [links[f'{side}_finger{n}_link{m}'] for m in range(1, 5)] +
                                                             [links[f'{side}_finger{n}_tip_link']] for n in range(1, 6)})
        self.palm = self.hands['left'].palm; self.tcp = self.palm
        self.self_collision_contract = dict(enabled=True, inter_hand_enabled=True, exclusion='within each hand only: URDF graph distance <= 2',
                                            hand_collision_ids=dict(left=1, right=2), verified=True)

    def contract_to_qpos(self, roots, fingers):
        q = np.zeros(len(self.active_names), np.float32)
        for side, hand in self.hands.items():
            q[hand.root_idx] = roots[side]; q[hand.finger_idx] = fingers[side]
        return q
