"""`V2SEgoHand-v1`: the hand-track simulator profile. The scene contract is the one every family uses (support / props /
objects built by `V2SBridgeEnv._build_entity`), plus

* recovered one-DoF joints realised as SAPIEN articulations (the parent stays a static, possibly non-convex prop),
* an optional robot (Franka arm + gripper, or Wuji hands on floating bases) driven by joint position targets,
* a moving camera (`set_camera_pose`), because an egocentric camera is carried on the demonstrator's head.

Nothing calls `env.step`: the executors set robot targets and advance physics with `step_physics()` at `sim_freq`.
"""
from __future__ import annotations

import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig
from sapien.physx import PhysxMaterial
from video2sim.bench.bridge_env import V2SBridgeEnv, CV_TO_SAPIEN

SIM_FREQ, CONTROL_FREQ = 300, 30          # 30 Hz = the HOT3D frame rate; 10 physics substeps per frame


@register_env("V2SEgoHand-v1", max_episode_steps=1_000_000)
class V2SEgoHandEnv(V2SBridgeEnv):
    SUPPORTED_OBS_MODES = ["state", "none"]
    SUPPORTED_REWARD_MODES = ["none"]

    def __init__(self, *args, scene_spec: dict, camera: dict | None = None, enable_cameras: bool = True,
                 joint_friction: float = 0.5, joint_damping: float = 5.0, robot_uid: str = 'none', robot_base_pose=None, **kwargs):
        self.scene_spec = scene_spec
        self.camera_spec = camera
        self.enable_cameras = enable_cameras
        from video2sim.bench.bridge_env import library_info
        self._info = library_info()
        self.objs, self.props, self.arena, self.settled_poses = {}, {}, None, {}
        self.arts, self.joints, self._art_geom = {}, {}, {}
        self.joint_friction, self.joint_damping = joint_friction, joint_damping
        # robot embodiment: 'panda' (Franka arm + gripper at robot_base_pose), 'wuji_{right,left}_floating' or
        # 'wuji_bimanual_floating' (dexterous hands on floating bases); 'none' builds the scene only
        self.robot_uid = robot_uid; self.robot_base_pose = list(robot_base_pose) if robot_base_pose is not None else [0, 0, 0, 1, 0, 0, 0]
        if robot_uid.startswith('wuji'):
            from . import robots  # noqa: F401  registers the Wuji agents
        if robot_uid != 'none': kwargs.setdefault('control_mode', 'pd_joint_pos')
        BaseEnv.__init__(self, *args, robot_uids=robot_uid, **kwargs)

    def _load_agent(self, options: dict):
        if self.robot_uid == 'none': return
        bp = self.robot_base_pose
        BaseEnv._load_agent(self, options, sapien.Pose(p=bp[:3], q=bp[3:7]))

    # ---- robot helpers (settings A / B)
    def has_robot(self): return self.robot_uid != 'none'

    def set_robot_targets(self, q, qvel=None):
        """position (and optional velocity feed-forward) targets for ALL active joints (articulation order) — the pd_joint_pos drives.
        Without the velocity target a PD drive lags a moving command by damping x speed / stiffness (1 cm at 0.5 m/s on the Wuji root)."""
        qv = np.zeros_like(np.asarray(q, np.float32)) if qvel is None else np.asarray(qvel, np.float32)
        for i, (j, v) in enumerate(zip(self.agent.robot.active_joints, np.asarray(q, np.float32))):
            j.set_drive_target(torch.tensor([float(v)]))
            j.set_drive_velocity_target(torch.tensor([float(qv[i])]))

    def stop_robot_velocity_targets(self):
        for joint in self.agent.robot.active_joints:
            joint.set_drive_velocity_target(torch.zeros(1))

    def robot_qpos_np(self): return self.agent.robot.get_qpos()[0].cpu().numpy().astype(np.float64)

    def robot_link_entity(self, name):
        return next(l for l in self.agent.robot.links if l.name == name)._objs[0]

    def robot_link_pose_np(self, name):
        l = next(l for l in self.agent.robot.links if l.name == name); return np.r_[l.pose.p[0].cpu().numpy(), l.pose.q[0].cpu().numpy()].astype(np.float64)

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=SIM_FREQ, control_freq=CONTROL_FREQ, spacing=20)

    def _load_lighting(self, options: dict):
        BaseEnv._load_lighting(self, options)

    def _add_geometry(self, b, e: dict, visual: bool):
        """The upstream builder, except that a static prop may declare `nonconvex: true` and keep its exact mesh as
        collision geometry (a convex decomposition would fill in a cavity)."""
        if e.get('kind') == 'mesh' and e.get('nonconvex'):
            b.add_nonconvex_collision_from_file(filename=e['collision_path'],
                                                scale=[float(e.get('scale', 1.0))] * 3,
                                                material=self._material(e))
            if visual:
                b.add_visual_from_file(filename=e.get('mesh_path') or e['collision_path'],
                                       scale=[float(e.get('scale', 1.0))] * 3)
            return
        V2SBridgeEnv._add_geometry(self, b, e, visual)

    @staticmethod
    def _joint_frame(axis, point):
        """SAPIEN puts the joint axis on the joint frame's X axis; build such a frame on `axis` at `point`."""
        a = np.asarray(axis, float); a = a / (np.linalg.norm(a) or 1.0)
        t = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        y = np.cross(t, a); y /= (np.linalg.norm(y) or 1.0)
        z = np.cross(a, y)
        M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2] = a, y, z; M[:3, 3] = np.asarray(point, float)
        return sapien.Pose(M)

    def _build_articulations(self, joints, spec, mat):
        """Realise each recovered one-DoF joint as a SAPIEN articulation. The parent stays a static prop (keeping its exact,
        possibly non-convex collision); the root link carries no geometry and anchors the joint; the child link carries
        the moving part."""
        by_name = {e['name']: e for e in (spec.get('props') or []) + (spec.get('objects') or [])}
        for j in joints:
            par, chi = by_name.get(j['parent']), by_name.get(j['child'])
            if par is None or chi is None:
                continue
            T_b_par = sapien.Pose(p=[float(v) for v in par['pos']], q=[float(v) for v in par['quat']])
            T_b_chi = sapien.Pose(p=[float(v) for v in chi['pos']], q=[float(v) for v in chi['quat']])
            T_par_joint = self._joint_frame(j['axis'], j['point'])
            T_chi_joint = (T_b_par.inv() * T_b_chi).inv() * T_par_joint

            ab = self.scene.create_articulation_builder()
            root = ab.create_link_builder()
            root.set_name(f"{j['parent']}__anchor")
            link = ab.create_link_builder(root)
            link.set_name(j['child'])
            self._add_geometry(link, chi, self.enable_cameras)
            link.set_joint_name(j['name'])
            lo, hi = float(min(j['limits'])), float(max(j['limits']))
            span = max(hi - lo, 1e-3)
            pad = 0.5 * span + (0.175 if j['type'] == 'revolute' else 0.05)
            link.set_joint_properties(
                'revolute' if j['type'] == 'revolute' else 'prismatic',
                [[lo - pad, hi + pad]], T_par_joint, T_chi_joint,
                float(j.get('friction', self.joint_friction)), float(j.get('damping', self.joint_damping)))
            ab.set_initial_pose(T_b_par)
            art = ab.build(name=f"art_{j['child']}", fix_root_link=True)
            # ManiSkill's LinkBuilder stores friction but does not assign it to PhysX.
            desired_friction=float(j.get('friction',self.joint_friction))
            for native in art.active_joints[0]._objs:
                native.set_friction(desired_friction)
                if abs(float(native.friction)-desired_friction)>1e-6:
                    raise RuntimeError('articulation joint friction readback mismatch')
            try:
                art.set_qpos(torch.zeros(1, 1))
            except Exception:
                pass
            links = {l.name: l for l in art.get_links()}
            self.objs[j['child']] = links.get(j['child'], art.get_links()[-1])
            self.arts[j['child']] = art
            self.joints[j['child']] = j
            axis = np.asarray(j['axis'], float); axis /= (np.linalg.norm(axis) or 1.0)
            T_par_chi = T_b_par.inv() * T_b_chi
            from scipy.spatial.transform import Rotation as _R
            q0 = np.asarray(T_par_chi.q, float)
            self._art_geom[j['child']] = dict(
                axis=axis, point=np.asarray(j['point'], float), p0=np.asarray(T_par_chi.p, float),
                R0=_R.from_quat([q0[1], q0[2], q0[3], q0[0]]).as_matrix(),
                T_b_par=T_b_par, limits=(lo - pad, hi + pad))

    def joint_q_now(self, name):
        art = self.arts.get(name)
        return None if art is None else float(art.get_qpos()[0, 0].cpu().numpy())

    def set_joint_q(self, name, q):
        art = self.arts.get(name)
        if art is None or q is None:
            return
        art.set_qpos(torch.tensor([[float(q)]], device=self.device))
        art.set_qvel(torch.zeros(1, 1, device=self.device))

    def _load_scene(self, options: dict):
        # a joint CHILD must not also be built as a free rigid body: hold it out of the upstream builder and
        # build it inside the articulation below.  The parent stays a static prop, non-convex collision and all.
        joints = list(self.scene_spec.get('joints') or [])
        full = self.scene_spec
        if joints:
            drop = {j['child'] for j in joints}
            self.scene_spec = {**full,
                               'props': [p for p in (full.get('props') or []) if p['name'] not in drop],
                               'objects': [o for o in (full.get('objects') or []) if o['name'] not in drop]}
        V2SBridgeEnv._load_scene(self, options)
        self.scene_spec = full
        mat = PhysxMaterial(static_friction=1.2, dynamic_friction=1.0, restitution=0.0)
        if joints:
            self._build_articulations(joints, full, mat)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            for o in self.scene_spec.get('objects') or []:
                if o['name'] in self.arts:
                    # the joint frames were built from the frame-0 relative pose, so q=0 IS this pose
                    self.arts[o['name']].set_qpos(torch.zeros(1, 1))
                    self.arts[o['name']].set_qvel(torch.zeros(1, 1))
                    continue
                a = self.objs[o['name']]
                a.set_pose(Pose.create_from_pq(p=torch.tensor([[float(v) for v in o['pos']]]),
                                               q=torch.tensor([[float(v) for v in o.get('quat', (1, 0, 0, 0))]])))
                a.set_linear_velocity(torch.zeros(1, 3)); a.set_angular_velocity(torch.zeros(1, 3))
            if self.has_robot() and options.get('qpos') is not None:
                q = np.asarray(options['qpos'], np.float32); self.agent.reset(init_qpos=q); self.set_robot_targets(q)
            self._settle(float(options.get('settle', 0.0)))
            self.settled_poses = {n: self.obj_pose_np(n) for n in self.objs}

    # ------------------------------------------------------------------ camera
    def set_camera_pose(self, T_base_cam):
        """Re-pose the eval camera (OpenCV T_base_cam) -- the egocentric camera moves every frame."""
        cam = (self._sensors or {}).get('eval_camera')
        if cam is None:
            return
        T = np.asarray(T_base_cam, dtype=np.float64).copy()
        T[:3, :3] = T[:3, :3] @ CV_TO_SAPIEN
        cam.camera.set_local_pose(sapien.Pose(T))

    def render_rgb(self):
        cam = (self._sensors or {}).get('eval_camera')
        if cam is None:
            return None
        self.scene.update_render()
        cam.capture()
        img = cam.get_obs(rgb=True, depth=False, segmentation=False)['rgb']
        return img[0].cpu().numpy().astype(np.uint8)

    # ------------------------------------------------------------------ ManiSkill plumbing
    def get_state_dict(self):
        return self.scene.get_sim_state()          # BaseEnv's version asks the (absent) agent for controller state

    def set_state_dict(self, state: dict, env_idx=None):
        self.scene.set_sim_state(state, env_idx)

    def _get_obs_agent(self, *a, **k):
        return dict()

    def _step_action(self, action):
        return None                                # nothing calls env.step on this profile

    # ------------------------------------------------------------------ physics
    def step_physics(self, n: int = 1):
        for _ in range(n):
            self.scene.step()
