"""Execute a Bridge V2 action stream on the SIMPLER WidowX (ManiSkill 3).

Conventions, from rail-berkeley/bridge_data_robot:

    state = [x y z roll pitch yaw grip]          (widowx_env.get_full_state)
    R_link = R_zyx(roll, pitch, yaw) @ DEFAULT_ROTATION
    DEFAULT_ROTATION = [[0 0 1] [0 1 0] [-1 0 0]]  (gripper down <-> rpy ~ 0)
    step: next_T = delta_T @ prev_T with delta about the EE origin, i.e.
          R_next = R_zyx(d_rpy) @ R_prev,  p_next = p_prev + d_xyz
    (base frame; prev_T is the previous TARGET, not the measured pose)

SIMPLER's controller `arm_pd_ee_target_delta_pose_align2` is
`root_translation:root_aligned_body_rotation` with use_target=True, i.e.
exactly that composition, but it parses the 3 rotation channels as
pytorch3d "XYZ" Euler angles. We therefore convert Bridge's ZYX-extrinsic
delta into the matrix and back into the controller's convention, so the
composed rotation is identical rather than first-order close.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

DEFAULT_ROTATION = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
GRIPPER_OPEN_THRESHOLD = 0.5


def rot_zyx(rpy) -> np.ndarray:
    """R = Rz(yaw) Ry(pitch) Rx(roll)  (bridge eulerAnglesToRotationMatrix)."""
    return R.from_euler("xyz", np.asarray(rpy, dtype=np.float64)).as_matrix()


def state_to_T(state) -> np.ndarray:
    """Bridge 7-state -> 4x4 link pose (ee_gripper_link) in the base frame."""
    T = np.eye(4)
    T[:3, :3] = rot_zyx(state[3:6]) @ DEFAULT_ROTATION
    T[:3, 3] = np.asarray(state[:3], dtype=np.float64)
    return T


def T_to_state(T, grip: float = 1.0) -> np.ndarray:
    rpy = R.from_matrix(T[:3, :3] @ DEFAULT_ROTATION.T).as_euler("xyz")
    return np.r_[T[:3, 3], rpy, grip]


def _euler_xyz_pytorch3d(Rm: np.ndarray) -> np.ndarray:
    """Euler angles (a, b, c) such that pytorch3d euler_angles_to_matrix(.., "XYZ")
    == Rm, i.e. Rm = Rx(a) Ry(b) Rz(c)  -> scipy intrinsic 'XYZ'."""
    return R.from_matrix(Rm).as_euler("XYZ")


def bridge_action_to_sim(a: np.ndarray) -> np.ndarray:
    """(7,) bridge action -> (7,) SIMPLER action [dx dy dz ex ey ez grip]."""
    out = np.zeros(7, dtype=np.float32)
    out[:3] = a[:3]
    out[3:6] = _euler_xyz_pytorch3d(rot_zyx(a[3:6]))
    out[6] = 1.0 if a[6] > GRIPPER_OPEN_THRESHOLD else -1.0
    return out


class WidowXRig:
    """Thin handle over a running SIMPLER-style env (WidowX250S bridge agent)."""

    def __init__(self, env):
        self.env = env
        self.u = env.unwrapped
        self.robot = self.u.agent.robot
        self.ee = self.robot.links_map["ee_gripper_link"]
        self.arm = self.u.agent.controller.controllers["arm"]

    def T_world_base(self) -> np.ndarray:
        return self.robot.pose.to_transformation_matrix()[0].cpu().numpy().astype(np.float64)

    def ee_T_base(self) -> np.ndarray:
        T_we = self.ee.pose.to_transformation_matrix()[0].cpu().numpy().astype(np.float64)
        return np.linalg.inv(self.T_world_base()) @ T_we

    def ee_state(self) -> np.ndarray:
        g = float(self.robot.qpos[0, 6].cpu())
        return T_to_state(self.ee_T_base(), grip=g)

    def ik_base(self, T_base_link: np.ndarray, q0=None, seeds: int = 8):
        """Joint positions (6,) for a link pose given in the base frame, or None.

        Numerical IK (scipy least_squares, joint limits as bounds) over the
        pinocchio FK of the robot model; the sapien IK diverges on large jumps.
        Returns (q6 | None, residual) with residual = |dp| + 0.1 * |drot|.
        """
        from scipy.optimize import least_squares
        kin = self.arm.kinematics
        pm = kin.pmodel
        idx = int(kin.end_link_idx)
        # pinocchio FK is expressed in the robot ROOT frame (== base frame here)
        tgt_p, tgt_R = T_base_link[:3, 3], T_base_link[:3, :3]
        lim = self.robot.get_qlimits()[0, :6].cpu().numpy()
        q_now = self.robot.qpos[0].cpu().numpy() if q0 is None else np.asarray(q0)
        rest = q_now[:8].copy()

        def fk(q6):
            q = rest.copy(); q[:6] = q6
            pm.compute_forward_kinematics(q)
            P = pm.get_link_pose(idx)
            Tm = np.eye(4)
            Tm[:3, :3] = R.from_quat(np.r_[P.q[1:], P.q[0]]).as_matrix()
            Tm[:3, 3] = P.p
            return Tm

        def resid(q6):
            Tm = fk(q6)
            dr = R.from_matrix(tgt_R.T @ Tm[:3, :3]).as_rotvec()
            return np.r_[Tm[:3, 3] - tgt_p, 0.1 * dr]

        rng = np.random.default_rng(0)
        best = None
        for s in range(seeds):
            init = q_now[:6].copy() if s == 0 else rng.uniform(lim[:, 0], lim[:, 1])
            init = np.clip(init, lim[:, 0] + 1e-4, lim[:, 1] - 1e-4)
            r = least_squares(resid, init, bounds=(lim[:, 0], lim[:, 1]), xtol=1e-10, ftol=1e-10, max_nfev=2000, diff_step=1e-4)
            e = float(np.linalg.norm(r.fun))
            if best is None or e < best[1]:
                best = (r.x, e)
            if e < 2e-3:
                break
        q6, e = best
        return (q6 if e < 5e-3 else None), e

    def set_arm(self, q6, grip_open: bool = True, settle_steps: int = 50):
        g = 0.037 if grip_open else 0.015
        full = torch.tensor(np.r_[q6, g, g], dtype=torch.float32)[None]
        self.robot.set_qpos(full)
        self.robot.set_qvel(torch.zeros_like(full))
        # agent.reset() does NOT move the joint drive targets; without this the
        # PD drives pull the arm back toward the previous targets during settle
        for j, v in zip(self.robot.active_joints, full[0]):
            j.set_drive_target(v.reshape(1))
        for _ in range(settle_steps):
            self.u.scene.step()
        # re-anchor the controller's virtual target on the new pose
        self.arm.reset()
        gc = self.u.agent.controller.controllers["gripper"]
        gc.reset()

    def init_from_state(self, state) -> dict:
        q, err = self.ik_base(state_to_T(state))
        if q is None:
            return {"ok": False, "ik_err": err}
        self.set_arm(q, grip_open=state[6] > GRIPPER_OPEN_THRESHOLD)
        got = self.ee_state()
        return {"ok": True, "ik_err": err, "qpos": q.tolist(),
                "pos_err": float(np.linalg.norm(got[:3] - np.asarray(state[:3]))),
                "rot_err_deg": float(np.degrees(np.linalg.norm(
                    R.from_matrix(state_to_T(state)[:3, :3].T @ self.ee_T_base()[:3, :3]).as_rotvec())))}

    def step(self, bridge_action: np.ndarray):
        """Open-loop delta step, exactly as the real robot was commanded."""
        return self.env.step(bridge_action_to_sim(np.asarray(bridge_action, dtype=np.float64)))

    def target_T_base(self) -> np.ndarray:
        """The controller's current virtual target (link pose, base frame)."""
        tp = self.arm._target_pose
        T = np.eye(4)
        T[:3, :3] = R.from_quat(np.r_[tp.q[0, 1:].cpu().numpy(), tp.q[0, 0].cpu().numpy()]).as_matrix()
        T[:3, 3] = tp.p[0].cpu().numpy()
        return T

    def step_to_state(self, state: np.ndarray, grip_cmd: float | None = None):
        """Drive the virtual target exactly onto a recorded absolute state.

        `grip_cmd`: the recorded gripper COMMAND (1 open / 0 close). The
        recorded gripper STATE is not a substitute — a real gripper holding a
        3 cm block reads ~0.56 and would be misread as "open".

        With use_target=True the controller integrates our delta onto its own
        target, so delta = state (-) target reproduces the recorded EE
        trajectory with no accumulated drift.
        """
        T_t = self.target_T_base()
        T_s = state_to_T(state)
        d = np.zeros(7, dtype=np.float32)
        d[:3] = T_s[:3, 3] - T_t[:3, 3]
        d[3:6] = _euler_xyz_pytorch3d(T_s[:3, :3] @ T_t[:3, :3].T)   # left-multiplied delta
        g = state[6] if grip_cmd is None else grip_cmd
        d[6] = 1.0 if g > GRIPPER_OPEN_THRESHOLD else -1.0
        return self.env.step(d)
