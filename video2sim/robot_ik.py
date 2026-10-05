from pathlib import Path
import numpy as np
PANDA_LIMS=np.array([[-2.8973,2.8973],[-1.7628,1.7628],[-2.8973,2.8973],[-3.0718,-.0698],[-2.8973,2.8973],[-.0175,3.7525],[-2.8973,2.8973]])
Q_NEUTRAL=np.array([0.,-.4,0.,-2.2,0.,2.,.785])

def robot_table(uid):
    """Per-robot rig facts for the candidate chain (2026-09-12: second robot, the reconstructed-twins Push-T xArm7 + pusher rod).
    urdf / tcp = the IK chain (the same URDF the rig loads); lims / neutral = IK bounds and seed; hand = the link the attach drive
    hangs a held part from; gripper = whether the action vector has a gripper column at all (False: `grip` is ignored, nothing is
    ever grasped, the arm commands are the 7 joints only)."""
    if uid == 'panda_robotiq':   # DROID rig (2026-09-13, unified executors): Panda arm + Robotiq 2F-85, IK to the fingertip centre `eef`, the same URDF ref_replay / replay_fb load
        return dict(uid=uid, urdf=str(Path(__file__).parent/'robot_rigs/assets/panda_robotiq.urdf'), tcp='eef', lims=PANDA_LIMS, neutral=Q_NEUTRAL, hand='robotiq_arg2f_base_link', gripper=True)
    if uid == 'xarm7_pusher':
        from .robot_rigs import xarm7_pusher as X
        return dict(uid=uid, urdf=str(X.URDF), tcp='link_tcp', lims=X.JOINT_LIMITS, neutral=X.REST_QPOS.astype(np.float64), hand='link7', gripper=False)
    import os, mani_skill
    return dict(uid='panda', urdf=os.path.dirname(mani_skill.__file__) + '/assets/robots/panda/panda_v2.urdf', tcp='panda_hand_tcp', lims=PANDA_LIMS, neutral=Q_NEUTRAL, hand='panda_hand', gripper=True)

def make_ik(base_pose, robot='panda'):
    """IK against the same URDF the rig loads, in the same base frame the actions arrive in."""
    import pytorch_kinematics as pk, torch
    from scipy.spatial.transform import Rotation as R
    from scipy.optimize import least_squares
    RB = robot_table(robot); urdf = RB['urdf']; PANDA_LIMS = RB['lims']; Q_NEUTRAL = RB['neutral']   # names kept: the solver below is robot-agnostic
    chain = pk.build_serial_chain_from_urdf(open(urdf, 'rb').read(), RB['tcp']).to(dtype=torch.float64)
    bp, bq = np.asarray(base_pose[:3], float), np.asarray(base_pose[3:7], float)
    Rb = R.from_quat([bq[1], bq[2], bq[3], bq[0]]).as_matrix()

    def fk(q):
        t = chain.forward_kinematics(torch.as_tensor(q, dtype=torch.float64)[None])
        M = t.get_matrix()[0].numpy()
        return Rb @ M[:3, 3] + bp, Rb @ M[:3, :3]

    def solve(pos, quat, seed):
        Rt = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()

        s0 = seed if seed is not None else Q_NEUTRAL

        def res(q):
            p, Mr = fk(q)
            # the joint-regularization term picks ONE of the 7-DoF branch solutions deterministically
            # (closest to the seed); its weight is small enough not to disturb the TCP fit
            return np.r_[(p - pos) * 10.0, R.from_matrix(Mr.T @ Rt).as_rotvec(), (q - s0) * 0.02]

        best = None
        for s in ([seed] if seed is not None else []) + [Q_NEUTRAL]:
            r = least_squares(res, np.clip(s, PANDA_LIMS[:, 0] + 1e-3, PANDA_LIMS[:, 1] - 1e-3),
                              bounds=(PANDA_LIMS[:, 0], PANDA_LIMS[:, 1]), xtol=1e-10, max_nfev=200)
            if best is None or r.cost < best.cost: best = r
            p_, M_ = fk(r.x)
            if np.linalg.norm(p_ - pos) < 1e-3 and np.linalg.norm(R.from_matrix(M_.T @ Rt).as_rotvec()) < 0.01:
                best = r; break
        p, Mr = fk(best.x)
        return best.x, float(np.linalg.norm(p - pos)), float(np.linalg.norm(R.from_matrix(Mr.T @ Rt).as_rotvec()))

    return solve