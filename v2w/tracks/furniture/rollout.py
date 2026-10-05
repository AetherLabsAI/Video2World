"""Execute a package's own action stream in its own scene on the sample's robot rig.

The aligned package carries `actions.npy` in the robot base frame: (T, 7) [x y z roll pitch yaw grip], absolute TCP
poses, R = Rz(yaw) Ry(pitch) Rx(roll); grip 1 = open, 0 = close; row 0 = initial TCP state.
Execution: numerical IK per row (warm-started, joint limits as bounds) at reset and for every row, joint targets
interpolated over --sub control steps; a commanded closure blocks the arm until the fingers latch; a part is held only
after real finger contact (force-limited fingers and pad friction). A release that already satisfies the family's
placement geometry locks the part in place (snap-fit proxy for threads / tight fits). Terminal: settle 1 s, record final
poses, trajectories, grasp state and quiescence.

Usage: python -m v2w.tracks.furniture.rollout <aligned_pkg> --sample <dir> --out rollout.json [--video x.mp4] [--sub 4] [--no-render]
"""
import argparse, json, math, os
import numpy as np
from pathlib import Path

from v2w import paths
from v2w.tracks.furniture import sim
from video2sim.bench.protocol import load_manifest, scene_in_base_frame

PANDA_LIMS = np.array([[-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973], [-3.0718, -0.0698],
                       [-2.8973, 2.8973], [-0.0175, 3.7525], [-2.8973, 2.8973]])
Q_NEUTRAL = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 2.0, 0.785])


def robot_table(uid):
    """Per-robot rig facts.
    urdf / tcp = the IK chain (the same URDF the rig loads); lims / neutral = IK bounds and seed; hand = the link the attach drive
    hangs a held part from; gripper = whether the action vector has a gripper column at all (False: `grip` is ignored, nothing is
    ever grasped, the arm commands are the 7 joints only)."""
    if uid == 'panda_robotiq':   # DROID rig: Panda arm + Robotiq 2F-85, IK to the fingertip centre `eef`
        return dict(uid=uid, urdf=str(sim.PANDA_ROBOTIQ_URDF), tcp='eef', lims=PANDA_LIMS, neutral=Q_NEUTRAL, hand='robotiq_arg2f_base_link', gripper=True)
    if uid == 'xarm7_pusher':
        return dict(uid=uid, urdf=str(sim.XARM7_PUSHER_URDF), tcp='link_tcp', lims=sim.XARM7_JOINT_LIMITS, neutral=sim.XARM7_REST_QPOS.astype(np.float64), hand='link7', gripper=False)
    import mani_skill
    return dict(uid='panda', urdf=os.path.dirname(mani_skill.__file__) + '/assets/robots/panda/panda_v2.urdf', tcp='panda_hand_tcp', lims=PANDA_LIMS, neutral=Q_NEUTRAL, hand='panda_hand', gripper=True)


def grasp_phases(gw, w_open):
    """Closure start kc, hold start kh (width stops decreasing) and release kr from a recorded gripper width."""
    closed = gw < w_open - 0.003; kc = int(np.argmax(closed)); kh = kc
    while kh + 1 < len(gw) and gw[kh] - gw[kh + 1] > 0.001: kh += 1
    w_hold, kr = gw[kh], None
    for k in range(kh + 1, len(gw)):
        w_hold = min(w_hold, gw[k])
        if gw[k] > w_hold + max(0.004, 0.1 * (w_open - w_hold)): kr = k; break
    return kc, kh, (kr if kr is not None else len(gw))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pkg'); ap.add_argument('--sample', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--video', default=None); ap.add_argument('--sub', type=int, default=4)
    ap.add_argument('--no-render', action='store_true')
    ap.add_argument('--squeeze-mm', type=float, default=1.0)
    ap.add_argument('--snap-fit', type=int, default=1, help='thread / tight-fit proxy: when the gripper releases the part and it already satisfies the family placement geometry , lock it in place (kinematic) instead of a free release. 0 = off.')
    ap.add_argument('--attach-on-grasp', type=int, default=1); ap.add_argument('--hold', default='friction', choices=['friction', 'drive'], help='how a grasped part is held: friction (default) = only the latched force-limited fingers and pad friction; drive = the earlier rigid drive hand<-part proxy')
    ap.add_argument('--settle', type=float, default=0.5, help='settle time at reset (s); the GT-as-candidate study uses 0 like the Reference replay: with 0.5 the arm spawned at row 0 pushes a MEASURED part out of the open fingers before the stream starts')
    ap.add_argument('--attach-max-gap-cm', type=float, default=0.0, help='when the gripper is commanded closed and no finger contact is detected, still attach if the part SURFACE is within this distance of the TCP 0 = off (contact required).')
    a = ap.parse_args(); pkg, sample = Path(a.pkg), Path(a.sample)
    m = load_manifest(pkg); scene = scene_in_base_frame(m, pkg)
    act_rel = (m.get('actions') or {}).get('path')
    if not act_rel or not (pkg / act_rel).exists():
        json.dump(dict(ok=False, error='package delivers no actions'), open(a.out, 'w')); print('no actions'); return
    A7 = np.load(pkg / act_rel).astype(np.float64)          # (T,7) [x y z roll pitch yaw grip]
    from scipy.spatial.transform import Rotation as _R
    _q = _R.from_euler('ZYX', A7[:, [5, 4, 3]]).as_quat()   # -> wxyz for the IK below
    A = np.c_[A7[:, :3], _q[:, 3], _q[:, 0], _q[:, 1], _q[:, 2], A7[:, 6]]
    hid = np.load(sample / 'hidden' / 'trajectory.npz', allow_pickle=True)
    cam = json.loads((sample / 'hidden' / 'camera.json').read_text())
    base_pose = [float(x) for x in hid['robot_base_pose']]
    robot = 'xarm7_pusher' if ('robot_uid' in hid.files and str(hid['robot_uid']) == 'xarm7_pusher') else ('panda_robotiq' if ('tcp_offset_m' in hid.files and float(hid['tcp_offset_m']) > 0) else 'panda')   # the rig the SAMPLE was recorded on (the candidate's protocol.json robot is nominal); DROID (tcp_offset_m > 0) = Panda + Robotiq 2F-85 as in replay_fb --robot auto
    ROBOTIQ = robot == 'panda_robotiq'
    if ROBOTIQ:
        # the DROID rig runs `sub` control steps per action row (e.g. 30 Hz control / 300 Hz physics for 7.5 Hz rows):
        # the thin-part friction hold is solver-step sensitive
        dt = float((m.get('actions') or {}).get('dt') or 1.0 / float(hid['fps'])); hz = int(round(a.sub / dt))
        sim.CONTROL_FREQ = hz; sim.SIM_FREQ = math.lcm(100, hz)
    RB = robot_table(robot); HAS_G = RB['gripper']
    mk = (lambda q, g: np.r_[q, g]) if HAS_G else (lambda q, g: np.asarray(q))   # action vector: 7 joints (+ the gripper command when the robot has one)
    if ROBOTIQ:   # Robotiq gripper action = outer-knuckle angle (rad); pad friction 1.0, knuckle torque 6 N m
        sim.PandaRobotiq.urdf_config['_materials']['gripper'].update(static_friction=1.0, dynamic_friction=1.0); sim.PandaRobotiq.gripper_force_limit = 6.0
    solve = make_ik(base_pose, robot)

    # IK every 5 Hz target up front (warm-started); report the reachability of the STREAM itself
    q = None; targets = []; ik_err = []
    for i in range(len(A)):
        q, ep, er = solve(A[i, :3], A[i, 3:7], q)
        targets.append(q); ik_err.append((ep, er))
    targets = np.asarray(targets); ik_err = np.asarray(ik_err)
    grip_open = (A[:, 7] > 0.5) if HAS_G else np.ones(len(A), bool)   # no gripper: the `grip` column is ignored (nothing can be held)

    import gymnasium as gym, torch
    import mani_skill.envs  # noqa
    env = gym.make('V2SPanda-v1', obs_mode='rgb' if not a.no_render else 'state', num_envs=1, sim_backend='physx_cpu',
                   render_backend=os.environ.get('V2S_RENDER_BACKEND', 'gpu') if not a.no_render else 'none', scene_spec=scene,
                   camera=None if a.no_render else cam, enable_cameras=not a.no_render,
                   robot_base_pose=base_pose, robot_uid=robot)
    LO, HI = -0.01, 0.04
    g0 = HI          # fingers start OPEN; a stream that starts closed gets a latched pregrasp closing (below)
    rep = dict(sample=sample.name, package=str(pkg), sub=a.sub, T=int(len(A)), mode='own_actions', robot=robot, hold=a.hold,
               ik_pos_err_max_cm=round(float(ik_err[:, 0].max()) * 100, 2),
               ik_rot_err_max_deg=round(float(np.degrees(ik_err[:, 1].max())), 2))
    try:
        env.reset(seed=0, options=dict(qpos=(np.r_[targets[0], g0, g0] if HAS_G else targets[0]).astype(np.float32), settle=(0.0 if not grip_open[0] else a.settle))); u = env.unwrapped   # a stream that STARTS held: no settle before the pregrasp closing, or the part placed between the open fingers falls out before they close
        names = list(u.objs); rep['objects'] = names
        snap_ctx = None
        if a.snap_fit and names:
            try:
                from v2w.metrics.task import context, snap_decision
                snap_ctx = context(sample, pkg, None); snap_phi = json.loads((sample / 'hidden' / 'phi.json').read_text()); snap_name = snap_ctx['cname'] if snap_ctx['cname'] in names else names[0]
            except Exception as ex:
                rep['snap_error'] = str(ex); snap_ctx = None
        rep['init_poses'] = {n: u.obj_pose_np(n).tolist() for n in names}
        traj = {n: [u.obj_pose_np(n)] for n in names}; ee = [u.tcp_pose_np()]; frames = {}
        if not a.no_render: frames[0] = u.get_obs()['sensor_data']['eval_camera']['rgb'][0].cpu().numpy().astype(np.uint8)
        import sapien
        sub_sc = u.scene.sub_scenes[0]
        hand = next(l for l in u.agent.robot.links if l.name == RB['hand'])._objs[0]
        drive = [None]; held = [False]; ri = 0

        def attach(n):
            if held[0]: return
            held[0] = True; rep.setdefault('attach_events', []).append(int(ri))
            if a.hold != 'drive': return   # friction hold: the latched, force-limited fingers and pad friction carry the part; no rigid drive
            ent = u.objs[n]._objs[0]
            rel = hand.pose.inv() * ent.pose
            d_ = sub_sc.create_drive(hand, rel, ent, sapien.Pose())
            for f_ in ('set_drive_property_x', 'set_drive_property_y', 'set_drive_property_z',
                       'set_drive_property_twist', 'set_drive_property_swing', 'set_drive_property_slerp'):
                try: getattr(d_, f_)(1e6, 1e4, 1e6)
                except Exception: getattr(d_, f_)(1e6, 1e4)
            drive[0] = d_

        def release():
            if not held[0]: return
            held[0] = False
            snap = False
            if snap_ctx is not None and not rep.get('snap_event'):
                try:
                    snap, det = snap_decision(snap_ctx, snap_phi, u.obj_pose_np(snap_name), snap_name); rep['snap_check'] = dict(frame=int(ri), **det)
                except Exception as ex:
                    rep['snap_error'] = str(ex); snap = False
            if drive[0] is not None:
                try: drive[0].disable()
                except Exception: pass
                try: drive[0].entity.remove_component(drive[0])
                except Exception: pass
                drive[0] = None
            rep.setdefault('release_events', []).append(int(ri))
            if snap:   # thread / tight-fit proxy: the part stays where it was released
                comp = u.objs[snap_name]._objs[0].find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
                comp.set_kinematic(True); rep['snap_event'] = dict(rep['snap_check'], pose=[float(x) for x in u.obj_pose_np(snap_name)])

        hold_j = None
        if ROBOTIQ:
            # ---- Robotiq contact grasp: the pads close kinematically at
            # the nominal 2F-85 rate until the first touch, then the force-limited PD pads press; a grasp = both pads >= 0.5 N for 3
            # control steps AND a knuckle stalled > 0.03 rad behind its command (or fully closed); while held the command stays
            # 0.06 rad past the stall (friction hold, no drive); an open command opens at the same rate and releases.
            import torch as _T
            _clock = sim
            RQ = dict(th=0.0, contact=False, touching=0, th_att=None); TH_MAX = 0.79; HOLD_SQ = 0.06; JN = [j.name for j in u.agent.robot.active_joints][7:]
            # closing rate = the rig's demonstrated closure (recorded width over the closing frames), never below 0.02 rad / control step
            _gw = np.asarray(hid['gripper_width']).ravel(); _wo = float(np.median(np.sort(_gw)[-max(3, len(_gw) // 10):])); _kc, _kh, _ = grasp_phases(_gw, _wo)
            DTH = max((u.gripper_cmd(float(_gw[_kh])) - u.gripper_cmd(float(_gw[_kc]))) / max((_kh - _kc) * a.sub, 1), 0.02); rep['robotiq_closure'] = dict(rate_rad_per_step=round(DTH, 4), control_hz=int(_clock.CONTROL_FREQ), recorded_close_frames=[int(_kc), int(_kh)])
            def rq_step(qcmd, closing):
                if closing: th = min((RQ['th_att'] if RQ['th_att'] is not None else RQ['th']) + HOLD_SQ, TH_MAX) if held[0] else min(RQ['th'] + DTH, TH_MAX)
                else: th = max(RQ['th'] - DTH, 0.0)
                RQ['th'] = th
                if closing and not RQ['contact']:
                    qn = u.qpos_np(); qn[7:] = sim.gripper_qpos(th, JN); u.agent.robot.set_qpos(_T.tensor(qn[None], dtype=_T.float32)); u.agent.robot.set_qvel(_T.zeros(1, len(qn)))
                env.step(_T.tensor(np.r_[qcmd, th].astype(np.float32)[None], dtype=_T.float32))
                if closing and not held[0] and names:
                    for n_ in names:
                        lf = u.scene.get_pairwise_contact_forces(u.agent.finger1_link, u.objs[n_]); rf = u.scene.get_pairwise_contact_forces(u.agent.finger2_link, u.objs[n_]); fl, fr = float(_T.linalg.norm(lf)), float(_T.linalg.norm(rf))
                        if fl >= 0.2 or fr >= 0.2: RQ['contact'] = True
                        if fl >= 0.5 and fr >= 0.5:
                            RQ['touching'] += 1; th_meas = float(u.qpos_np()[7]); stalled = RQ['contact'] and (th - th_meas > 0.03)
                            if RQ['touching'] >= 3 and (stalled or th >= TH_MAX - 1e-6): RQ['th_att'] = th_meas; attach(n_); rep['attach_forces_N'] = [round(fl, 2), round(fr, 2)]
                            break
                    else: RQ['touching'] = 0
                if not closing:
                    RQ['contact'] = False; RQ['touching'] = 0; RQ['th_att'] = None
                    if held[0]: release()
        PQ_gap = [None]   # proximity query on the manipulated part's collision hull (for --attach-max-gap-cm and the gap diagnostics)
        if names:
            try:
                import trimesh as _tm
                _e0 = m['scene']['objects'][0]; _mp = pkg / (_e0.get('collision_path') or _e0.get('mesh_path') or '')
                if _mp.is_file():
                    _sc = _tm.load(_mp, process=False); _hull = _tm.util.concatenate(list(_sc.geometry.values())) if hasattr(_sc, 'geometry') else _sc
                    PQ_gap[0] = _tm.proximity.ProximityQuery(_hull)
                elif _e0.get('kind') in ('box', 'cylinder', 'sphere'):   # primitive part
                    _hs = np.asarray(_e0.get('half_size', [0.03] * 3), float)
                    _prim = _tm.creation.box(extents=2 * _hs) if _e0['kind'] == 'box' else (_tm.creation.cylinder(radius=float(_hs[0]), height=2 * float(_hs[2])) if _e0['kind'] == 'cylinder' else _tm.creation.icosphere(radius=float(_hs[0])))
                    PQ_gap[0] = _tm.proximity.ProximityQuery(_prim)
            except Exception as ex: rep['gap_query_error'] = str(ex)
        def surface_gap_cm(n):
            if PQ_gap[0] is None: return None
            from scipy.spatial.transform import Rotation as _Rg
            op = u.obj_pose_np(n); tp = u.tcp_pose_np(); Ro = _Rg.from_quat([op[4], op[5], op[6], op[3]]).as_matrix()
            p_obj = Ro.T @ (tp[:3] - op[:3]); near = PQ_gap[0].on_surface(p_obj[None])[0][0]
            return float(np.linalg.norm(p_obj - near)) * 100
        def lenient_attach(k_):
            # no finger contact at a commanded closure: attach anyway if the surface is within --attach-max-gap-cm of the TCP
            g = surface_gap_cm(names[0]) if names else None
            rep.setdefault('close_gap_cm', []).append(dict(row=int(k_), gap_cm=None if g is None else round(g, 2)))
            if a.attach_max_gap_cm > 0 and drive[0] is None and g is not None and g <= a.attach_max_gap_cm:
                attach(names[0]); rep.setdefault('attach_lenient_gap', []).append(dict(row=int(k_), gap_cm=round(g, 2))); return True
            return False
        if not grip_open[0]:
            # the stream begins already holding: bounded surface snap (the package's
            # object pose and its row-0 TCP are two estimates; PhysX needs actual contact or the part is left behind)
            if names:
                try:
                    from scipy.spatial.transform import Rotation as _Rr
                    n0 = names[0]; e0 = m['scene']['objects'][0]
                    PQ = PQ_gap[0]   # mesh hull or primitive
                    if PQ is not None:
                        op = u.obj_pose_np(n0); tp = u.tcp_pose_np()
                        Ro = _Rr.from_quat([op[4], op[5], op[6], op[3]]).as_matrix()
                        p_obj = Ro.T @ (tp[:3] - op[:3])
                        near = PQ.on_surface(p_obj[None])[0][0]
                        delta = Ro @ (p_obj - near)
                        d = float(np.linalg.norm(delta))
                        rep['tcp_surface_gap_cm'] = round(d * 100, 2)
                        if d > 0.03: delta = delta / d * 0.03
                        from mani_skill.utils.structs.pose import Pose as _Pose
                        u.objs[n0].set_pose(_Pose.create_from_pq((op[:3] + delta).astype(np.float32), op[3:].astype(np.float32)))
                        rep['grasp_snap_cm'] = round(float(np.linalg.norm(delta)) * 100, 2)
                except Exception as ex:
                    rep['grasp_snap_error'] = str(ex)
            # physically close on the part at row 0's pose (contact-latched) before the first move; attach only if contact actually happened
            for _i in range(int(1.5 * float(_clock.CONTROL_FREQ)) if ROBOTIQ else 40):        # 2 s at 20 Hz: the position-driven fingers need ~1.5 s for the full 4 cm travel
                if ROBOTIQ:
                    rq_step(targets[0], True)
                    if held[0] and _i > 3: break
                    continue
                cmd = mk(targets[0], -1.0).astype(np.float32)
                if hold_j is None and any(u.is_grasping_np(n) for n in names):
                    hold_j = max(float(u.qpos_np()[7]) - a.squeeze_mm / 1000.0, LO)
                if hold_j is not None:
                    cmd[7] = np.float32((hold_j - LO) / (HI - LO) * 2 - 1)
                env.step(torch.tensor(cmd[None], dtype=torch.float32))
            rep['pregrasp'] = dict(grasping={n: bool(u.is_grasping_np(n)) for n in names},
                                   finger=[float(x) for x in u.qpos_np()[7:]])
            if a.attach_on_grasp:
                g_ = [n for n in names if u.is_grasping_np(n)]
                if g_: attach(g_[0])
                elif names and rep.get('tcp_surface_gap_cm') is not None and rep['tcp_surface_gap_cm'] <= 3.5:
                    # start-held leniency: the stream
                    # DECLARES the part held at row 0; if the TCP is actually at the part (snap measured the gap
                    # to its surface, bounded 3 cm) the hold is granted even when the position-driven fingers
                    # cannot pass the antipodal-contact test (smooth ball, side pinch). Mid-stream grasps still
                    # require real contact — those are entirely the candidate's plan.
                    rep['attach_lenient_start'] = True
                    attach(names[0])
        for k in range(1, len(A)):
            q0, q1 = targets[k - 1], targets[k]
            if grip_open[k - 1] and not grip_open[k]:
                # close onset: the real Franka grasp() BLOCKS — hold the arm while the position-driven fingers
                # travel (~1.5 s for 4 cm), latch on first contact; without this the 5 Hz stream walks away
                # before the fingers arrive and every mid-stream grasp is lost
                if ROBOTIQ:   # move to the row where the close command starts (the grasp pose), pads already closing, then hold the tool still and wait for the stall
                    for j in range(1, a.sub + 1): rq_step(q0 + (q1 - q0) * (j / a.sub), True)
                for _i in range(int(1.2 * float(_clock.CONTROL_FREQ)) if ROBOTIQ else 30):
                    if ROBOTIQ:
                        rq_step(q1, True)
                        if held[0]: break
                        continue
                    cmd = mk(q0, -1.0).astype(np.float32)
                    if hold_j is None and any(u.is_grasping_np(n) for n in names):
                        hold_j = max(float(u.qpos_np()[7]) - a.squeeze_mm / 1000.0, LO)
                    if hold_j is not None:
                        cmd[7] = np.float32((hold_j - LO) / (HI - LO) * 2 - 1)
                    env.step(torch.tensor(cmd[None], dtype=torch.float32))
                    if hold_j is not None: break
                rep.setdefault('grasp_waits', []).append(dict(row=int(k), latched=(held[0] if ROBOTIQ else hold_j is not None)))
            for j in range(1, a.sub + 1):
                if ROBOTIQ:
                    rq_step(q0 + (q1 - q0) * (j / a.sub), not grip_open[k]); continue
                cmd = mk(q0 + (q1 - q0) * (j / a.sub), 1.0 if grip_open[k] else -1.0).astype(np.float32)
                if HAS_G and cmd[7] < 0:
                    if hold_j is None and any(u.is_grasping_np(n) for n in names):
                        hold_j = max(float(u.qpos_np()[7]) - a.squeeze_mm / 1000.0, LO)
                    if hold_j is not None:
                        cmd[7] = np.float32((hold_j - LO) / (HI - LO) * 2 - 1)
                else:
                    hold_j = None
                env.step(torch.tensor(cmd[None], dtype=torch.float32))
            ri = k
            if a.attach_on_grasp and names:
                # contact-verified latch: attach only while COMMANDED closed and actually grasping
                if not ROBOTIQ and not grip_open[k] and not held[0] and any(u.is_grasping_np(n) for n in names):
                    attach(next(n for n in names if u.is_grasping_np(n)))
                elif not grip_open[k] and not held[0] and grip_open[k - 1]:   # closure onset without contact: measure the gap (and attach if within --attach-max-gap-cm)
                    lenient_attach(k)
                elif grip_open[k]:
                    release()
            for n in names: traj[n].append(u.obj_pose_np(n))
            ee.append(u.tcp_pose_np())
            if not a.no_render and (a.video or k in (len(A) // 2, len(A) - 1)):
                frames[k] = u.get_obs()['sensor_data']['eval_camera']['rgb'][0].cpu().numpy().astype(np.uint8)
        u._settle(1.0)
        rep['final_poses'] = {n: u.obj_pose_np(n).tolist() for n in names}
        rep['traj'] = {n: np.stack(v).tolist() for n, v in traj.items()}
        rep['ee'] = np.stack(ee).tolist()
        # a part still held by the attach drive is HELD even though the
        # position-controlled fingers report no contact; phi2's `released` test reads this flag.
        rep['grasping_final'] = {n: bool(u.is_grasping_np(n)) or (n == names[0] and drive[0] is not None) for n in names}
        vel = {}
        for n in names:
            o = u.objs[n]
            vel[n] = float(np.linalg.norm(o.linear_velocity.cpu().numpy())
                           + 0.1 * np.linalg.norm(o.angular_velocity.cpu().numpy()))
        rep['quiescent'] = bool(max(vel.values(), default=0.0) < 5e-3)
        rep['ok'] = True
        if frames:
            import imageio.v2 as iio
            if a.video:
                seq = [frames[k] for k in sorted(frames)]
                Path(a.video).parent.mkdir(parents=True, exist_ok=True)
                iio.mimwrite(a.video, seq, fps=float(hid['fps']), macro_block_size=1)
                ff = paths.tool('ffmpeg', required=False)
                if ff and Path(ff).exists():
                    import subprocess, shutil
                    tmp = str(Path(a.video).with_suffix('.h264.mp4'))
                    if subprocess.run([ff, '-y', '-loglevel', 'error', '-i', a.video, '-c:v', 'libx264',
                                       '-pix_fmt', 'yuv420p', '-movflags', '+faststart', tmp]).returncode == 0:
                        shutil.move(tmp, a.video)
    finally:
        env.close()
    json.dump(rep, open(a.out, 'w'), indent=1)
    tr = np.asarray(rep['traj'][rep['objects'][0]]) if rep.get('traj') else None
    print(json.dumps(dict(objects=rep.get('objects'), T=rep['T'], ik_pos_err_max_cm=rep['ik_pos_err_max_cm'],
                          z_gain=None if tr is None else round(float(tr[:, 2].max() - tr[0, 2]), 3),
                          attach=len(rep.get('attach_events', [])), quiescent=rep.get('quiescent'))))


if __name__ == '__main__':
    main()
