"""Execute a hand-track package's own action stream with contact dynamics (hand-contact-physics/1.3).

Only robot joint targets are controlled: a Franka arm + gripper (TCP xyz/rpy/open rows, IK per row), one Wuji hand (palm
xyz/rpy rows + 20 finger targets) or two Wuji hands on one clock. Objects and passive hinges are never attached,
repositioned or driven; all transport depends on the simulator's contact and friction. attach/release events are
observed contact transitions.

Usage: python -m v2w.tracks.hand.rollout PKG --sample SAMPLE_DIR --out rollout.json --embodiment arm|dexhand|dexhand_bimanual
"""
from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from video2sim.bench.protocol import load_manifest, scene_in_base_frame

from .actions import BIMANUAL_VERSION, SIDES, actions_to_quat, load_bimanual, root6_stream
from .physics import VERSION, MotionAudit, assess, configure_robot, self_collision, terminal_motion

SIM_FREQ = 300
PANDA_FINGER_OPEN, PANDA_FINGER_LO = 0.04, 0.0
FLEX_IDX = {1: [0, 1, 2, 3], 2: [0, 2, 3], 3: [0, 2, 3], 4: [0, 2, 3], 5: [0, 2, 3]}   # per-finger flexion joints (index within the finger's 4)
FINGER_RATE = 6.0 / SIM_FREQ           # rad per physics substep (a human finger closes 1.5 rad in ~0.25 s)
CONTACT_FORCE_N = 1e-3


def use_software_renderer():
    """Let SAPIEN fall back to its default render device when ManiSkill's device string does not map to one (lavapipe)."""
    import sapien
    _RS = sapien.render.RenderSystem
    if getattr(_RS, '_v2s_patched', False):
        return
    def RenderSystem(device=None):
        try:
            return _RS(device)
        except Exception:
            return _RS()
    RenderSystem._v2s_patched = True
    sapien.render.RenderSystem = RenderSystem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pkg'); ap.add_argument('--sample', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--embodiment', required=True, choices=['arm', 'dexhand', 'dexhand_bimanual'])
    ap.add_argument('--video', default=None); ap.add_argument('--sub', type=int, default=0, help='physics substeps per action row (0 = dt * 300 Hz)')
    ap.add_argument('--settle', type=float, default=1.0)
    ap.add_argument('--object-match', default=None, help='JSON {gt_name: candidate_name}')
    ap.add_argument('--no-render', action='store_true')
    ap.add_argument('--full-video', default=None, help='additional 30 fps video including prelude and settle; requires --audit-dir')
    ap.add_argument('--audit-dir', default=None, help='read-only full physics-clock state/contact archive')
    ap.add_argument('--base-pose', default=None, help='arm: 7-vector JSON overriding hidden/robot_base.json')
    ap.add_argument('--side', default=None, help='dexhand: right|left (default: the demonstrator hand side in hidden/trajectory.npz)')
    a = ap.parse_args()
    if a.full_video and a.no_render: ap.error('--full-video requires rendering')
    if not a.audit_dir: a.audit_dir=str(Path(a.out).with_suffix(''))+'_audit'
    if a.embodiment == 'dexhand_bimanual':
        execute_bimanual(a)
        return
    pkg, sample = Path(a.pkg).resolve(), Path(a.sample).resolve()
    m = load_manifest(pkg); scene = scene_in_base_frame(m, pkg)
    cam = json.loads((sample / 'hidden' / 'camera.json').read_text()); phi = json.loads((sample / 'hidden' / 'phi.json').read_text())
    hid = np.load(sample / 'hidden' / 'trajectory.npz', allow_pickle=True)
    act_spec = m.get('actions') or {}
    if not act_spec.get('path') or not (pkg / act_spec['path']).exists():
        json.dump(dict(ok=False, error='package delivers no actions', sample=sample.name, embodiment=a.embodiment), open(a.out, 'w')); print('no actions'); return
    acts_raw = np.load(pkg / act_spec['path']).astype(np.float64)
    acts, fmt = actions_to_quat(acts_raw, act_spec.get('format'))
    dt = float(act_spec.get('dt') or (1.0 / 30.0)); sub = a.sub or max(1, int(round(dt * SIM_FREQ)))
    ext = np.asarray(cam['extrinsics_base_cam_seq']); T_video = len(ext)
    g2c = {k: v for k, v in (json.loads(a.object_match) if a.object_match else {}).items() if v}
    rep = dict(sample=sample.name, package=str(pkg), N=int(len(acts)), sub=int(sub), dt=dt, actions_format=fmt, embodiment=a.embodiment, build=True, ok=False, physics_protocol=VERSION, hold_mode='friction', object_attachment=False, object_state_override=False, contact_force_threshold_N=CONTACT_FORCE_N, event_semantics='observed contact transitions; no attachment', prelude_s=0.0 if a.embodiment == 'arm' else 1.2, settle_s=float(a.settle), video_includes_prelude=False, video_includes_settle=False)

    rep['adaptive_wait_s'] = 0.

    # ---- embodiment-specific preparation
    if a.embodiment == 'arm':
        from v2w.tracks.furniture.rollout import make_ik
        bp = json.loads(a.base_pose) if a.base_pose else json.loads((sample / 'hidden' / 'robot_base.json').read_text())['base_pose']
        bp = [float(x) for x in bp]; rep['robot'] = 'panda'; rep['base_pose'] = bp
        solve = make_ik(bp, 'panda'); q = None; targets = []; ik_err = []
        for i in range(len(acts)):
            q, ep, er = solve(acts[i, :3], acts[i, 3:7], q); targets.append(q); ik_err.append((ep, er))
        targets = np.asarray(targets); ik_err = np.asarray(ik_err)
        rep['ik_pos_err_max_cm'] = round(float(ik_err[:, 0].max()) * 100, 2); rep['ik_rot_err_max_deg'] = round(float(np.degrees(ik_err[:, 1].max())), 2)
        rep['ik_pos_err_cm'] = [round(float(v) * 100, 2) for v in ik_err[:, 0]]
        robot_uid, base_pose = 'panda', bp
        fj = None
    else:
        side = a.side or str(hid['hand_side']) if 'hand_side' in hid.files else 'right'
        side = a.side or side
        if side not in ('left', 'right'): side = 'right'
        robot_uid, base_pose = f'wuji_{side}_floating', None; rep['robot'] = robot_uid; rep['side'] = side
        fj_rel = act_spec.get('finger_joints_path', 'finger_joints.npy')
        if not (pkg / fj_rel).exists():
            rep['error'] = f'dexhand package delivers no finger joint stream ({fj_rel})'; json.dump(rep, open(a.out, 'w')); print(rep['error']); return
        fj = np.load(pkg / fj_rel).astype(np.float64)
        if fj.shape != (len(acts), 20):
            rep['error'] = f'finger_joints must be (T,20) matching actions (T={len(acts)}), got {fj.shape}'; json.dump(rep, open(a.out, 'w')); print(rep['error']); return
        root6 = root6_stream(acts)

    import gymnasium as gym, torch, mani_skill.envs  # noqa: F401
    from . import env as _env  # noqa: F401  registers V2SEgoHand-v1
    if not a.no_render: use_software_renderer()
    env = gym.make('V2SEgoHand-v1', obs_mode='none', num_envs=1, sim_backend='physx_cpu', render_backend='sapien_cpu' if not a.no_render else 'none', scene_spec=scene,
                   camera=None if a.no_render else cam, enable_cameras=not a.no_render, robot_uid=robot_uid, robot_base_pose=base_pose)
    u = env.unwrapped; ag = u.agent
    try:
        if a.embodiment == 'arm':
            q0 = np.r_[targets[0], PANDA_FINGER_OPEN, PANDA_FINGER_OPEN].astype(np.float32)
            hand_link = 'panda_hand'; finger_links = [next(l for l in ag.robot.links if l.name == n) for n in ('panda_leftfinger', 'panda_rightfinger')]
            fj_lo = fj_hi = None
        else:
            if 'ik_pos_err_max_cm' not in rep: rep['ik_pos_err_max_cm'] = 0.0
            from .robots import open_fingers
            q_open0 = np.asarray(open_fingers(side), float)
            initial_fingers = act_spec.get('initial_finger_joints_path')
            if initial_fingers:
                q_open0 = np.load(pkg / initial_fingers, allow_pickle=False).astype(float)
                if q_open0.shape != (20,) or not np.isfinite(q_open0).all():
                    raise ValueError('initial_finger_joints must be a finite 20-vector')
                rep['initial_finger_policy'] = 'explicit package robot initial state'
            else:
                rep['initial_finger_policy'] = 'legacy pinch open'
            # spawn OPEN and 8 cm BEHIND the row-0 palm pose along the palm's finger axis (a stream that starts held would otherwise spawn the open
            # fingers / folded fist on top of the part and knock it away), then approach over the prelude and close
            R0 = R.from_quat([acts[0, 4], acts[0, 5], acts[0, 6], acts[0, 3]]).as_matrix()
            offset = np.asarray(act_spec.get('prelude_offset_in_palm', [0., 0., -.08]), float)
            if offset.shape != (3,) or not np.isfinite(offset).all() or np.linalg.norm(offset) > .25:
                raise ValueError('prelude offset must be a finite palm-frame 3-vector within 25 cm')
            back0 = root6[0].copy(); back0[:3] += R0 @ offset
            rep['prelude_offset_in_palm'] = offset.tolist()
            q0 = ag.contract_to_qpos(back0, q_open0).astype(np.float32)
            hand_link = ag.ee_link_name
            lo, hi = ag.robot.get_qlimits()[0].cpu().numpy().T
            fj_lo = np.array([lo[i] for i in ag._contract_idx]); fj_hi = np.array([hi[i] for i in ag._contract_idx])
            fj = np.clip(fj, fj_lo + 1e-4, fj_hi - 1e-4); rep['finger_joints_clipped'] = True
            if np.any(q_open0 < fj_lo) or np.any(q_open0 > fj_hi): raise ValueError('initial finger pose outside joint limits')
        env.reset(seed=0, options=dict(settle=0.0, qpos=q0))
        rep['robot_contract']=configure_robot(ag)
        rep['robot_urdf_sha256']=rep['robot_contract']['urdf_sha256']
        rep['robot_urdf_path']=rep['robot_contract']['urdf_path']
        rep['joint_names']=[j['name'] for j in rep['robot_contract']['joints']]
        rep['joint_velocity_limits']=[j['max_velocity'] for j in rep['robot_contract']['joints']]
        audit = None; full_frames=[]
        if a.full_video:
            u.set_camera_pose(ext[0]);full_frames.append(u.render_rgb())
        if a.audit_dir:
            audit = MotionAudit(u)
            original_step = u.step_physics
            def audited_step(n=1):
                for _ in range(n):
                    original_step(1)
                    audit.step()
                    if a.full_video and audit.steps % 10 == 0:
                        action_time=max(0.,audit.steps/SIM_FREQ-rep['prelude_s'])
                        k=min(len(ext)-1,int(round(action_time/dt)))
                        u.set_camera_pose(ext[k]);full_frames.append(u.render_rgb())
            u.step_physics = audited_step
        names = list(u.objs)
        target = (g2c.get(phi['source']) if g2c.get(phi['source']) in names else None) or (phi['source'] if phi['source'] in names else (names[0] if names else None))
        audit.primary_target=target
        if target in names: names = [target] + [n for n in names if n != target]
        rep['objects'] = names; rep['target'] = target; rep['object_match'] = g2c
        rep['init_poses'] = {n: u.obj_pose_np(n).tolist() for n in names}
        joint_q_rec = {n: [] for n in names if n in getattr(u, 'joints', {})}
        held = [False]
        rep['attach_events'], rep['release_events'], rep['grasp_misses'] = [], [], 0
        is_art = target in getattr(u, 'arts', {})
        tgt_actor = u.objs.get(target)

        def link_touch(link):
            """Actual target contact force, also for articulation links; no proximity fallback."""
            if tgt_actor is None: return False
            try:
                return float(torch.linalg.norm(u.scene.get_pairwise_contact_forces(link, tgt_actor))) > CONTACT_FORCE_N
            except Exception as ex:
                raise RuntimeError('target contact-force query failed') from ex

        # Preserve the existing passive hinge damping, but never actuate the door.
        if is_art:
            u.arts[target].active_joints[0].set_drive_properties(stiffness=0.0, damping=2.0, force_limit=1e5)

        def observe_grasp(k, touching, note=None):
            """Record current contacts. A lost contact immediately ends the observation."""
            if touching and not held[0]:
                rep['attach_events'].append(dict(frame=int(k), **(note or {})))
            elif held[0] and not touching:
                rep['release_events'].append(int(k))
            held[0] = bool(touching)

        def cam_index(k): return int(round(k / max(len(acts) - 1, 1) * (T_video - 1)))
        frames = []; rec = {n: [u.obj_pose_np(n)] for n in names}; ee = []
        def snapshot(k):
            for n in names: rec[n].append(u.obj_pose_np(n))
            for n in joint_q_rec: joint_q_rec[n].append(u.joint_q_now(n))
            ee.append(u.robot_link_pose_np(hand_link).tolist())
            if not a.no_render: u.set_camera_pose(ext[cam_index(k)]); frames.append(u.render_rgb())
        if not a.no_render: u.set_camera_pose(ext[0]); frames.append(u.render_rgb())
        ee.append(u.robot_link_pose_np(hand_link).tolist())

        if a.embodiment == 'arm':
            if audit: audit.mark('actions')
            # ---------------- Franka arm + gripper: joint targets from IK, fingers position-driven with a contact latch
            grip_open = acts[:, 7] <= 0.5; hold_j = [None]; rep['close_stop_width_cm'] = []
            def grasping():
                return all(link_touch(link) for link in finger_links)
            def step_arm(qa, fopen, qv=None):
                fq = PANDA_FINGER_OPEN if fopen else PANDA_FINGER_LO
                if not fopen:
                    if hold_j[0] is not None and not grasping(): hold_j[0] = None
                    if hold_j[0] is None and grasping():
                        hold_j[0] = max(float(u.robot_qpos_np()[7]) - 0.001, PANDA_FINGER_LO); rep['close_stop_width_cm'].append(round(200 * hold_j[0], 2))
                    if hold_j[0] is not None: fq = hold_j[0]
                else: hold_j[0] = None
                u.set_robot_targets(np.r_[qa, fq, fq], None if qv is None else np.r_[qv, 0.0, 0.0])
                u.step_physics(1)
            def close_wait(qa, k):
                """closure onset: hold the arm while the fingers travel (<= 1.5 s), latch on first contact"""
                for _ in range(int(1.5 * SIM_FREQ)):
                    step_arm(qa, False)
                    rep['adaptive_wait_s'] += 1 / SIM_FREQ
                    if hold_j[0] is not None: break
                rep.setdefault('grasp_waits', []).append(dict(row=int(k), latched=hold_j[0] is not None, width_cm=round(200 * float(u.robot_qpos_np()[7]), 2)))
            if not grip_open[0]: close_wait(targets[0], 0)
            observe_grasp(0, grasping())
            for k in range(len(acts) - 1):
                q_a, q_b = targets[k], targets[k + 1]
                if grip_open[k] and not grip_open[k + 1]: close_wait(q_a, k + 1)
                for j in range(1, sub + 1):
                    t = j / sub
                    step_arm(q_a + (q_b - q_a) * t, grip_open[k + 1], (q_b - q_a) / dt)
                touching = grasping()
                observe_grasp(k + 1, touching)
                if not grip_open[k + 1] and not touching: rep['grasp_misses'] += 1
                snapshot(k + 1)
        else:
            # ---------------- Wuji hand: root joints track the palm rows, fingers track bounded commands through contact
            q_now = np.array(q_open0, float); rep['finger_control'] = 'rate-limited-direct/1.0'
            def flex(qf, n): return float(sum(qf[4 * (n - 1) + i] for i in FLEX_IDX[n]))
            finger_contact_links = ag.finger_links
            if any(not links for links in finger_contact_links.values()):
                raise RuntimeError('Wuji contact link mapping incomplete')
            def finger_touch(n):
                return any(link_touch(link) for link in finger_contact_links[n])
            def hold_condition(fs):
                return (1 in fs and len(fs) >= 2) or len(fs) >= 3
            def step_hand(root, q_cmd, root_vel=None):
                q_t = q_now.copy()
                for n in range(1, 6):
                    sl = slice(4 * (n - 1), 4 * n); cmd = q_cmd[sl]
                    step = np.clip(cmd - q_now[sl], -FINGER_RATE, FINGER_RATE); q_t[sl] = q_now[sl] + step
                q_now[:] = q_t
                u.set_robot_targets(ag.contract_to_qpos(root, q_now), None if root_vel is None else ag.contract_to_qpos(root_vel, np.zeros(20)))
                u.step_physics(1)
            def hold_update(k):
                fs = {n for n in range(1, 6) if finger_touch(n)}
                observe_grasp(k, hold_condition(fs), dict(fingers=sorted(fs)))
            if audit: audit.mark('prelude')
            n_pre = int(0.6 * SIM_FREQ)
            for i_ in range(n_pre):                                              # prelude: approach from 8 cm back with OPEN fingers, then close to the row-0 targets
                t_ = min(1.0, i_ / (0.5 * n_pre)); step_hand(back0 + (root6[0] - back0) * t_, q_open0 if t_ < 1.0 else fj[0])
            for _ in range(n_pre):
                step_hand(root6[0], fj[0])
            if audit: audit.mark('actions')
            hold_update(0)
            close_wait_left = [0.0]; rep['close_waits'] = []
            for k in range(len(acts) - 1):
                r_a, r_b = root6[k], root6[k + 1]; f_a, f_b = fj[k], fj[k + 1]; r_v = (r_b - r_a) / dt
                for j in range(1, sub + 1):
                    t = j / sub; step_hand(r_a + (r_b - r_a) * t, f_a + (f_b - f_a) * t, r_v)
                hold_update(k + 1)
                # closure onset (thumb command flexing, nothing held): like the arm's grasp() the wrist WAITS while the fingers travel (<= 1.5 s per
                # closing episode) and the hold is re-checked every 1/30 s; without this a 30 Hz stream walks away before the fingers arrive
                if flex(f_b, 1) > flex(f_a, 1) + 0.02 and not held[0]: close_wait_left[0] = 1.5
                elif flex(f_b, 1) < flex(f_a, 1) - 0.02: close_wait_left[0] = 0.0
                waited = 0.0
                while not held[0] and close_wait_left[0] > 0 and flex(f_b, 1) > flex(q_now, 1) + 0.02:
                    for _ in range(sub): step_hand(r_b, f_b, None)
                    waited += sub / SIM_FREQ; close_wait_left[0] -= sub / SIM_FREQ; hold_update(k + 1)
                if waited > 0: rep['close_waits'].append(dict(row=int(k + 1), waited_s=round(waited, 2), held=held[0]))
                snapshot(k + 1)
            rep['root_track_err_cm'] = round(100 * float(np.linalg.norm(u.robot_link_pose_np(hand_link)[:3] - root6[-1, :3])), 2)
        rep['grasping_final'] = {n: False for n in names}
        if target is not None: rep['grasping_final'][target] = held[0]
        u.stop_robot_velocity_targets()
        if audit: audit.mark('settle')
        if audit: u.step_physics(int(u.sim_freq * a.settle))
        else: u._settle(a.settle)
        if audit:
            audit.mark('final')
            rep['motion_audit'] = audit.save(a.audit_dir)
        if target is not None:
            rep['grasping_final'][target] = grasping() if a.embodiment == 'arm' else hold_condition({n for n in range(1, 6) if finger_touch(n)})
        rep['terminal_motion']=terminal_motion(u)
        rep['quiescent']=rep['terminal_motion']['quiescent']
        rep['terminal_velocity']=max((v['linear_m_s'] for v in rep['terminal_motion']['objects'].values()),default=None)
        rep['final_poses'] = {n: u.obj_pose_np(n).tolist() for n in names}
        rep['traj'] = {n: np.stack(v).tolist() for n, v in rec.items()}; rep['ee'] = ee
        if joint_q_rec:
            rep['joint_q'] = {n: [0.0] + [float(v) if v is not None else None for v in q] for n, q in joint_q_rec.items()}
            rep['joint_q_final'] = {n: u.joint_q_now(n) for n in joint_q_rec}
        rep['prop_poses'] = {}
        for n, act in u.props.items():
            try: pp = act.pose; rep['prop_poses'][n] = np.r_[pp.p[0].cpu().numpy(), pp.q[0].cpu().numpy()].tolist()
            except Exception: pass
        rep['T'] = int(len(acts)); rep['ok'] = True
        if a.embodiment == 'dexhand': rep['adaptive_wait_s'] = sum(e['waited_s'] for e in rep.get('close_waits',[]))
        if a.embodiment=='dexhand':
            rep['self_collision'] = self_collision(Path(a.audit_dir) / 'motion.npz', side)
        rep['physical_validity']=assess(rep)
        if a.video and frames:
            import imageio
            w = imageio.get_writer(a.video, fps=max(1, int(round(1 / dt))), codec='libx264', quality=8, macro_block_size=1)
            for f in frames: w.append_data(f)
            w.close(); rep['video'] = a.video
        if a.full_video and full_frames:
            import imageio
            with imageio.get_writer(a.full_video,fps=30,codec='libx264',quality=8,macro_block_size=1) as writer:
                for frame in full_frames:writer.append_data(frame)
            rep['full_video']=dict(path=a.full_video,fps=30,frames=len(full_frames),includes_prelude=True,includes_settle=True,action_start_s=rep['prelude_s'],action_end_s=rep['prelude_s']+(len(acts)-1)*dt)
    except Exception as e:
        rep['ok'] = False; rep['error'] = f'{type(e).__name__}: {e}'; rep['traceback'] = traceback.format_exc()[-3000:]
    finally:
        env.close()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(rep, open(a.out, 'w'), indent=1, default=float)
    tr = np.asarray(rep['traj'][rep['objects'][0]]) if rep.get('traj') and rep.get('objects') else None
    print(json.dumps(dict(ok=rep.get('ok'), embodiment=a.embodiment, robot=rep.get('robot'), target=rep.get('target'), T=rep['N'], ik_err_cm=rep.get('ik_pos_err_max_cm'), attach=len(rep.get('attach_events', [])), release=len(rep.get('release_events', [])), misses=rep.get('grasp_misses'),
                          z_gain=None if tr is None else round(float(tr[:, 2].max() - tr[0, 2]), 3), final=None if tr is None else tr[-1, :3].round(3).tolist(), quiescent=rep.get('quiescent'), held=(rep.get('grasping_final') or {}).get(rep.get('target')), err=rep.get('error')), default=str))



def execute_bimanual(a):
    """Both Wuji hands on one physical clock; scene objects are never actuated or attached."""
    pkg,sample=Path(a.pkg).resolve(),Path(a.sample).resolve()
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    rep=dict(ok=False,build=False,embodiment='dexhand_bimanual',robot='wuji_bimanual_floating',sample=sample.name,
        package=str(pkg),physics_protocol=VERSION,bimanual_action_version=BIMANUAL_VERSION,hand_count=2,
        object_attachment=False,object_state_override=False,hold_mode='friction',prelude_s=1.2,
        settle_s=float(a.settle),adaptive_wait_s=0.,
        contact_force_threshold_N=CONTACT_FORCE_N,event_semantics='actual contact only; either hand contact prevents released verdict',
        video_includes_prelude=False,video_includes_settle=False)
    env=None
    try:
        m=load_manifest(pkg);data,dt=load_bimanual(pkg,m['actions']);N=len(data['left']['palms'])
        sub=int(round(dt*300))
        if a.sub and a.sub!=sub: raise ValueError('sub override disagrees with the common bimanual clock')
        if not np.isfinite(a.settle) or a.settle<0 or abs(a.settle*300-round(a.settle*300))>1e-6: raise ValueError('invalid settle clock')
        rep.update(N=N,T=N,dt=dt,sub=sub,actions_format='tcp_abs_rpy_grip',primary_hand=m['actions']['primary_hand'])
        cam=json.loads((sample/'hidden/camera.json').read_text());ext=np.asarray(cam['extrinsics_base_cam_seq'])
        phi=json.loads((sample/'hidden/phi.json').read_text())
        import gymnasium as gym,torch,mani_skill.envs  # noqa: F401
        from . import env as _env  # noqa: F401  registers V2SEgoHand-v1
        from .robots import open_fingers
        if not a.no_render:use_software_renderer()
        env=gym.make('V2SEgoHand-v1',obs_mode='none',num_envs=1,sim_backend='physx_cpu',
            render_backend='none' if a.no_render else 'sapien_cpu',scene_spec=scene_in_base_frame(m,pkg),
            camera=None if a.no_render else cam,enable_cameras=not a.no_render,
            robot_uid=rep['robot'])
        u=env.unwrapped;ag=u.agent;rep['build']=True
        lo,hi=ag.robot.get_qlimits()[0].cpu().numpy().T
        backs={};initial={};state={}
        for side in SIDES:
            d=data[side];h=ag.hands[side];rows,_=actions_to_quat(d['palms'],'tcp_abs_rpy_grip');root=root6_stream(rows)
            spec=d['spec'];q0=np.asarray(open_fingers(side),float)
            if spec.get('initial_finger_joints_path'):q0=np.load(pkg/spec['initial_finger_joints_path'],allow_pickle=False).astype(float)
            if q0.shape!=(20,) or not np.isfinite(q0).all() or np.any(q0<lo[h.finger_idx]) or np.any(q0>hi[h.finger_idx]):raise ValueError(f'{side}: invalid initial finger pose')
            offset=np.asarray(spec.get('prelude_offset_in_palm',[0,0,-.08]),float)
            if offset.shape!=(3,) or not np.isfinite(offset).all() or np.linalg.norm(offset)>.25:raise ValueError(f'{side}: invalid prelude offset')
            back=root[0].copy();back[:3]+=R.from_euler('xyz',d['palms'][0,3:6]).as_matrix()@offset
            backs[side]=back;initial[side]=q0
            state[side]=dict(root=root,q=q0.copy(),
                lo=lo[h.finger_idx],hi=hi[h.finger_idx],fingers=np.clip(d['fingers'],lo[h.finger_idx]+1e-4,hi[h.finger_idx]-1e-4))
        env.reset(seed=0,options=dict(settle=0.,qpos=ag.contract_to_qpos(backs,initial)))
        rep['robot_contract']=configure_robot(ag);rep['robot_urdf_sha256']=rep['robot_contract']['urdf_sha256'];rep['robot_urdf_path']=ag.urdf_path
        rep['joint_names']=[j['name'] for j in rep['robot_contract']['joints']]
        rep['joint_velocity_limits']=[j['max_velocity'] for j in rep['robot_contract']['joints']]
        audit=MotionAudit(u);names=list(u.objs)
        mapping=json.loads(a.object_match) if a.object_match else {}
        target=mapping.get(phi['source']) or phi['source']
        if target not in names:raise ValueError('primary object correspondence missing')
        audit.primary_target=target;rep.update(objects=names,target=target,object_match=mapping,
            init_poses={n:u.obj_pose_np(n).tolist() for n in names},attach_events=[],release_events=[],grasp_misses=0,
            finger_control='rate-limited-direct/1.0',hold_diag=[],hands={s:dict(source=data[s]['spec'].get('source'),finger_joints_clipped=True) for s in SIDES})
        for art in u.arts.values():art.active_joints[0].set_drive_properties(stiffness=0.,damping=2.,force_limit=1e5)
        def contacts(side,name=target):
            return {n for n,links in ag.hands[side].finger_links.items() if any(float(torch.linalg.norm(u.scene.get_pairwise_contact_forces(link,u.objs[name])))>CONTACT_FORCE_N for link in links)}
        def touching(side,name):
            links=[ag.hands[side].palm]+[link for ls in ag.hands[side].finger_links.values() for link in ls]
            return any(float(torch.linalg.norm(u.scene.get_pairwise_contact_forces(link,u.objs[name])))>CONTACT_FORCE_N for link in links)
        def flex(q,n):return float(sum(q[4*(n-1)+i] for i in FLEX_IDX[n]))
        frames=[];full=[];held=False
        def camera_at(t):
            k=min(len(ext)-1,max(0,int(round(t/dt))));u.set_camera_pose(ext[k]);return u.render_rgb()
        if a.full_video:full.append(camera_at(0))
        def step(roots,commands,velocities=None):
            for side in SIDES:
                st=state[side]
                st['q']+=np.clip(commands[side]-st['q'],-FINGER_RATE,FINGER_RATE)
            q=ag.contract_to_qpos(roots,{s:state[s]['q'] for s in SIDES})
            qv=None if velocities is None else ag.contract_to_qpos(velocities,{s:np.zeros(20) for s in SIDES})
            u.set_robot_targets(q,qv);u.step_physics(1);audit.step()
            if a.full_video and audit.steps%10==0:full.append(camera_at(max(0,audit.steps/300-rep['prelude_s'])))
        audit.mark('prelude')
        for i in range(180):
            t=min(1.,i/90)
            step({s:backs[s]+(state[s]['root'][0]-backs[s])*t for s in SIDES},
                 {s:initial[s] if t<1 else state[s]['fingers'][0] for s in SIDES})
        for _ in range(180):step({s:state[s]['root'][0] for s in SIDES},{s:state[s]['fingers'][0] for s in SIDES})
        audit.mark('actions');rec={n:[] for n in names};ee={s:[] for s in SIDES};joint={n:[] for n in u.arts}
        def snapshot(k):
            nonlocal held
            for n in names:rec[n].append(u.obj_pose_np(n).tolist())
            for n in joint:joint[n].append(u.joint_q_now(n))
            for s in SIDES:ee[s].append(u.robot_link_pose_np(f'{s}_palm_link').tolist())
            fs={s:sorted(contacts(s)) for s in SIDES};now=any(touching(s,target) for s in SIDES)
            if now and not held:rep['attach_events'].append(dict(frame=k,hands=[s for s in SIDES if touching(s,target)]))
            if held and not now:rep['release_events'].append(k)
            held=now;rep['hold_diag'].append(dict(row=k,contact_fingers=fs,contact_any=now))
            if any(data[s]['palms'][k,6]<.5 for s in SIDES) and not now:rep['grasp_misses']+=1
            if not a.no_render:frames.append(camera_at(k*dt))
        snapshot(0)
        for k in range(N-1):
            for j in range(1,sub+1):
                t=j/sub
                step({s:state[s]['root'][k]*(1-t)+state[s]['root'][k+1]*t for s in SIDES},
                     {s:state[s]['fingers'][k]*(1-t)+state[s]['fingers'][k+1]*t for s in SIDES},
                     {s:(state[s]['root'][k+1]-state[s]['root'][k])/dt for s in SIDES})
            snapshot(k+1)
        rep['root_track_err_cm_by_hand']={s:float(np.linalg.norm(u.robot_link_pose_np(f'{s}_palm_link')[:3]-state[s]['root'][-1,:3])*100) for s in SIDES}
        rep['root_track_err_cm']=max(rep['root_track_err_cm_by_hand'].values())
        u.stop_robot_velocity_targets();audit.mark('settle')
        for _ in range(int(round(a.settle*300))):
            u.step_physics(1);audit.step()
            if a.full_video and audit.steps%10==0:full.append(camera_at((N-1)*dt))
        audit.mark('final');rep['motion_audit']=audit.save(a.audit_dir)
        rep['contact_final_by_hand']={s:{n:touching(s,n) for n in names} for s in SIDES}
        rep['grasping_final']={n:any(rep['contact_final_by_hand'][s][n] for s in SIDES) for n in names}
        rep['terminal_motion']=terminal_motion(u);rep['quiescent']=rep['terminal_motion']['quiescent']
        rep['final_poses']={n:u.obj_pose_np(n).tolist() for n in names};rep['traj']=rec
        rep['ee_by_hand']=ee;rep['ee']=ee[rep['primary_hand']]
        rep['joint_q']=joint;rep['joint_q_final']={n:u.joint_q_now(n) for n in joint}
        rep['prop_poses']={n:np.r_[v.pose.p[0].cpu().numpy(),v.pose.q[0].cpu().numpy()].tolist() for n,v in u.props.items()}
        rep['self_collision']=self_collision(Path(a.audit_dir)/'motion.npz','bimanual')
        rep['ok']=True;rep['physical_validity']=assess(rep)
        if a.video and frames:
            import imageio
            Path(a.video).parent.mkdir(parents=True,exist_ok=True)
            with imageio.get_writer(a.video,fps=1/dt,codec='libx264',quality=8,macro_block_size=1) as w:
                for f in frames:w.append_data(f)
            rep['video']=str(a.video)
        if a.full_video and full:
            import imageio
            Path(a.full_video).parent.mkdir(parents=True,exist_ok=True)
            with imageio.get_writer(a.full_video,fps=30,codec='libx264',quality=8,macro_block_size=1) as w:
                for f in full:w.append_data(f)
            rep['full_video']=dict(path=str(a.full_video),fps=30,frames=len(full),includes_prelude=True,includes_settle=True,action_start_s=1.2,action_end_s=1.2+(N-1)*dt)
    except Exception as ex:
        rep.update(ok=False,error=f'{type(ex).__name__}: {ex}',traceback=traceback.format_exc())
    finally:
        if env is not None:env.close()
    out.write_text(json.dumps(rep,indent=1,default=float));print(json.dumps({k:rep.get(k) for k in ['ok','robot','N','error','physical_validity']},default=float))
    return rep



if __name__ == '__main__':
    os.environ.setdefault('VK_ICD_FILENAMES', '/etc/vulkan/icd.d/nvidia_icd.json' if Path('/etc/vulkan/icd.d/nvidia_icd.json').exists() else '/usr/share/vulkan/icd.d/lvp_icd.x86_64.json')
    main()
