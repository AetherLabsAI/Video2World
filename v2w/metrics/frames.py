"""Evaluator-side frame alignment of packages delivered in the camera frame.

A package whose cameras[0].extrinsics_base_cam is the agent's own estimate is rewritten into the robot base frame with
the hidden camera extrinsics (never exposed to the agent): entity poses, expected trajectories, the camera and the
action streams. Library: align_package(pkg_dir, T_base_cam (4x4), out_dir) -> info dict.
"""
import json, shutil, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
BIMANUAL_SIDES = ('left', 'right')
BIMANUAL_VERSION = 'wuji-bimanual-actions/1.0'


def hand_specs(actions):
    """Validated per-hand streams of a two-hand action contract: synchronized palm rows and 20 finger joints per hand."""
    if actions.get('bimanual_version') != BIMANUAL_VERSION: raise ValueError('missing/incompatible bimanual action contract')
    hands = actions.get('hands')
    if not isinstance(hands, dict) or set(hands) != set(BIMANUAL_SIDES): raise ValueError('both left and right action streams are required')
    primary = actions.get('primary_hand')
    if primary not in BIMANUAL_SIDES or actions.get('path') != hands[primary].get('path'): raise ValueError('primary action path must alias the declared primary hand')
    files = []
    for side in BIMANUAL_SIDES:
        h = hands[side]
        if set(h) & {'dt', 'format'}: raise ValueError('both hands share the top-level dt and format')
        for key in ['path', 'finger_joints_path']:
            path = h.get(key)
            if not isinstance(path, str) or Path(path).is_absolute() or '..' in Path(path).parts: raise ValueError('hand files must be package-relative paths')
            files.append(path)
    if len(set(files)) != 4: raise ValueError('left/right palm and finger streams must use distinct files')
    return hands


def _q_wxyz_to_R(q): w, x, y, z = q; return R.from_quat([x, y, z, w]).as_matrix()
def _R_to_q_wxyz(M): x, y, z, w = R.from_matrix(M).as_quat(); return [float(w), float(x), float(y), float(z)]
def _xf(T, pos, quat):
    p = T[:3, :3] @ np.asarray(pos, float) + T[:3, 3]; M = T[:3, :3] @ _q_wxyz_to_R(quat); return [float(v) for v in p], _R_to_q_wxyz(M)
def cube_half_from_entity(e, pkg):
    if e.get('kind') == 'box' and e.get('half_size'): return float(np.mean(e['half_size']) * float(e.get('scale', 1.0)))
    if e.get('kind') in ('mesh',) and e.get('mesh_path'):
        # info-only field: a malformed candidate mesh (seen in the wild: an OBJ whose last face indexes one past its
        # vertex count) must not abort the frame alignment -- the evaluator renders its own verdict on the mesh later.
        try:
            import trimesh; m = trimesh.load(str(Path(pkg) / e['mesh_path']), force='mesh')
            return float(np.mean(m.bounding_box.extents) * float(e.get('scale', 1.0)) / 2)
        except Exception:
            return None
    if e.get('half_size'): return float(np.mean(e['half_size']))
    return None
def align_package(pkg, T_hidden_cam, out):
    pkg, out = Path(pkg), Path(out); m = json.load(open(pkg / 'protocol.json'))
    cams = m.get('cameras') or []; E = np.asarray(cams[0]['extrinsics_base_cam'], float) if cams else np.eye(4)
    identity = np.allclose(E, np.eye(4), atol=1e-6); T = np.asarray(T_hidden_cam, float) @ np.linalg.inv(E)   # base_hidden <- base_agent
    if out.exists(): shutil.rmtree(out)
    shutil.copytree(pkg, out, symlinks=True)
    info = dict(source=str(pkg), agent_extrinsics_identity=bool(identity), T_hidden_from_agent=T.tolist(), objects=[])
    # Exact same frame: preserve commands/poses byte-for-byte. Recomputing an
    # identity transform normalizes Euler branches and perturbs IK/contact inputs.
    # Use exact equality (not an approximate threshold), so real small transforms
    # are still applied to candidate submissions.
    if np.array_equal(E, np.asarray(T_hidden_cam, float)) and np.isfinite(E).all():
        info['T_hidden_from_agent'] = np.eye(4).tolist()
        info['already_in_hidden_frame'] = True
        for e in (m.get('scene') or {}).get('objects') or []:
            info['objects'].append(dict(name=e.get('name'), kind=e.get('kind'), pos=e.get('pos'), quat=e.get('quat'), cube_half=cube_half_from_entity(e, pkg)))
        action = (m.get('actions') or {}).get('path')
        if action and (pkg / action).exists(): info['actions_T'] = len(np.load(pkg / action))
        if (pkg / 'expected/obj_poses.npy').exists(): info['trajectory_frames'] = len(np.load(pkg / 'expected/obj_poses.npy'))
        if (pkg / 'expected/obj_poses.npz').exists():
            with np.load(pkg / 'expected/obj_poses.npz') as z: info['trajectory_objects'] = z.files
        json.dump(info, open(out / 'frame_alignment.json', 'w'), indent=1)
        return info
    for key in ('objects', 'props', 'articulations'):
        for e in (m.get('scene') or {}).get(key) or []:
            if 'pos' in e:
                e['pos'], e['quat'] = _xf(T, e['pos'], e.get('quat', [1, 0, 0, 0]))
            if key == 'objects': info['objects'].append(dict(name=e.get('name'), kind=e.get('kind'), pos=e['pos'], quat=e.get('quat'), cube_half=cube_half_from_entity(e, pkg)))
    sup = (m.get('scene') or {}).get('support')
    if sup is not None:   # support plane given in the camera frame cannot be re-expressed as a z-plane in general; keep the hidden convention
        info['support_note'] = 'support left unchanged'
    for c in cams: c['extrinsics_base_cam'] = np.asarray(T_hidden_cam, float).tolist(); c['frame_alignment'] = 'evaluator: hidden T_base_cam @ inv(agent extrinsics)'
    m['provenance'] = dict(m.get('provenance') or {}, frame_alignment='camera-anchored (evaluator-side, hidden extrinsics)')
    json.dump(m, open(out / 'protocol.json', 'w'), indent=1)
    trz = pkg / 'expected' / 'obj_poses.npz'
    if trz.exists():
        z = dict(np.load(trz)); zo = {}
        for k, arr in z.items():
            arr = np.asarray(arr, float); b = arr.copy()
            for i in range(len(arr)):
                p_, q_ = _xf(T, arr[i, :3], arr[i, 3:7]); b[i, :3] = p_; b[i, 3:7] = q_
            zo[k] = b.astype(np.float32)
        (out / 'expected').mkdir(exist_ok=True); np.savez(out / 'expected' / 'obj_poses.npz', **zo); info['trajectory_objects'] = list(zo)
    tr = pkg / 'expected' / 'obj_poses.npy'
    if tr.exists():
        a = np.load(tr).astype(np.float64); b = a.copy()
        for i in range(len(a)):
            p, q = _xf(T, a[i, :3], a[i, 3:7]); b[i, :3] = p; b[i, 3:7] = q
        (out / 'expected').mkdir(exist_ok=True); np.save(out / 'expected' / 'obj_poses.npy', b.astype(np.float32)); info['trajectory_frames'] = int(len(a))
    # v2 contract: the package's OWN action stream, (T,7) [x y z roll pitch yaw grip] absolute TCP poses
    # (R = Rz(yaw)Ry(pitch)Rx(roll)) in the delivery (camera) frame -> re-expressed in the hidden base frame;
    # the gripper command passes through untouched.
    action_spec = m.get('actions') or {}
    action_paths = [action_spec.get('path')]
    if action_spec.get('hands') is not None:
        action_paths.extend(h['path'] for h in hand_specs(action_spec).values())
    for ac in dict.fromkeys(action_paths):
        if not ac or not (pkg / ac).exists(): continue
        A = np.load(pkg / ac).astype(np.float64); B = A.copy()
        for i in range(len(A)):
            M = T[:3, :3] @ R.from_euler('ZYX', [A[i, 5], A[i, 4], A[i, 3]]).as_matrix()
            B[i, :3] = T[:3, :3] @ A[i, :3] + T[:3, 3]
            yaw, pitch, roll = R.from_matrix(M).as_euler('ZYX')
            B[i, 3:6] = [roll, pitch, yaw]
        (out / Path(ac)).parent.mkdir(parents=True, exist_ok=True)
        np.save(out / ac, B); info['actions_T'] = int(len(A))
    json.dump(info, open(out / 'frame_alignment.json', 'w'), indent=1); return info
