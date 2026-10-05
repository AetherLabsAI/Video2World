"""Action streams of the hand track: single-stream TCP / palm rows and the two-hand (Wuji pair) contract."""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

SIDES = ('left', 'right')
BIMANUAL_VERSION = 'wuji-bimanual-actions/1.0'


def actions_to_quat(A, fmt=None):
    """(N,8) [x y z qw qx qy qz closed] from either an (N,8) quaternion stream or a (T,7) [x y z roll pitch yaw grip] stream."""
    A = np.asarray(A, np.float64)
    if A.ndim == 2 and A.shape[1] == 8:
        return A.copy(), 'ego (N,8) quat + grip(1=closed)'
    if A.ndim == 2 and A.shape[1] == 7:
        out = np.zeros((len(A), 8))
        out[:, :3] = A[:, :3]
        q = R.from_euler('ZYX', A[:, [5, 4, 3]]).as_quat()          # yaw, pitch, roll -> R = Rz Ry Rx
        out[:, 3] = q[:, 3]; out[:, 4:7] = q[:, :3]
        out[:, 7] = 1.0 - np.clip(A[:, 6], 0, 1)                    # grip 1 = open -> closed = 1 - grip
        return out, 'bridge (T,7) euler + grip(1=open) -> quat, closed = 1 - grip'
    raise ValueError(f'actions must be (N,8) or (T,7), got {A.shape}')


def root6_stream(rows8):
    """(N,8) palm rows -> (N,6) floating-root joint targets [x y z a b c] with R = Rx(a) Ry(b) Rz(c), unwrapped along the stream."""
    raw = R.from_quat(np.asarray(rows8)[:, [4, 5, 6, 3]]).as_euler('XYZ')
    ang = raw.copy()
    for i in range(1, len(raw)):
        # Crossing the middle-axis singularity changes all three Euler angles; pick the branch closest to the previous row.
        x, y, z = raw[i]
        choices = np.array([[x, y, z], [x + np.pi, np.pi - y, z + np.pi]])
        choices += 2 * np.pi * np.round((ang[i - 1] - choices) / (2 * np.pi))
        ang[i] = choices[np.argmin(np.linalg.norm(choices - ang[i - 1], axis=1))]
    return np.c_[np.asarray(rows8)[:, :3], ang]


def hand_specs(actions):
    """Validated per-hand stream declarations of a two-hand package (`actions.hands`)."""
    if actions.get('bimanual_version') != BIMANUAL_VERSION:
        raise ValueError('missing/incompatible bimanual action contract')
    hands = actions.get('hands')
    if not isinstance(hands, dict) or set(hands) != set(SIDES):
        raise ValueError('both left and right action streams are required')
    primary = actions.get('primary_hand')
    if primary not in SIDES or actions.get('path') != hands[primary].get('path'):
        raise ValueError('primary action path must alias the declared primary hand')
    paths = []
    for side in SIDES:
        h = hands[side]
        if set(h) & {'dt', 'format'}:
            raise ValueError('both hands share the top-level dt and format')
        for key in ['path', 'finger_joints_path']:
            path = h.get(key)
            if not isinstance(path, str) or Path(path).is_absolute() or '..' in Path(path).parts:
                raise ValueError('hand files must be package-relative paths')
            paths.append(path)
    if len(set(paths)) != 4:
        raise ValueError('left/right palm and finger streams must use distinct files')
    return hands


def load_bimanual(pkg, actions):
    """({side: dict(spec, palms (T,7), fingers (T,20))}, dt) on one common clock."""
    specs = hand_specs(actions); data = {}; length = None
    if actions.get('format') != 'tcp_abs_rpy_grip':
        raise ValueError('bimanual palms require tcp_abs_rpy_grip')
    dt = float(actions.get('dt', 0))
    if not np.isfinite(dt) or dt <= 0 or abs(dt * 300 - round(dt * 300)) > 1e-6:
        raise ValueError('bimanual dt must be a positive multiple of 1/300 s')
    for side, h in specs.items():
        palms = np.load(Path(pkg) / h['path'], allow_pickle=False).astype(float)
        fingers = np.load(Path(pkg) / h['finger_joints_path'], allow_pickle=False).astype(float)
        if palms.ndim != 2 or palms.shape[1] != 7 or len(palms) < 2 or fingers.shape != (len(palms), 20):
            raise ValueError('expected (T,7) palms and (T,20) fingers, T >= 2')
        if not np.isfinite(palms).all() or not np.isfinite(fingers).all():
            raise ValueError('nonfinite hand commands')
        if length is not None and len(palms) != length:
            raise ValueError('left/right action clocks differ')
        length = len(palms); data[side] = dict(spec=h, palms=palms, fingers=fingers)
    return data, dt
