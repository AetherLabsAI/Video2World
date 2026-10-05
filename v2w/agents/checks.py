"""Package checks fed back to the agent between rounds: schema, robot identity and what the outcome test needs."""
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np

from v2w import paths

HERE = Path(__file__).resolve().parent
NATIVE_PROFILES = ('robodojo_candidate_task_v1', 'inhouse_provider_native_v2')
TWIN_PROFILES = ('twin_rope', 'twin_toy')
CLOTH_PROFILES = ('droid_cloth',)
# execution robot per rigid/hand profile
ROBOTS = {'fb_lamp': 'panda', 'fb_furniture': 'panda', 'fb_droid': 'panda_robotiq', 'fb_pusht': 'xarm7_pusher',
          'ego_arm': 'panda', 'ego_bimanual': 'wuji_bimanual_floating'}
# accepted names of the receiving part per furniture (what the task predicate matches)
FURNITURE_TARGET = {'one_leg': ('table top with the corner holes', ['table_top', 'tabletop', 'top_plate', 'table_plate', 'square_table_top']),
                    'drawer': ('drawer box / housing the tray slides into', ['drawer_box', 'drawer_body', 'drawer_frame', 'drawer_cabinet', 'drawer_housing', 'box']),
                    'cabinet': ('cabinet body / shell the door is fitted onto', ['cabinet_body', 'cabinet_frame', 'cabinet_shell', 'cabinet_housing', 'body'])}
BIMANUAL_VERSION = 'wuji-bimanual-actions/1.0'


def _handout(name):
    spec = importlib.util.spec_from_file_location('v2w_handout_' + name, HERE / 'handouts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_errors(profile, pkg, public):
    from v2w.tracks.robodojo.contract import errors
    if public is not None:
        public = dict(public, robot_asset_directory=str(paths.asset('robots', public['robot_asset'])))
    return errors(profile, pkg, public)


def twin_errors(pkg):
    """The twin validator runs in the twin interpreter (MuJoCo)."""
    code = 'import json,sys;from video2sim.native_twin import validate;print(json.dumps(validate(sys.argv[1])))'
    result = subprocess.run([paths.tool('twin_python'), '-c', code, str(Path(pkg).resolve())], env=paths.env(),
                            text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError('twin validator unavailable: ' + result.stderr[-2000:])
    errors = json.loads(result.stdout.strip().splitlines()[-1])
    if not isinstance(errors, list):
        raise RuntimeError('invalid twin validator response')
    return errors


def wallet_errors(pkg, public):
    try:
        protocol = json.loads((Path(pkg) / 'protocol.json').read_text())
        scene = json.loads((Path(pkg) / protocol['scene_file']).read_text())
        expected = 'complete_deformable_surface' if scene['objects'][0].get('deformable') else 'rigid_object_pose'
        if protocol.get('state_kind') != expected:
            raise ValueError('state_kind must be ' + expected + ' for the submitted dynamics')
        _handout('wallet_contract').validate(scene, pkg)
        if len(np.load(Path(pkg) / protocol['actions']['path'])) != public['control_frames']:
            raise ValueError('Actions must match public control_frames at 20 Hz')
    except (ValueError, TypeError, KeyError, OSError) as exc:
        return [str(exc)]
    return []


def robot_errors(profile, manifest):
    expected = ROBOTS.get(profile)
    if profile.startswith('fb_') and expected and manifest.get('robot', {}).get('uid') != expected:
        return ['This profile executes ' + expected + '; robot.uid must declare that robot (nominal WidowX is not a replay backend).']
    return []


def bimanual_errors(pkg, actions):
    """Synchronized left/right palm rows and 20 finger joints per hand."""
    if actions.get('bimanual_version') != BIMANUAL_VERSION:
        raise ValueError('missing/incompatible bimanual action contract')
    hands = actions.get('hands')
    if not isinstance(hands, dict) or set(hands) != {'left', 'right'}:
        raise ValueError('both left and right action streams are required')
    primary = actions.get('primary_hand')
    if primary not in hands or actions.get('path') != hands[primary].get('path'):
        raise ValueError('primary action path must alias the declared primary hand')
    files = []
    for side in ('left', 'right'):
        h = hands[side]
        if set(h) & {'dt', 'format'}:
            raise ValueError('both hands share the top-level dt and format')
        for key in ('path', 'finger_joints_path'):
            path = h.get(key)
            if not isinstance(path, str) or Path(path).is_absolute() or '..' in Path(path).parts:
                raise ValueError('hand files must be package-relative paths')
            files.append(path)
    if len(set(files)) != 4:
        raise ValueError('left/right palm and finger streams must use distinct files')
    if actions.get('format') != 'tcp_abs_rpy_grip':
        raise ValueError('bimanual palms require tcp_abs_rpy_grip')
    dt = float(actions.get('dt', 0))
    if not np.isfinite(dt) or dt <= 0 or abs(dt * 300 - round(dt * 300)) > 1e-6:
        raise ValueError('bimanual dt must be a positive multiple of 1/300 s')
    length = None
    for side in ('left', 'right'):
        palms = np.load(Path(pkg) / hands[side]['path'], allow_pickle=False).astype(float)
        fingers = np.load(Path(pkg) / hands[side]['finger_joints_path'], allow_pickle=False).astype(float)
        if palms.ndim != 2 or palms.shape[1] != 7 or len(palms) < 2 or fingers.shape != (len(palms), 20):
            raise ValueError('expected (T,7) palms and (T,20) fingers, T >= 2')
        if not np.isfinite(palms).all() or not np.isfinite(fingers).all():
            raise ValueError('nonfinite hand commands')
        if length is not None and len(palms) != length:
            raise ValueError('left/right action clocks differ')
        length = len(palms)


def semantic_errors(profile, pkg, m, sample):
    """What the outcome test needs from the package beyond the schema."""
    errs = []
    scene = m.get('scene') or {}
    objs, props = scene.get('objects', []), scene.get('props', [])
    if profile == 'fb_furniture':
        try:
            furniture = json.load(open(Path(sample) / 'meta.json')).get('furniture')
        except Exception:
            furniture = None
        desc, names = FURNITURE_TARGET.get(furniture, (None, []))
        if names:
            if not [o for o in props if any(k in str(o.get('name', '')).lower() for k in names)]:
                errs.append(f"the RECEIVING part ({desc}) is missing or not recognisably named: deliver it as a static prop whose name contains one of {names} "
                            f"(pieces may be suffixed, e.g. {names[0]}_left_wall). Task Success is judged by where the manipulated part ends relative to this prop; "
                            "without it the outcome cannot be scored.")
        if len(objs) != 1:
            errs.append(f"scene.objects must contain exactly ONE entry, the manipulated part (found {len(objs)}); everything static goes under scene.props")
    if profile == 'fb_pusht':
        if len(objs) != 1:
            errs.append(f"scene.objects must contain exactly ONE entry, the pushed T block (found {len(objs)}); the yellow goal marker is a rendered overlay, "
                        "not a physical object -- do not deliver it as an object (a static prop of zero height is fine but unnecessary)")
        elif objs[0].get('kind') == 'mesh' and not objs[0].get('collision_path'):
            errs.append("the T block is NOT convex: deliver collision_path as an OBJ with two groups (bar / stem, `o bar` / `o stem`) or model it as two boxes "
                        "in one mesh with two groups; a single convex hull fills the notch the pusher works in")
    if profile == 'ego_bimanual':
        try:
            bimanual_errors(pkg, m.get('actions') or {})
        except Exception as ex:
            errs.append(f'bimanual actions invalid: {ex}')
    if profile == 'fb_droid':
        if len(objs) != 1:
            errs.append(f"scene.objects must contain exactly ONE entry, the manipulated object (found {len(objs)}); everything static goes under scene.props")
        if not props:
            errs.append("the object the task refers to (the block it is stacked on / the bowl or box it goes into / the object it is placed next to) is missing: "
                        "deliver it as a static prop with its size and pose")
    return errs


def errors(profile, pkg, sample=None):
    """Error strings for the agent; an empty list means the package is acceptable for this profile."""
    from video2sim.bench.protocol import load_manifest, validate_manifest
    try:
        m = load_manifest(pkg)
        if profile in TWIN_PROFILES:
            return twin_errors(pkg)
        if profile in CLOTH_PROFILES:
            from v2w.tracks.cloth.contract import errors as cloth_errors
            return cloth_errors(pkg)
        if profile in NATIVE_PROFILES:
            public = json.loads((HERE / 'tasks.json').read_text()).get(Path(sample).name) if sample else None
            errs = native_errors(profile, pkg, public)
            if not errs and public and public.get('state_kind') == 'complete_deformable_surface':
                errs = wallet_errors(pkg, public)
            return errs
        errs = robot_errors(profile, m) or validate_manifest(pkg, m)
        if not errs and sample is not None:
            errs = semantic_errors(profile, pkg, m, sample)
    except Exception as e:
        return [str(e)]
    if not errs and not (Path(pkg) / 'expected' / 'obj_poses.npy').exists():
        errs.append('expected/obj_poses.npy missing (required by this profile)')
    return errs
