"""Candidate packages of the in-house track (profile ``inhouse_provider_native_v2``).

A package is authored in the video camera's OpenCV frame: protocol.json (T_world_camera, actions), scene.json and a
(T, 20) joint command array at 20 Hz: left arm 1-7, right arm 1-7, waist lift / pitch, head yaw / pitch, grippers
(0 open, 1 closed). ``load`` validates it against the public task; ``native_package`` rewrites it in the robot base
frame for the Isaac runtime.
"""
import copy
import json
import re
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from v2w import paths

PHYSICS_PROFILE = 'inhouse_camera_native_v1'
ACTION_FORMAT = 'arm14_waist2_head2_gripper2'
JOINTS = [f'idx2{i}_arm_l_joint{i}' for i in range(1, 8)] + [f'idx6{i}_arm_r_joint{i}' for i in range(1, 8)] + \
         ['idx01_body_joint1', 'idx02_body_joint2', 'idx11_head_joint1', 'idx12_head_joint2']


def read(p):
    return json.loads(Path(p).read_text())


def write(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def robot_dir():
    return paths.asset('robots', 'g1_omnipicker')


def local(pkg, name):
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError('Package member must be relative')
    p = (Path(pkg) / name).resolve()
    if not p.is_relative_to(Path(pkg).resolve()) or not p.is_file():
        raise ValueError('Missing or external package member: ' + name)
    return p


def rigid_transform(value):
    t = np.asarray(value, float)
    if t.shape != (4, 4) or not np.isfinite(t).all() or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError('T_world_camera must be finite rigid 4x4')
    r = t[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or abs(np.linalg.det(r) - 1) > 1e-5:
        raise ValueError('Camera transform may not scale, shear or reflect')
    return t


def load(pkg, public):
    """(protocol, scene, actions, T_world_camera) of a valid package; ValueError names the first violation."""
    pkg = Path(pkg).resolve()
    pr = json.loads((pkg / 'protocol.json').read_text())
    if not isinstance(pr, dict):
        raise ValueError('protocol.json must be an object')
    if pr.get('physics_profile') != PHYSICS_PROFILE or pr.get('frame') != 'camera_opencv':
        raise ValueError('Expected ' + PHYSICS_PROFILE + ' in camera_opencv metres')
    t = rigid_transform(pr.get('T_world_camera'))
    scene = json.loads(local(pkg, pr.get('scene_file')).read_text())
    if not isinstance(scene, dict):
        raise ValueError('scene.json must be an object')
    objects, props = scene.get('objects', []), scene.get('props', [])
    if not isinstance(objects, list) or not isinstance(props, list):
        raise ValueError('objects and props must be arrays')
    if scene.get('table') is not None and not isinstance(scene['table'], dict):
        raise ValueError('table must be an object')
    entities = objects + props + ([scene['table']] if scene.get('table') else [])
    if any(not isinstance(e, dict) for e in entities):
        raise ValueError('Scene entities must be objects')
    if any(not isinstance(e.get('name'), str) for e in entities):
        raise ValueError('Entity names must be strings')
    names = [e['name'] for e in entities]
    if not objects or len(names) != len(set(names)):
        raise ValueError('Dynamic objects required; entity names must be unique')
    for e in entities:
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', e['name']):
            raise ValueError('Entity ID must be a USD identifier')
        for key, n in [('size', 3), ('pos', 3), ('quat_wxyz', 4), ('color', 3)]:
            x = np.asarray(e.get(key), float)
            if x.shape != (n,) or not np.isfinite(x).all():
                raise ValueError('Invalid ' + key)
            if key == 'size' and np.min(x) <= 0:
                raise ValueError('Nonpositive size')
            if key == 'quat_wxyz' and abs(np.linalg.norm(x) - 1) > 1e-4:
                raise ValueError('Quaternion must be normalized')
        mass = float(e.get('mass', .1))
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError('Mass must be finite and positive')
        for key, default in [('static_friction', .6), ('dynamic_friction', .6), ('restitution', 0.)]:
            value = float(e.get(key, default))
            if not np.isfinite(value) or value < 0 or (key == 'restitution' and value > 1):
                raise ValueError('Invalid material coefficient')
        if e.get('collision') is False:
            raise ValueError('Submitted entities must collide')
        if e.get('asset_path'):
            raise ValueError('Submit mesh geometry_npz, not a USD with external composition')
        if 'scale' in e and not np.array_equal(e['scale'], [1, 1, 1]):
            raise ValueError('Bake scale into mesh vertices and size')
        if e.get('geometry_npz'):
            with np.load(local(pkg, e['geometry_npz']), allow_pickle=False) as z:
                v, c, i = np.asarray(z['vertices']), np.asarray(z['face_vertex_counts']), np.asarray(z['face_vertex_indices'])
            if v.ndim != 2 or v.shape[1] != 3 or not len(v) or not np.isfinite(v).all():
                raise ValueError('Invalid mesh vertices')
            if c.ndim != 1 or i.ndim != 1 or not np.issubdtype(c.dtype, np.integer) or not np.issubdtype(i.dtype, np.integer) or np.any(c < 3) or c.sum() != len(i) or np.any(i < 0) or np.any(i >= len(v)):
                raise ValueError('Invalid mesh topology')
            if not np.allclose(np.ptp(v, axis=0), e['size'], atol=1e-5, rtol=1e-3):
                raise ValueError('size must describe submitted mesh bounds')
    if pr.get('task') != public['task'] or pr.get('embodiment') != public['embodiment']:
        raise ValueError('Task/embodiment mismatch')
    if not set(public['dynamic_objects']).issubset({e['name'] for e in objects}):
        raise ValueError('Missing dynamic task object')
    if not set(public['required_entities']).issubset(names):
        raise ValueError('Missing required task receiver')
    if 'receiver' in public['required_entities'] and not any(e['name'] == 'receiver' for e in props):
        raise ValueError('The receiver must be a static prop')
    if pr.get('embodiment') != 'g1_omnipicker':
        raise ValueError('Expected g1_omnipicker')
    a = pr.get('actions', {})
    if not isinstance(a, dict):
        raise ValueError('actions must be an object')
    if a.get('dt') != .05 or a.get('format') != ACTION_FORMAT:
        raise ValueError('Incorrect action format or dt')
    q = np.load(local(pkg, a.get('path')), allow_pickle=False)
    if not isinstance(q, np.ndarray):
        raise ValueError('Actions must be an NPY array')
    if q.ndim != 2 or q.shape[1] != 20 or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError('Invalid action array')
    if np.any(q[:, [18, 19]] < 0) or np.any(q[:, [18, 19]] > 1):
        raise ValueError('Gripper commands must be in [0,1]')
    spec = read(robot_dir() / 'robot_spec.json')['joints']
    for i, name in enumerate(JOINTS):
        limit = spec[name]
        if ('lower' in limit and np.any(q[:, i] < float(limit['lower']) - 1e-6)) or ('upper' in limit and np.any(q[:, i] > float(limit['upper']) + 1e-6)):
            raise ValueError('Joint command outside public native limit at column ' + str(i))
    return pr, scene, q, t


def world_scene(scene, t):
    """Scene entities moved from the camera frame to the robot base (world) frame."""
    result = copy.deepcopy(scene)
    for e in result['objects'] + result.get('props', []) + ([result['table']] if result.get('table') else []):
        e['pos'] = (t[:3, :3] @ np.asarray(e['pos']) + t[:3, 3]).tolist()
        r = t[:3, :3] @ Rotation.from_quat(np.roll(e['quat_wxyz'], -1)).as_matrix()
        e['quat_wxyz'] = np.roll(Rotation.from_matrix(r).as_quat(), 1).tolist()
    return result


def native_package(pkg, sample, dest, public, settle_frames, initial_arm=None):
    """Write the robot-base package the Isaac runtime executes; returns the world-frame scene."""
    pr, scene, q, t = load(pkg, public)
    world = world_scene(scene, t)
    target = next(e for e in world['objects'] if e['name'] == 'target')
    shutil.copytree(pkg, dest)
    write(dest / 'scene.json', world)
    np.save(dest / 'actions.npy', np.c_[q[:, :14], q[:, 18:20]])
    np.savez(dest / 'posture.npz', waist=q[:, [15, 14]], head=q[:, 16:18], initial_arm=q[0, :14] if initial_arm is None else initial_arm,
             timestamp_ns=np.arange(len(q), dtype=np.int64) * 50_000_000)
    write(dest / 'protocol.json', dict(physics_profile=PHYSICS_PROFILE, sample=Path(sample).name, kind=public['task'], target='/World/Entities/target',
                                       T_world_base=np.eye(4).tolist(), initial_object_xyz=target['pos'], initial_object_wxyz=target['quat_wxyz'],
                                       settle_frames=settle_frames, frames=len(q), source_camera_transform=t.tolist()))
    return world, q


def entity_geometry(entity, pkg, dest):
    """An eval-metric entity (box, or mesh exported to dest as OBJ) from a package scene entity."""
    import trimesh
    result = dict(name=entity['name'], kind='box', pos=entity['pos'], quat=entity['quat_wxyz'], half_size=(np.asarray(entity['size']) / 2).tolist())
    if entity.get('geometry_npz'):
        with np.load(Path(pkg) / entity['geometry_npz'], allow_pickle=False) as z:
            vertices, counts, indices = z['vertices'], z['face_vertex_counts'], z['face_vertex_indices']
        faces, offset = [], 0
        for count in counts:
            poly = indices[offset:offset + count]
            offset += count
            faces.extend([[poly[0], poly[j], poly[j + 1]] for j in range(1, len(poly) - 1)])
        path = Path(dest) / (entity['name'] + '.obj')
        trimesh.Trimesh(vertices, faces, process=False).export(path)
        result.update(kind='mesh', mesh_path=str(path.resolve()), scale=1.)
    return result


def scene_geometry(authored, pkg, dest):
    scene = dict(objects=[entity_geometry(e, pkg, dest) for e in authored['objects']], props=[entity_geometry(e, pkg, dest) for e in authored.get('props', [])], support=None)
    if authored.get('table'):
        scene['props'].append(entity_geometry(authored['table'], pkg, dest))
    return scene


def validate_wallet(scene, pkg):
    """One rigid or closed-shell elastic wallet target with a full triangle mesh; an optional open receiver cavity."""
    import trimesh
    pkg = Path(pkg)
    if len(scene.get('objects', [])) != 1 or scene['objects'][0]['name'] != 'target':
        raise ValueError('One dynamic target is required')
    e = scene['objects'][0]
    d = e.get('deformable', {})
    if d and d.get('model') != 'closed_particle_shell':
        raise ValueError('Unsupported wallet deformable model')
    if e.get('dynamics', 'rigid' if not d else 'elastic') not in ['rigid', 'elastic']:
        raise ValueError('Unsupported wallet dynamics')
    if e.get('dynamics') == 'rigid' and d:
        raise ValueError('Rigid wallet cannot declare a particle shell')
    ranges = {'particle_radius_m': (.0003, .004), 'stretch_stiffness': (1., 100000.), 'bend_stiffness': (.001, 10000.), 'shear_stiffness': (1., 100000.), 'damping': (0., 10.), 'pressure': (0., 1.)}
    for k, (lo, hi) in (ranges.items() if d else []):
        value = float(d.get(k, float('nan')))
        if not np.isfinite(value) or not lo <= value <= hi:
            raise ValueError('Invalid shell ' + k)
    if not e.get('geometry_npz'):
        raise ValueError('Full wallet triangle mesh is required')
    with np.load(pkg / e['geometry_npz'], allow_pickle=False) as z:
        v, counts, ii = z['vertices'], z['face_vertex_counts'], z['face_vertex_indices']
    if np.any(counts != 3) or len(v) > 5000 or len(counts) > 20000:
        raise ValueError('Triangular surface required, max 5000 vertices / 20000 faces')
    m = trimesh.Trimesh(v, ii.reshape(-1, 3), process=False)
    if d and (not m.is_watertight or not m.is_winding_consistent or m.volume <= 1e-10):
        raise ValueError('Elastic target must be a consistently oriented, closed positive-volume shell')
    if not d and (len(v) < 4 or (np.ptp(v, axis=0) <= 0).any()):
        raise ValueError('Rigid target must have finite nonzero 3D extents')
    receiver = next((p for p in scene.get('props', []) if p['name'] == 'receiver'), None)
    if receiver is not None:
        r = receiver.get('interior', {})
        center, half = np.asarray(r.get('center'), float), np.asarray(r.get('half_extents_xy'), float)
        if r.get('frame') != 'entity_local' or center.shape != (3,) or half.shape != (2,) or not np.isfinite(center).all() or not np.isfinite(half).all() or (half <= 0).any():
            raise ValueError('Receiver needs finite local cavity center and half_extents_xy')
        if not np.isfinite([r.get('bottom_z'), r.get('rim_z')]).all() or not r['rim_z'] > r['bottom_z']:
            raise ValueError('Invalid receiver opening/bottom')
        if not receiver.get('geometry_npz'):
            raise ValueError('Receiver needs actual open triangle mesh geometry')
