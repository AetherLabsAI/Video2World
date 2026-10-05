"""Metric record of one RoboDojo execution against the source recording of the demonstration.

Geometry: the shared Scene CD on per-object interaction roles after one common initial-anchor translation, with
context weights from the source object sweeps and both source TCP streams; object shape/size per task object.
Dynamics: demonstration-clock trajectory error per task object. Function: the native task predicates replayed on
the recorded states, with an initially completed goal requiring an observed exit and re-entry.
"""
import copy
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from v2w import paths
from v2w.metrics import geometry as M
from v2w.metrics.record import set_scene_result
from v2w.metrics.trajectory import demo_clock_metrics, terminal_position

VERSION = 'robodojo-task-metrics/3'
CATALOG = paths.DATA / 'sources/robodojo/all_tasks_v1/catalog'
CONSTANT = {'cap_cm', 'n', 'delta', 'delta_seconds', 'requested_delta_seconds', 'start_frame_indices', 'end_frame_indices', 'interval_seconds'}


def read(p):
    return json.loads(Path(p).read_text())


def write(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, indent=2, allow_nan=False) + '\n')


def definition(task, source):
    """Task objects, interaction roles and anchor from the task catalog and the source task program."""
    cat = read(CATALOG / f'{task}.json')
    labels = [e['label'] for e in cat['entities'] if e['label'] and e['type'] == 'Rigid']
    try:
        ops = read(Path(source) / 'program.json').get('program', {}).get('operations', [])
    except FileNotFoundError:
        ops = []
    roles = {o['object']: {'object': o['object'], **({'target': o['target_object']} if o.get('target_object') else {})} for o in ops}
    if task == 'stack_blocks':
        roles = {x: dict(object=x, target='block_1') for x in ['block_0', 'block_2']}
    if task == 'insert_tubes':
        roles = {x: dict(object=x, target='slot') for x in labels}
    if not roles:
        raise ValueError('Task roles not defined: ' + task)
    return dict(task=task, objects=labels, roles=list(roles.values()), anchor=labels[0], task_sha256=cat['task_sha256'],
                terminal_rule='last GT-defined interaction per manipulated object; receiver when specified, otherwise initial-object displacement',
                trajectory_aggregation='equal mean over named native Rigid task objects, excluding unlabeled clutter',
                scene_aggregation='equal mean over final per-object interaction roles; common initial-anchor translation, shared ' + M.SCENE5_REVISION,
                progress_rule='maximum native stage score / 100; binary native success only when task defines no stage score')


# ---------------------------------------------------------------- scene geometry
def export_scene(record, dest):
    """Exported recording geometry as a metric scene (one PLY per entity)."""
    import trimesh
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True); scene = dict(objects=[], props=[], support=None)
    for e in read(Path(record) / 'geometry/manifest.json')['objects']:
        if not e.get('vertices') or not e.get('faces'):
            continue
        with np.load(Path(record) / 'geometry' / e['directory'] / 'geometry.npz') as z:
            vertices = z['vertices']; counts = z['face_vertex_counts']; indices = z['face_vertex_indices']
        faces = []; off = 0
        for n in counts:
            f = indices[off:off + n]; off += n
            faces.extend([f[0], f[j], f[j + 1]] for j in range(1, len(f) - 1))
        name = e['label']; trimesh.Trimesh(vertices, faces, process=False).export(dest / (name + '.ply'))
        if 'root_pose' in e:
            pose = e['root_pose']
        else:
            t = np.array(e['root_matrix']); q = Rotation.from_matrix(t[:3, :3]).as_quat(); pose = [*t[:3, 3], q[3], *q[:3]]
        obj = dict(name=name, kind='mesh', mesh_path=name + '.ply', pos=pose[:3], quat=pose[3:])
        scene['objects' if name.startswith('block_') else 'props'].append(obj)
    write(dest / 'scene.json', scene)
    return scene


def categorized(record, dest, labels):
    s = export_scene(record, dest); es = s['objects'] + s['props']
    s['objects'] = [e for e in es if e['name'] in labels]; s['props'] = [e for e in es if e['name'] not in labels]
    return s


def shift(scene, delta):
    result = copy.deepcopy(scene)
    for e in result.get('objects', []) + result.get('props', []):
        e['pos'] = (np.asarray(e['pos']) + delta).tolist()
    if result.get('support'):
        result['support']['z'] += float(delta[2])
        result['support']['center'] = (np.asarray(result['support'].get('center', [0, 0])) + delta[:2]).tolist()
    return result


def anchor_scene(scene, gt, mapping, anchor):
    """One common translation bringing the candidate anchor object onto the GT anchor."""
    entities = lambda s: s.get('objects', []) + s.get('props', [])
    gp = np.array(next(e for e in entities(gt) if e['name'] == anchor)['pos'])
    cp = np.array(next(e for e in entities(scene) if e['name'] == mapping[anchor])['pos'])
    return shift(scene, gp - cp), dict(translation_m=(gp - cp).tolist(), anchor_gt=anchor, anchor_candidate=mapping[anchor])


# ---------------------------------------------------------------- source TCP context (X5 forward kinematics)
def _transform(xyz=(0., 0., 0.), rpy=(0., 0., 0.)):
    result = np.eye(4); result[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix(); result[:3, 3] = xyz
    return result


class SerialArm:
    """URDF chain from the root link to a tip link; ``fk`` returns world-from-TCP."""

    def __init__(self, urdf_path, tip_link, world_from_base=None, tip_from_tcp=None):
        root = ET.parse(str(urdf_path)).getroot()
        by_child = {j.find('child').get('link'): j for j in root.findall('joint')}
        if tip_link not in {l.get('name') for l in root.findall('link')}:
            raise ValueError(f'Unknown tip link: {tip_link}')
        chain, current = [], tip_link
        while current in by_child:
            joint = by_child[current]; chain.append(joint); current = joint.find('parent').get('link')
        chain.reverse()
        self.world_from_base = np.eye(4) if world_from_base is None else np.array(world_from_base, float)
        self.tip_from_tcp = np.eye(4) if tip_from_tcp is None else np.array(tip_from_tcp, float)
        self.joint_names, self._chain = [], []
        for joint in chain:
            kind = joint.get('type')
            if kind not in ('fixed', 'revolute', 'continuous', 'prismatic'):
                raise ValueError(f'Unsupported joint type: {kind}')
            origin = joint.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            axis_tag = joint.find('axis')
            axis = np.fromstring(axis_tag.get('xyz', '1 0 0'), sep=' ') if axis_tag is not None else np.array([1., 0., 0.])
            axis = axis / np.linalg.norm(axis)
            index = None
            if kind != 'fixed':
                index = len(self.joint_names); self.joint_names.append(joint.get('name'))
            self._chain.append((_transform(xyz, rpy), kind, axis, index))

    def fk(self, q):
        q = np.asarray(q, float)
        pose = self.world_from_base.copy()
        for origin, kind, axis, index in self._chain:
            pose = pose @ origin
            if index is not None:
                motion = np.eye(4)
                if kind == 'prismatic':
                    motion[:3, 3] = axis * q[index]
                else:
                    motion[:3, :3] = Rotation.from_rotvec(axis * q[index]).as_matrix()
                pose = pose @ motion
        return pose @ self.tip_from_tcp


def x5_arm(urdf_path, base_pos, base_quat_wxyz):
    """ARX X5 arm with its pad-centre TCP (link6 +x 0.1501 m; TCP +z = link6 +x)."""
    base = _transform(base_pos); w, x, y, z = base_quat_wxyz
    base[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    tcp = np.eye(4); tcp[:3, 3] = (0.1501, 0., 0.)
    tcp[:3, :3] = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    return SerialArm(urdf_path, 'link6', world_from_base=base, tip_from_tcp=tcp)


def tcp_context(source, geometry_context, dest):
    """Context-weight inputs: GT scene + both source TCP streams from measured joints (X5 FK)."""
    import shutil
    source, geometry_context, dest = map(Path, [source, geometry_context, dest])
    shutil.copytree(geometry_context / 'gt_pkg', dest / 'gt_pkg')
    (dest / 'hidden').mkdir()
    shutil.copyfile(geometry_context / 'hidden/objects_6d.npz', dest / 'hidden/objects_6d.npz')
    urdf = paths.asset('robodojo', 'Assets/Robots/x5/X5A.urdf')
    z = np.load(source / 'trajectory.npz'); names = read(source / 'robot_joints.json'); robots = read(source / 'resolved_config.json')['robot']['robots']
    poses = []
    for side, robot in enumerate(robots):
        arm = x5_arm(urdf, robot['default_root_pos'], robot['default_root_rot'])
        indices = [names[side].index(n) for n in arm.joint_names]
        q = z['robot_joint_positions'][:, side][:, indices]
        transforms = np.array([arm.fk(row) for row in q]); xyzw = Rotation.from_matrix(transforms[:, :3, :3]).as_quat()
        poses.append(np.c_[transforms[:, :3, 3], xyzw[:, 3], xyzw[:, :3]])
    np.savez_compressed(dest / 'hidden/trajectory.npz', tcp_pose=np.concatenate(poses),
                        tcp_side=np.repeat([0, 1], len(poses[0])), tcp_time_s=np.tile(z['video_time_s'], 2))
    return dict(method='measured source joint states -> X5 URDF FK -> pad-center TCP', frames_per_side=len(poses[0]))


def prepare_source(task, source, out):
    """GT metric scene, GT object trajectories and context weights of the source recording."""
    out = Path(out)
    spec = definition(task, source); gdir = out / 'gt_pkg'; gs = categorized(source, gdir, spec['objects'])
    z = np.load(source / 'trajectory.npz', allow_pickle=False); names = list(z['object_labels'])
    for e in gs['objects'] + gs['props']:
        if e['name'] in names:
            v = z['object_pose_wxyz'][0, names.index(e['name'])]; e['pos'] = v[:3].tolist(); e['quat'] = v[3:].tolist()
    write(gdir / 'scene.json', gs); (out / 'hidden').mkdir(exist_ok=True)
    np.savez_compressed(out / 'hidden/objects_6d.npz', **{n: z['object_pose_wxyz'][:, i] for i, n in enumerate(names)})
    tcp = tcp_context(source, out, out / 'context')
    c = read(source / 'camera.json')
    camera = dict(intrinsics=c['intrinsics'], extrinsics_base_cam=c['camera_to_world_ros'], height=c.get('height', 480), width=c.get('width', 640))
    weights = {r['object']: M.context_weights(out / 'context', gs, r) for r in spec['roles']}
    result = dict(spec=spec, camera=camera, weights=weights, source=str(source), tcp=tcp)
    write(out / 'prepared.json', result)
    return result


# ---------------------------------------------------------------- trajectory aggregation
def _mean_block(values, name):
    if all(v[name] is None for v in values):
        return None
    if any(v[name] is None for v in values):
        raise ValueError('Mixed metric availability: ' + name)
    out = {}
    for key, first in values[0][name].items():
        xs = [v[name][key] for v in values]
        if key in CONSTANT or isinstance(first, str) or first is None:
            if any(x != first for x in xs):
                raise ValueError('Metric contract differs: ' + name + '.' + key)
            out[key] = first
        elif isinstance(first, bool):
            out[key] = any(xs)
        else:
            m = np.asarray(xs, float).mean(axis=0); out[key] = m.tolist() if m.ndim else float(m)
    return out


def aggregate(per_object):
    """Equal mean of per-object trajectory metric blocks; protocol fields must agree."""
    if not per_object:
        raise ValueError('No per-object trajectory metrics')
    values = list(per_object.values()); out = {}
    for key, first in values[0].items():
        if key in ('ape', 'rpe'):
            out[key] = _mean_block(values, key)
        elif isinstance(first, dict):
            out[key] = aggregate({name: v[key] for name, v in per_object.items()})
        else:
            if any(v[key] != first for v in values):
                raise ValueError('Objects do not share one clock/protocol: ' + key)
            out[key] = first
    return out


# ---------------------------------------------------------------- outcome policies
def apply_environment_failure(result, native, audit, run_status, frames):
    """A completed recording whose native environment declared itself invalid is a task failure."""
    if native.get('environment_valid') is not False:
        raise ValueError('Expected explicit native environment_valid=false')
    if (run_status.get('status') != 'recorded' or native.get('status') != 'complete' or type(frames) is not int or frames < 1
            or any(type(n) is not int or n != frames for n in [run_status.get('frames'), native.get('frames'), audit.get('frames'), audit.get('video_frames')])
            or audit.get('status') != 'pass' or audit.get('invalid_instance_mask_frames') != []
            or audit.get('all_scene_instances_mapped') is not True or audit.get('render_gt_alignment_verified') is not True):
        raise ValueError('Environment failure requires complete recorded frames and passing recording/render audit')
    history = native.get('history')
    if not isinstance(history, list) or len(history) != frames:
        raise ValueError('Environment failure requires complete native predicate history')
    if any(row.get('frame') != i or type(row.get('environment_valid')) is not bool for i, row in enumerate(history)):
        raise ValueError('Incomplete native environment verdict timeline')
    if result.get('build') is not True:
        raise ValueError('Environment outcome requires a built execution')
    result.setdefault('task_success_raw', result.get('task_success'))
    result.setdefault('raw_progress', (result.get('progress') or {}).get('progress'))
    result['task_success'] = False
    result['progress'] = dict(result.get('progress') or {}, progress=0., rejection='native_environment_invalid')
    result['task_failure_reason'] = 'native_environment_invalid'
    result['native_environment_valid'] = False
    result['execution_validity'] = dict(valid=False, scorable=True, reason='native_environment_invalid')
    result['physical_validity'] = dict(valid=False, scope='native task environment rejected; completed recording retained')
    return result


def apply_mimic_outcome(result, audit):
    """A completed recording that violates the gripper mimic constraints is a task failure."""
    if audit.get('status') != 'pass' or audit.get('render_gt_alignment_verified') is not True:
        raise ValueError('Recording/render integrity failure is not a mimic task failure')
    if type(audit.get('mimic_constraints_passed')) is not bool:
        raise ValueError('Missing boolean mimic audit verdict')
    result['control_audit'] = audit
    if not audit['mimic_constraints_passed']:
        if result.get('build') is not True:
            raise ValueError('Mimic outcome policy requires a built, completed execution')
        result.setdefault('task_success_raw', result.get('task_success'))
        result.setdefault('raw_progress', (result.get('progress') or {}).get('progress'))
        result['task_success'] = False
        result['progress'] = dict(result.get('progress') or {}, progress=0., rejection='mimic_constraints_failed')
        result['task_failure_reason'] = 'mimic_constraints_failed'
        result['execution_validity'] = dict(valid=False, scorable=True, reason='mimic_constraints_failed')
        result['physical_validity'] = dict(valid=False, scope='recorded mimic constraint violation')
    return result


# ---------------------------------------------------------------- record
def submitted_poses(scene):
    return {e['name']: e['pos'] + e['quat_wxyz'] for e in scene['objects'] + scene.get('props', []) + ([scene['table']] if scene.get('table') else [])}


def score(task, sample, record, source_prepared, out, submitted_scene):
    """Full metric record of a built execution."""
    from . import isaac
    sample, record, source_prepared, out = map(Path, [sample, record, source_prepared, out]); out.mkdir(parents=True, exist_ok=False)
    prep = read(source_prepared / 'prepared.json'); spec = prep['spec']; source = Path(prep['source'])
    gs = read(source_prepared / 'gt_pkg/scene.json'); cs = categorized(record, out / 'geometry_candidate', spec['objects'])
    delivered = submitted_poses(submitted_scene)
    for e in cs['objects'] + cs['props']:
        if e['name'] in delivered:
            e['pos'] = delivered[e['name']][:3]; e['quat'] = delivered[e['name']][3:]
    mapping = {n: n for n in spec['objects']}
    aligned, frame = anchor_scene(cs, gs, mapping, spec['anchor'])
    terms = {r['object']: M.scene_cd_v5(aligned, out / 'geometry_candidate', gs, source_prepared / 'gt_pkg', prep['camera'], roles=r,
                                        weights=prep['weights'][r['object']]) for r in spec['roles']}
    for term in terms.values():
        if term.get('error') or term.get('metric_valid') is False:
            raise ValueError('Scene metric invalid: ' + str(term))
    mean = lambda key: float(np.mean([v[key] for v in terms.values()])) if all(v.get(key) is not None for v in terms.values()) else None
    first = next(iter(terms.values()))
    scene = dict(revision=first['revision'], version=first['version'], metric_valid=True,
                 **{k: mean(k) for k in ['scene_cd_cm', 'scene_cd_unaligned_cm', 'table_depth_err_cm', 'missing_fraction', 'extra_fraction']},
                 alignment=first.get('alignment', {}), table=first.get('table', {}), per_interaction=terms, coordinate_preparation=frame,
                 aggregation=spec['scene_aggregation'], surface_sampling=first.get('surface_sampling'))
    a = np.load(source / 'trajectory.npz', allow_pickle=False); b = np.load(record / 'trajectory.npz', allow_pickle=False)
    an = list(a['object_labels']); bn = list(b['object_labels'])
    dt = float(np.median(np.diff(b['video_time_s']))); gdt = float(np.median(np.diff(a['video_time_s'])))
    traj = {n: demo_clock_metrics(b['object_pose_wxyz'][:, bn.index(n)], dt, a['object_pose_wxyz'][:, an.index(n)], gdt, symmetry=None, axis=None)
            for n in spec['objects']}
    own = dict(aggregate(traj), per_object=traj, aggregation=spec['trajectory_aggregation'])
    per_terminal = {}
    for r in spec['roles']:
        n = r['object']; g = a['object_pose_wxyz'][:, an.index(n)]; c = b['object_pose_wxyz'][:, bn.index(n)]; target = r.get('target')
        per_terminal[n] = terminal_position(dict(type='in_region', **({'target': target} if target else {})), c[0], c[-1], g,
                                            b['object_pose_wxyz'][-1, bn.index(target)] if target else None,
                                            a['object_pose_wxyz'][-1, an.index(target)] if target else None, axis=None)
    terminal = dict(protocol='terminal-position/1.0', source='own_actions', stage='terminal', anchor='per_object',
                    rel_trans_err_cm=float(np.mean([v['rel_trans_err_cm'] for v in per_terminal.values()])),
                    per_object=per_terminal, aggregation=spec['terminal_rule'])
    ge = {e['name']: e for e in gs['objects']}; ce = {e['name']: e for e in cs['objects']}
    objects = {n: M.object_level(ce[n], out / 'geometry_candidate', ce[n]['pos'] + ce[n]['quat'], ge[n], source_prepared / 'gt_pkg', ge[n]['pos'] + ge[n]['quat'])
               for n in spec['objects']}
    native = read(record / 'task_result.json')
    success = bool(native['task_success']); vals = [x['score'] for x in native['history'] if x.get('score') is not None]
    progress = max(vals) / 100 if vals else float(success)
    # Replay the native object goals on the recorded states: an initially satisfied goal needs exit and re-entry.
    transition_path = out / 'completion_transition.json'
    task_file = sample / 'reference/native_task.py'
    if not task_file.exists():
        raise ValueError('Missing bound native task for completion replay')
    dolls = None
    if task == 'sort_nesting_dolls_by_size':
        from .nesting_dolls import rescore
        dolls = rescore(record)
        write(out / 'nesting_dolls_rescore.json', dolls); write(transition_path, dolls['completion_transition'])
        success = dolls['task_success']; progress = dolls['progress']
    else:
        subprocess.run(isaac.command('completion', '--record', record, '--task-file', task_file, '--task', task, '--out', transition_path),
                       check=True, timeout=120, env=isaac.environment())
    transition = read(transition_path)
    native_success = native.get('native_task_success', success)
    native_progress = max((x.get('native_score', x.get('score')) for x in native['history'] if x.get('native_score', x.get('score')) is not None),
                          default=100 * progress) / 100
    if not transition['completion_allowed']:
        success = False; progress = 0.
    audit = read(record / 'audit.json')
    environment_failed = native.get('environment_valid') is False
    valid = audit['status'] == 'pass' and audit['render_gt_alignment_verified']
    if not valid:
        raise ValueError('Invalid execution cannot be converted to task failure')
    if not 0 <= progress <= 1:
        raise ValueError('Native stage score outside [0,100]')
    result = dict(schema_version='eval2/4', robodojo_metric_protocol=VERSION, sample=sample.name, task=task,
                  build=True, task_success=success, task_success_raw=native_success, raw_progress=native_progress, completion_transition=transition,
                  execution_validity={'valid': True}, physical_validity={'valid': None, 'scope': 'sampled execution checks; no full contact/collision certification'},
                  object={k: float(np.mean([v[k] for v in objects.values()])) for k in ['shape_cd_cm', 'size_err_cm', 'center_err_cm', 'origin_err_cm']},
                  objects=objects, own_trajectory=own, relative=terminal, progress={'progress': float(progress), 'definition': spec['progress_rule']},
                  metric_status={'task_success': {'status': 'ok'}, 'relative.rel_trans_err_cm': {'status': 'ok', 'version': 'terminal-position/1.0'}},
                  task_spec=spec)
    if dolls is not None:
        result['nesting_dolls_rescore'] = dolls
        result['task_success_raw'] = dolls['task_success']; result['raw_progress'] = dolls['progress']
    apply_mimic_outcome(result, audit)
    if environment_failed:
        apply_environment_failure(result, native, audit, read(record / 'run.json'), len(b['video_time_s']))
    result['object']['size_protocol'] = 'intrinsic-obb-full-extents/2'
    set_scene_result(result, scene)
    write(out / 'eval2.json', result)
    return result
