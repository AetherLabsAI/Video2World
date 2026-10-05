"""Task-local size semantics from geometry exported from the executed USD stage.

Object names identify the five actors only. Neither array order, model_id nor
submitted bounding-box metadata supplies their size. Scale is already baked
by export_scene_geometry; applying objects.json scale again would be wrong.
"""
from pathlib import Path
import hashlib
import json
import types

import numpy as np

VERSION = 'nesting-dolls-executed-size/1'
TASK = 'sort_nesting_dolls_by_size'
LABELS = tuple(f'doll{i}' for i in range(5))
SIZE_EPS_M = 1e-6  # Numerical ambiguity, not a task-placement tolerance.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def size_order(vertices_by_label):
    """Ascending upright local-z extent, with no arbitrary tie breaking."""
    if set(vertices_by_label) != set(LABELS):
        raise ValueError('Exactly five named doll geometries are required')
    extents = {}
    for label, vertices in vertices_by_label.items():
        v = np.asarray(vertices, dtype=float)
        if v.ndim != 2 or v.shape[1] != 3 or len(v) < 4 or not np.isfinite(v).all():
            raise ValueError('Invalid executed geometry: ' + label)
        e = np.ptp(v, axis=0)
        if np.any(e <= 0):
            raise ValueError('Degenerate executed geometry: ' + label)
        extents[label] = e.tolist()
    ordered = sorted(LABELS, key=lambda label: extents[label][2])
    heights = np.array([extents[label][2] for label in ordered])
    distinct = bool(np.all(np.diff(heights) > SIZE_EPS_M))
    return dict(protocol=VERSION, ordered_labels=ordered if distinct else [],
                distinct_sizes=distinct, extent_m=extents,
                size_definition='upright local-z extent of executed geometry; root scale baked once',
                ambiguity_tolerance_m=SIZE_EPS_M)


def from_record(record):
    record = Path(record)
    manifest = record / 'geometry/manifest.json'
    entries = json.loads(manifest.read_text())['objects']
    vertices = {}; inputs = {str(manifest): sha(manifest)}
    for item in entries:
        label = item.get('label')
        if label not in LABELS:
            continue
        if label in vertices:
            raise ValueError('Duplicate executed doll geometry: ' + label)
        path = (manifest.parent / item['directory'] / 'geometry.npz').resolve()
        if not path.is_relative_to(manifest.parent.resolve()):
            raise ValueError('Geometry escapes execution record')
        with np.load(path, allow_pickle=False) as z:
            vertices[label] = z['vertices']
        inputs[str(path)] = sha(path)
    result = size_order(vertices)
    result['inputs'] = inputs
    return result


def bind(env, record):
    """Replace only this task's ranking; keep its original native predicates."""
    if type(env).__name__ != TASK:
        return None
    if env.num_envs != 1:
        raise ValueError('Recorded geometry is bound to exactly one environment')
    evidence = from_record(record)
    def ordered(self, func_parser, labels):
        if set(labels) != set(LABELS):
            raise ValueError('Unexpected nesting-doll labels')
        # The explicit false predicate below rejects ties. This fallback is only
        # to let the native checks register; it cannot grant success.
        return [evidence['ordered_labels'] or list(LABELS)]
    env._get_ordered_labels_per_env = types.MethodType(ordered, env)
    original_reward = env.run_reward
    def run_reward(self):
        original_reward()
        self.reward_manager.check_list[0][-1].append(('nesting_dolls_distinct_sizes', {}))
    env.run_reward = types.MethodType(run_reward, env)
    env.reward_manager.func_parser.nesting_dolls_distinct_sizes = (
        lambda args: float(evidence['distinct_sizes']))
    return evidence


def object_goal(poses, labels, ranking):
    """Same native geometric thresholds, evaluated on recorded rigid poses."""
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 3 or poses.shape[2] != 7 or not np.isfinite(poses).all():
        raise ValueError('Invalid object trajectory')
    if len(labels) != len(set(labels)):
        raise ValueError('Duplicate recorded object labels')
    if not ranking['distinct_sizes']:
        return np.zeros(len(poses), dtype=bool)
    p = poses[:, [list(labels).index(n) for n in ranking['ordered_labels']]]
    q = p[:, :, 3:]
    norm = np.linalg.norm(q, axis=2)
    if np.any(np.abs(norm - 1) > .002):
        raise ValueError('Invalid recorded quaternion')
    q = q / norm[:, :, None]
    # Native is_axis_up uses a strict angle threshold of 15 degrees.
    upright = (1 - 2 * (q[:, :, 1] ** 2 + q[:, :, 2] ** 2)) > np.cos(np.deg2rad(15))
    return ((np.diff(p[:, :, 0], axis=1) > .03).all(axis=1)
            & (np.ptp(p[:, :, 1], axis=1) <= .035) & upright.all(axis=1))
