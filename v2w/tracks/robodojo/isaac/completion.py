"""Replay the native task's object goals on a finished recording (no physics) to apply the completion-transition rule.

Runs in the Isaac interpreter with the RoboDojo checkout importable (reward manager and task code); object poses come
from the recording. Writes the CompletionTransition result as JSON.
Usage: python -m v2w.tracks.robodojo.isaac.completion --record DIR --task-file native_task.py --task NAME --out OUT.json
"""
import ast
import copy
import json
import sys
import typing  # noqa: F401
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation  # noqa: F401

from . import source_root

SRC = source_root()
sys.path.insert(0, str(SRC))
# The native layout helpers bound below run in this module's namespace and use these names.
import torch  # noqa: E402,F401
import transforms3d as t3d  # noqa: E402,F401
from utils.transformer import *  # noqa: E402,F401,F403  (task code expects these helpers in its namespace)
from env.reward_manager.reward_manager import RewardManager  # noqa: E402

from v2w.tracks.robodojo.isaac.runtime import CompletionTransition  # noqa: E402
from v2w.tracks.robodojo.nesting_dolls import bind as bind_doll_sizes  # noqa: E402


class Layout:
    """Read-only layout manager serving recorded object poses (frame -1: the pre-action snapshot)."""

    def __init__(self, record, z):
        self.records = json.loads((record / 'objects.json').read_text())
        self.by_name = {x['inst_name']: x for x in self.records}; self.by_label = {x['label']: x for x in self.records}
        self.z = z; self.labels = list(z['object_labels']); self.frame = 0
        self.initial_poses = {x['label']: x.get('root_pose') for x in self.records}
        self.instance_type_by_env = [{x['inst_name']: x['object_type'].lower() for x in self.records}]

    def get_instance_name(self, env_idx=0, label=None):
        return self.by_label[label]['inst_name'] if label in self.by_label else None

    def get_instance_pose(self, env_idx=0, label=None, inst_name=None, relative=True):
        obj = self.by_name[inst_name] if inst_name else self.by_label[label]
        p = np.asarray(self.initial_poses[obj['label']], float) if self.frame < 0 else self.z['object_pose_wxyz'][self.frame, self.labels.index(obj['label'])]
        if p.shape != (7,) or not np.isfinite(p).all():
            raise ValueError('Missing/nonfinite pre-action object pose: ' + obj['label'])
        return p[:3].copy(), p[3:].copy()

    def get_instance_metadata(self, env_idx=0, inst_name=None, label=None):
        return copy.deepcopy((self.by_name[inst_name] if inst_name else self.by_label[label])['metadata'])

    def get_scene_object(self, env_idx=0, inst_name=None):
        return SimpleNamespace(record=self.by_name[inst_name]) if inst_name in self.by_name else None

    def get_instance_bbox_vertices(self, inst_name, env_idx=0):
        return np.asarray(self.by_name[inst_name]['metadata']['geometry']['oriented_bbox']['vertices'])

    def get_layout_records(self, env_idx, typ):
        return [x for x in self.records if x['object_type'] == typ]

    def get_labels_by_prefix(self, prefix, env_idx=0):
        return [x for x in self.by_label if x.startswith(prefix)]


# Pure geometric helpers of the native LayoutManager, bound unchanged onto the read-only layout.
_layout_file = SRC / 'env/scene_manager/layout_manager.py'
_cls = next(x for x in ast.parse(_layout_file.read_text()).body if isinstance(x, ast.ClassDef) and x.name == 'LayoutManager')
for _fn in _cls.body:
    if isinstance(_fn, ast.FunctionDef) and _fn.name in ['to_transformation_matrix', 'get_functional_points', 'get_support_points']:
        _module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), _fn], type_ignores=[])
        ast.fix_missing_locations(_module); _space = dict(globals()); exec(compile(_module, str(_layout_file), 'exec'), _space)
        setattr(Layout, _fn.name, _space[_fn.name])


def replay(record, task_file, task):
    record = Path(record); z = dict(np.load(record / 'trajectory.npz', allow_pickle=False)); lm = Layout(record, z)
    lm.frame = -1
    snapshot = json.loads((record / 'completion_initial_state.json').read_text())
    if snapshot.get('epoch') != 'pre_action':
        raise ValueError('Invalid native initial-state epoch')
    lm.initial_poses = snapshot['object_poses_wxyz']
    for label in lm.labels:   # missing initial evidence must never silently become a post-action frame
        pose = np.asarray(lm.initial_poses.get(label), float)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError('Missing native pre-action pose: ' + str(label))
    tree = ast.parse(Path(task_file).read_text())
    tree.body = [x for x in tree.body if not (isinstance(x, ast.ImportFrom) and x.module == 'env.environment.task_env')]
    space = {'TaskEnv': type('ReadOnlyTaskBase', (), {}), '__name__': 'completion_replay'}
    exec(compile(tree, str(task_file), 'exec'), space)
    env = object.__new__(space[task]); env.num_envs = 1; env.success = [True]; env.eval_seed = 0
    env.scene_manager = SimpleNamespace(layout_manager=lm, env_origins=np.zeros((1, 3))); env.robot_manager = SimpleNamespace(robot_list=[])
    env.reward_manager = RewardManager(1); env.reward_manager.initialize(env); env.reward_manager.init_state()
    doll_sizes = bind_doll_sizes(env, record)
    env.run_reward(); guard = CompletionTransition(env.reward_manager)
    if guard.initial_goal:   # a task not complete at the start keeps its native temporal semantics
        for frame in range(0, len(z['object_pose_wxyz'])):
            lm.frame = frame; guard.observe()
            if guard.allowed:
                break
    result = guard.result()
    if doll_sizes is not None:
        result['nesting_dolls_geometry'] = doll_sizes
    return result


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--record', required=True); p.add_argument('--task-file', required=True)
    p.add_argument('--task', required=True); p.add_argument('--out', required=True)
    a = p.parse_args()
    Path(a.out).write_text(json.dumps(replay(a.record, a.task_file, a.task), indent=2) + '\n')
