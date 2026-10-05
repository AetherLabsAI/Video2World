"""The benchmark definition: configurations from the data manifest and the evaluator registry."""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache

from . import paths

# evaluator -> (interpreter tool, entry module, extra arguments)
EVALUATORS = {
    'fb': ('python', 'v2w.tracks.furniture.evaluate', []),
    'ego_arm': ('python', 'v2w.tracks.hand.evaluate', ['--embodiment', 'arm']),
    'ego_bimanual': ('python', 'v2w.tracks.hand.evaluate', ['--embodiment', 'dexhand_bimanual']),
    'robodojo_native_task': ('python', 'v2w.tracks.robodojo.evaluate', []),
    'inhouse': ('python', 'v2w.tracks.inhouse.evaluate', ['--kind', 'camera']),
    'inhouse_point_robot': ('python', 'v2w.tracks.inhouse.evaluate', ['--kind', 'point']),
    'inhouse_reviewed': ('python', 'v2w.tracks.inhouse.evaluate', ['--kind', 'reviewed']),
    'inhouse_wallet': ('python', 'v2w.tracks.inhouse.evaluate', ['--kind', 'wallet']),
    'inhouse_fryer': ('python', 'v2w.tracks.inhouse.evaluate', ['--kind', 'fryer']),
    'twin': ('twin_python', 'v2w.tracks.twins.evaluate', []),
    'twin_toy': ('twin_python', 'v2w.tracks.twins.evaluate', []),
    'cloth': ('twin_python', 'v2w.tracks.cloth.evaluate', []),
}
TRACKS = {'fb': 'furniture', 'ego_arm': 'hand', 'ego_bimanual': 'hand', 'robodojo_native_task': 'robodojo',
          'inhouse': 'inhouse', 'inhouse_point_robot': 'inhouse', 'inhouse_reviewed': 'inhouse', 'inhouse_wallet': 'inhouse',
          'inhouse_fryer': 'inhouse', 'twin': 'twins', 'twin_toy': 'twins', 'cloth': 'cloth'}


@dataclass(frozen=True)
class Configuration:
    family: str
    sample: str
    spec: dict          # evaluator, agent profile and GT metric contract

    @property
    def sample_dir(self):
        return paths.sample(self.sample)

    @property
    def evaluator(self):
        return self.spec.get('evaluator') or self.spec.get('eval2')

    @property
    def track(self):
        return TRACKS[self.evaluator]

    @property
    def profile(self):
        return self.spec.get('profile') or ('droid_cloth' if self.evaluator == 'cloth' else None)

    @property
    def key(self):
        return f'{self.family}/{self.sample}'


@lru_cache(maxsize=1)
def manifest():
    path = paths.DATA / 'manifest.json'
    if not path.exists():
        raise FileNotFoundError(f'no data release at {paths.DATA}: download it there or set V2W_DATA (see README)')
    return json.loads(path.read_text())


def configurations(families=None, samples=None):
    out = []
    for row in manifest()['samples']:
        if families and row['family'] not in families:
            continue
        if samples and row['sample'] not in samples:
            continue
        out.append(Configuration(row['family'], row['sample'], row['metric_family']))
    return out


def families():
    groups = OrderedDict()
    for c in configurations():
        groups.setdefault(c.family, []).append(c)
    return groups


def scope_rows(configs=None):
    return [dict(family=c.family, sample=c.sample, metric_family=c.spec) for c in (configs or configurations())]
