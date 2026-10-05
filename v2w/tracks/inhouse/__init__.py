"""In-house track: household episodes of the AgiBot G1 + OmniPicker robot, executed natively in Isaac Sim.

Evaluators (``python -m v2w.tracks.inhouse.evaluate --kind ...``) write ``eval.json`` whose ``eval2`` block is the
metric record; ``metrics`` returns it.
"""
import json
from pathlib import Path


def metrics(family, sample_dir, eval_dir):
    r = json.loads((Path(eval_dir) / 'eval.json').read_text())
    if isinstance(r.get('eval2'), dict) and r['eval2']:
        result = dict(r['eval2'])
        if r.get('error_kind'):
            result['error_kind'] = r['error_kind']
        return result
    if r.get('error_kind'):
        return dict(sample=Path(sample_dir).name, build=r.get('build'), error_kind=r['error_kind'], error=r.get('error'))
    raise ValueError('evaluator wrote no metric record')
