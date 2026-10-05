"""Cloth folding track (DROID towel folds): MuJoCo cloth execution scored against the visible-surface GT."""
import json
from pathlib import Path


def metrics(family, sample_dir, eval_dir):
    """The metric record the evaluator wrote under eval.json['eval2']."""
    path = Path(eval_dir) / 'eval.json'
    if not path.is_file():
        return dict(sample=Path(sample_dir).name, build=None, error_kind='evaluator', error='no cloth evaluation report')
    report = json.loads(path.read_text())
    record = dict(report.get('eval2') or {})
    if not record:
        return dict(sample=Path(sample_dir).name, build=report.get('build'), error_kind=report.get('error_kind', 'evaluator'),
                    error=report.get('error', 'cloth evaluation report has no metric record'))
    if report.get('error_kind'):
        record['error_kind'] = report['error_kind']
    return record
