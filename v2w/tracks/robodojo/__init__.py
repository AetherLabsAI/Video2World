"""RoboDojo track: dual-arm (ARX X5 / xArm7) tabletop tasks executed with the native RoboDojo tasks in Isaac Sim.

Evaluator entry point: ``python -m v2w.tracks.robodojo.evaluate``. The evaluator writes the complete metric record
into eval.json; ``metrics`` returns it.
"""
import json
from pathlib import Path


def metrics(family, sample_dir, eval_dir):
    """Metric record of an evaluated package (``eval.json['eval2']``)."""
    report = json.loads((Path(eval_dir) / 'eval.json').read_text())
    if isinstance(report.get('eval2'), dict) and report['eval2']:
        result = dict(report['eval2'])
        if report.get('error_kind'):
            result['error_kind'] = report['error_kind']
        return result
    if report.get('error_kind'):
        return dict(sample=Path(sample_dir).name, build=report.get('build'), error_kind=report['error_kind'], error=report.get('error'))
    raise ValueError('eval.json carries no metric record')
