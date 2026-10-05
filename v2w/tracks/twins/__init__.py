"""Reconstructed-twin tracks (rope routing, toy packing) on the shared xArm7 gripper in MuJoCo."""
import json
from pathlib import Path


def metrics(family, sample_dir, eval_dir):
    """Metric record: the evaluator already writes it under eval.json['eval2']."""
    sd, path = Path(sample_dir), Path(eval_dir) / 'eval.json'
    report = json.loads(path.read_text()) if path.exists() else None
    if not report:
        return dict(sample=sd.name, error='no twin eval report')
    record = {k: v for k, v in (report.get('eval2') or {}).items() if not (k in ('error', 'error_kind') and v is None)}
    record.setdefault('sample', sd.name)
    record['build'] = bool(report.get('build'))
    record['task_success'] = bool(report.get('task_success'))
    return record
