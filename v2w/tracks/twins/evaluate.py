"""Evaluate a twin_mujoco_v2 package (rope routing, toy packing): audit its physics, execute it in MuJoCo and score the
executed states against the sample's hidden twin reference.

Usage: python -m v2w.tracks.twins.evaluate --run PKG --sample SAMPLE_DIR --out EVAL_DIR/eval.json
A package fault (invalid model, unstable simulation, missing receiver) is recorded as a failed build attributed to the
method; any other failure leaves no eval.json and is reported by the runner as an evaluator error.
"""
import argparse
import json
from pathlib import Path

from v2w.tracks.twins.task import audit_model, score

PROFILE = 'twin_mujoco_v2'


def evaluate(pkg, sd, out, prefix=0.):
    from video2sim.native_twin import rollout
    pkg, sd, out = (Path(p).resolve() for p in (pkg, sd, out))
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        protocol = json.loads((pkg / 'protocol.json').read_text())
        if not isinstance(protocol, dict):
            raise ValueError('protocol.json must be an object')
        if protocol.get('physics_profile') != PROFILE:
            raise ValueError(f'physics_profile must be {PROFILE}')
        audit = audit_model(pkg, sd, 'rope' if 'rope' in sd.name else 'toy')
        execution = rollout(pkg, out.parent / 'candidate_execution')
        metrics = score(sd, execution['executed_states'], audit, prefix)
        summary = {k: v for k, v in execution.items() if k not in ('points', 'nodes', 'edges', 'time_s', 'initial_target')}
        result = dict(build=True, task_success=metrics['task_success'], progress=metrics['progress'], eval2=metrics,
                      execution=dict(summary, profile=PROFILE))
    except (ValueError, KeyError, TypeError) as exc:
        metric = dict(schema_version='eval2/4', sample=sd.name, build=False, task_success=False, error_kind='agent', error=str(exc))
        result = dict(build=False, task_success=False, error_kind='agent', error=str(exc), eval2=metric)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--sample', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--prefix-hold-s', type=float, default=0.)
    a = p.parse_args()
    r = evaluate(a.run, a.sample, a.out, a.prefix_hold_s)
    print(json.dumps(dict(sample=Path(a.sample).name, build=r['build'], success=r['task_success'], error=r.get('error'))), flush=True)


if __name__ == '__main__':
    main()
