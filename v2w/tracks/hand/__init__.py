"""Hand track: human demonstrations (HOT3D, HOI4D, OakInk2, DexYCB) re-executed by a Franka arm + gripper or Wuji hands.

Evaluator: `python -m v2w.tracks.hand.evaluate` (see evaluate.py); metric record: `metrics()`.
"""
from pathlib import Path


def metrics(family, sample_dir, eval_dir):
    """Metric record from the evaluator's artifacts in eval_dir, gated on the physical-validity contract."""
    from v2w.metrics.record import rigid_record
    from . import phi
    phi.patch()
    ev = Path(eval_dir)
    return rigid_record(Path(sample_dir), ev / 'pkg_baseframe', ev / 'scene_pkg.json', ev / 'rollout_pkg.json',
                        kind_hint=family.get('eval2_kind'), require_hand_physics=True)
