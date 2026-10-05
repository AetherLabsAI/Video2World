"""Furniture track: FurnitureBench assembly, DROID pick-and-place and Push-T on a ManiSkill rig."""
from pathlib import Path


def metrics(family, sample_dir, eval_dir):
    """Metric record from the artifacts the evaluator left in eval_dir."""
    from v2w.metrics.record import rigid_metrics
    return rigid_metrics(family, Path(sample_dir), Path(eval_dir))
