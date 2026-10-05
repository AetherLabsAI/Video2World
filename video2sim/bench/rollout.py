"""Run a bridge_widowx package's OWN actions in its OWN scene — the author's
self-check (the analogue of `v2s replay` for this profile).

    python -m video2sim.bench.rollout <package> [--no-render]

Prints, per object: initial pose, final pose, displacement, grasp steps,
whether the terminal state is at rest; writes
<package>/verification/rollout_report.json and, unless --no-render,
<package>/verification/rollout.mp4 rendered through the package's declared
camera (or a default one). Exit 0 iff the rollout ran and the terminal state
is at rest. Whether the TASK was achieved is for you to judge from the
numbers and the video against what the source video shows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .evaluate import _json_default, run_agent_actions
from .protocol import load_manifest, scene_in_base_frame, validate_manifest


def rollout(pkg: Path, render: bool = True) -> dict:
    pkg = Path(pkg)
    m = load_manifest(pkg)
    errs = validate_manifest(pkg, m)
    if errs:
        raise SystemExit("package does not validate:\n- " + "\n- ".join(errs))
    if not m.get("actions"):
        raise SystemExit("package has no actions block")
    uid=(m.get('robot') or {}).get('uid', 'widowx250s_bridge')
    if uid in ('xarm7_pusher', 'panda', 'panda_robotiq'):
        from ..robot_scene_backend import rollout as robot_rollout
        return robot_rollout(pkg, render=render)
    if uid != 'widowx250s_bridge':
        raise ValueError('No registered bridge backend for robot: '+str(uid))
    A = np.load(pkg / m["actions"]["path"])
    scene = scene_in_base_frame(m, pkg)
    cam = (m.get("cameras") or [None])[0]
    if render and cam is None:
        from .annotate import default_K
        from .bridge_env import CV_TO_SAPIEN
        import sapien
        T = sapien.Pose([0.0, -0.16, 0.36], [0.8992917, -0.09263245, 0.35892478, 0.23209205]).to_transformation_matrix()
        T[:3, :3] = T[:3, :3] @ CV_TO_SAPIEN.T
        cam = {"width": 256, "height": 256, "intrinsics": default_K(256, 256).tolist(), "extrinsics_base_cam": T.tolist()}
    r = run_agent_actions(scene, A, render=render, camera=cam)
    rep = {"package": str(pkg), "T": int(len(A)), "init_ik": {k: v for k, v in r["init"].items() if k != "qpos"},
           "objects": {}, "quiescent": bool(max(r["final_velocity"].values(), default=0.0) < 5e-3),
           "contacts": r["contacts"]}
    for n in r["names"]:
        i = np.asarray(r["init_poses"][n]); f = np.asarray(r["final_poses"][n])
        rep["objects"][n] = {"init_pos": i[:3].round(4).tolist(), "final_pos": f[:3].round(4).tolist(),
                             "displacement_m": float(np.linalg.norm(f[:3] - i[:3])),
                             "max_height_gain_m": float(r["traj"][n][:, 2].max() - i[2]),
                             "grasp_steps": int(r["grasp"][n].sum())}
    ver = pkg / "verification"
    ver.mkdir(exist_ok=True)
    (ver / "rollout_report.json").write_text(json.dumps(rep, indent=2, default=_json_default))
    if render and r["frames"]:
        import imageio.v2 as imageio
        ks = sorted(r["frames"])
        imageio.mimwrite(ver / "rollout_keyframes.mp4", [r["frames"][k] for k in ks], fps=1, macro_block_size=1)
    return rep


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args(argv)
    rep = rollout(Path(a.package), render=not a.no_render)
    print(json.dumps(rep, indent=1, default=_json_default))
    return 0 if rep["quiescent"] else 1


if __name__ == "__main__":
    sys.exit(main())
