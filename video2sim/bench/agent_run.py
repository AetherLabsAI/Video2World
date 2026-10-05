"""Drive a coding agent over ONE benchmark sample and evaluate the result.

    v2s bench-run --sample <dir> --agent claude|codex|opencode [--model M]
                  [--out <pkg_dir>] [--reference <human_pkg>] [--max-rounds 2]

The agent gets a sandbox containing exactly one file — `video.mp4` — plus a
brief describing the `bridge_widowx` package it must write. The benchmark's
hidden data (`hidden/`, `human/`) never enters the sandbox, and the run log
is audited afterwards for any path into the sample directory.

The agent runs with the repository as its working directory so it can use
`v2s validate`, the library asset index and the env code, but it is told the
delivery must be a scene only — the evaluator supplies the actions.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

from .. import agents
from .protocol import load_manifest, validate_manifest

REPO = Path(__file__).resolve().parent.parent.parent

BRIEF = """# Task: rebuild this manipulation scene in simulation (Bridge real2sim benchmark)

Input: exactly one file, `{video}` (a 256x256, 5 Hz video of a WidowX 250S
robot manipulating tabletop objects). You have NOTHING else about this
episode: no robot logs, no calibration, no object poses. Everything metric
must come from the pixels and from what you know about the robot.

Deliverable: a **simulation package** at `{out}` in the `bridge_widowx` profile:

    {out}/protocol.json     physics_profile "bridge_widowx" (schema below)
    {out}/actions.npy       (T, 7) float64 — YOUR action stream that performs the task
    {out}/report.md         what you observed, measured, assumed, approximated
    {out}/assets/           any meshes you reference (package-relative paths)
    {out}/source/frames/    annotated evidence frames for every metric number

The contract is video2sim's: **the task, not the path.** Your package must
(1) rebuild the scene — where things are (robot-base frame, metres), how big,
what shape where the gripper touches them, what they rest on — and (2) carry
an action stream that, executed on the same robot in YOUR scene, reaches the
same task OUTCOME the video shows (the object ends where it ends, on/in what
it ends on/in). The evaluator executes your actions in your scene and judges
the terminal state with its own success predicate for this episode; how you
get there is yours to plan around this gripper. Path fidelity to the video is
evidence, not the criterion. As a secondary diagnostic the evaluator also
replays the withheld REAL trajectory in your scene, so put things where they
really are — a scene that only works for your own path is a weak scene.

`actions.npy` format: absolute end-effector states in the base frame at 5 Hz,
row = [x, y, z, roll, pitch, yaw, grip]; Bridge convention (gripper pointing
down == roll=pitch=yaw=0; R_link = R_zyx(rpy) · [[0,0,1],[0,1,0],[-1,0,0]]);
grip 1 = open, 0 = close (a command, not a width). Row 0 is the initial EE
state (the evaluator IK-drives the arm there before stepping). Keep per-step
moves ≤ 3 cm / ≤ 0.3 rad and end at rest — the controller is a real2sim PD
with ~1–4 cm tracking lag. The stock rest pose has the gripper tip at about
(0.29, 0.00, 0.135); fingertips are 1.6 cm below the EE frame; a closed
gripper leaves ~0.8 cm between the pads, so a 3–4 cm object is a firm grasp.

## Frame and robot facts (fixed by the benchmark — do not change them)

* World frame == robot base frame. The WidowX base_link origin is at the
  robot's mounting point on the table; +x points forward (away from the
  robot, toward the scene), +y to the robot's left, +z up. The robot's rest
  pose has the gripper tip at about (0.29, 0.00, 0.135).
* The gripper is the stock WidowX 250S parallel gripper, ~3.7 cm max opening
  per finger (7.4 cm total), fingers ~5 cm long.
* Bridge-dataset cameras are 640x480 captures resized to 256x256 (pixels are
  NOT square; fx/fy differ by 4/3). A typical focal length is ~623 px at
  640x480, i.e. fx≈249, fy≈333 at 256x256, but the camera pose differs per
  lab — estimate it from the picture (the robot base and gripper are visible
  and their geometry is known).
* The support surface (table / counter) top is close to z = 0 in the base
  frame (the base sits on it); a sink or tray raises objects above that.

## protocol.json

```jsonc
{{
  "protocol_version": "3.0",
  "physics_profile": "bridge_widowx",
  "name": "{name}",
  "status": "success",                 // success | failure | infeasible
  "task": {{"instruction": "YOUR reading of what happens in the video",
           "source_video": "source/video.mp4"}},
  "robot": {{"uid": "widowx250s_bridge", "base_pose": [0,0,0, 1,0,0,0]}},
  "cameras": [{{"name": "main", "width": 256, "height": 256,
               "intrinsics": [[fx,0,cx],[0,fy,cy],[0,0,1]],
               "extrinsics_base_cam": [[...4x4 OpenCV T_base_cam...]],
               "source": "estimated_from_video"}}],
  "scene": {{
    "support": {{"z": 0.0, "center": [0.35, 0.0], "size": [1.2, 1.2],
                "color": [0.75, 0.6, 0.4, 1]}},         // or null
    "arena": null,                                       // or {{"library": "bridge_table_1_v2"}}
    "props":   [ /* STATIC things: a sink, a tray, a rack — see entity */ ],
    "objects": [ /* DYNAMIC things the robot interacts with — at least one */ ]
  }},
  "actions": {{"path": "actions.npy", "dt": 0.2, "format": "ee_state_abs"}},
  "provenance": {{"method": "...", "agent_model": "...", "notes": "..."}}
}}
```

`status: success` requires actions that achieve the task in your scene;
`failure` = complete package whose actions do not achieve it (say why);
`infeasible` = the video cannot be reproduced (diagnosis in report.md).

entity (prop or object), all poses in the base frame, quaternion [w,x,y,z]:

```jsonc
{{"name": "can", "kind": "box" | "cylinder" | "sphere" | "container" | "mesh" | "library",
 "half_size": [hx, hy, hz],           // box; cylinder [radius, -, half_height] (axis +z); sphere [r, r, r];
                                      // container = open-top box: [inner_lx/2, inner_ly/2, wall_height/2] + "wall": 0.004,
                                      //   origin at the INNER FLOOR centre (a pot, a cup, a bin)
 "mesh_path": "assets/x.obj", "collision_path": "assets/x_collision.obj", "scale": 1.0,
 "library_id": "eggplant",            // SIMPLER Bridge asset library (list below)
 "pos": [x, y, z], "quat": [1, 0, 0, 0],
 "density": 300, "color": [r, g, b, a], "symmetry": null | "axis" | "box"}}
```

Library assets you may use with `"kind": "library"` (metre-scale, textured,
with collision meshes; the sink is 27x40x12 cm with its origin at the base
centre; cans/bottles have their long axis along local +y, so an upright can is
`"quat": [0.7071, 0.7071, 0, 0]`):
{library}

A retrieved/handmade mesh needs a convex-decomposed `collision_path` (the
`v2s asset` tool does this for Objaverse assets; `trimesh` + CoACD for your
own). A box or cylinder is right when the real shape IS one.

## Acceptance

`v2s validate {out}` must exit 0, and `python -m video2sim.bench.rollout {out}`
must run your actions in your scene and show the outcome: it prints each
object's initial/final pose, displacement, height gain and grasp steps and
writes `{out}/verification/rollout_keyframes.mp4` through your camera. Judge
that output against the video's semantics before you claim success — an
object that never gained height was not picked, one that ends 10 cm from the
target was not placed. `python -m video2sim.bench.preview {out}` renders the
initial scene through your camera; put it beside a video frame and fix what
does not line up. An object declared 3 cm above its support falls before the
first action; one declared inside another explodes.

## Judgement

Watch the video frame by frame first and decide what happens from the
picture. Estimate scale from the robot: the gripper opening, finger length
and link sizes are known; a soup can is ~6.6 cm wide. Put every measurement
on an annotated frame in `source/frames/`. Declare in report.md what you
could not observe. `status: infeasible` (with the diagnosis in report.md) is
a valid delivery for a video whose scene cannot be reconstructed.

Workspace rule: write only inside `{out}` and `/tmp/{name}/`. The only file
about this episode you may read is `{video}`.
"""


def library_listing() -> str:
    from .bridge_env import LIBRARY_ROOT, library_info
    info = library_info()
    names = sorted(p.name for p in (LIBRARY_ROOT / "custom/models").iterdir() if p.is_dir())
    out = []
    for n in names:
        bb = info.get(n, {}).get("bbox")
        if bb:
            s = np.subtract(bb["max"], bb["min"])
            out.append(f"  {n}  ({s[0]*100:.1f} x {s[1]*100:.1f} x {s[2]*100:.1f} cm)")
        else:
            out.append(f"  {n}")
    return "\n".join(out)


import numpy as np  # noqa: E402  (after the brief string for readability)


def stage(sample: Path, out: Path) -> Path:
    """Sandbox with only the video in it."""
    sb = Path(f"/tmp/v2s_bench_{sample.name}")
    if sb.exists():
        shutil.rmtree(sb)
    sb.mkdir(parents=True)
    shutil.copy(sample / "video.mp4", sb / "video.mp4")
    (out / "source").mkdir(parents=True, exist_ok=True)
    shutil.copy(sample / "video.mp4", out / "source" / "video.mp4")
    return sb / "video.mp4"


def audit(log: str, sample: Path) -> list[str]:
    """Any path into the sample directory (hidden/, human/, meta.json) is a leak."""
    hits = set(re.findall(re.escape(str(sample)) + r"[^\s\"')]*", log))
    bad = [h for h in hits if not h.endswith("video.mp4")]
    return sorted(bad)


def run(sample: Path, agent: str, model: str | None, out: Path, max_rounds: int = 2,
        skip_permissions: bool = False, reference: Path | None = None, evaluate_after: bool = True) -> dict:
    sample, out = Path(sample), Path(out)
    ad = agents.get(agent)
    if not ad.available():
        raise SystemExit(f"agent binary for {agent!r} not found")
    video = stage(sample, out)
    name = sample.name
    brief = BRIEF.format(video=video, out=out, name=name, library=library_listing())
    opts = {"model": model or ad.default_model, "skip_permissions": skip_permissions,
            "allowed_tools": "Bash,Read,Edit,Write,Glob,Grep", "add_dirs": [str(video.parent), str(out)]}
    rec = {"sample": name, "agent": agent, "model": opts["model"], "rounds": [], "cost_usd": 0.0}
    log_all = ""
    session = None
    prompt = brief
    t0 = time.time()
    for r in range(max_rounds):
        res = agents.run_round(ad, prompt, opts, cwd=REPO, resume=session if ad.supports_resume else None)
        log_all += res.log + "\n"
        rec["cost_usd"] += res.cost_usd or 0.0
        session = res.session
        try:
            m = load_manifest(out)
            errs = validate_manifest(out, m)
        except Exception as e:
            errs = [str(e)]
        rec["rounds"].append({"round": r + 1, "exit": res.exit_code, "validate_errors": errs})
        if not errs:
            break
        prompt = ("Your package at %s does not validate:\n- " % out + "\n- ".join(errs) +
                  "\nFix it (edit the package; the same rules apply) and run `v2s validate %s`." % out)
        if not ad.supports_resume:
            prompt = brief + "\n\n## Previous attempt failed validation\n" + prompt
    rec["seconds_agent"] = round(time.time() - t0)
    rec["leaks"] = audit(log_all, sample)
    (out / "agent_log.txt").write_text(log_all)
    (out / "agent_run.json").write_text(json.dumps(rec, indent=2))
    if evaluate_after:
        from .evaluate import evaluate
        rep = evaluate(out, sample, reference, render=True,
                       out=out / "verification" / "bench_report.json")
        rec["bench"] = {k: rep.get(k) for k in ("build", "stage", "replay_success", "lpips", "chamfer", "ape", "error")}
        (out / "agent_run.json").write_text(json.dumps(rec, indent=2, default=str))
    return rec


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="v2s bench-run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--agent", default="claude", choices=sorted(agents.ADAPTERS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None, help="package dir (default outputs/bench_<agent>/<sample>)")
    ap.add_argument("--reference", default=None, help="human package (default <sample>/human if present)")
    ap.add_argument("--max-rounds", type=int, default=2)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--skip-permissions", action="store_true",
                    help="claude --dangerously-skip-permissions (refused when running as root)")
    ap.add_argument("--dry-run", action="store_true", help="print the brief and exit")
    a = ap.parse_args(argv)
    sample = Path(a.sample)
    out = Path(a.out) if a.out else REPO / "outputs" / f"bench_{a.agent}" / sample.name
    ref = Path(a.reference) if a.reference else (sample / "human" if (sample / "human" / "protocol.json").exists() else None)
    if a.dry_run:
        print(BRIEF.format(video=sample / "video.mp4", out=out, name=sample.name, library=library_listing()))
        return 0
    rec = run(sample, a.agent, a.model, out, a.max_rounds, skip_permissions=a.skip_permissions,
              reference=ref, evaluate_after=not a.no_eval)
    print(json.dumps({k: v for k, v in rec.items() if k != "rounds"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
