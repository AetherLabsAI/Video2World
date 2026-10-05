"""The `bridge_widowx` package profile: what an agent delivers to the benchmark.

A package directory:

    protocol.json        manifest (below)
    assets/              any mesh files referenced (package-relative paths)
    report.md            what was observed / measured / approximated
    source/              the input video + evidence frames (optional)

protocol.json:

    {
      "protocol_version": "3.0",
      "physics_profile": "bridge_widowx",
      "name": "toysink2_00027",
      "status": "success" | "failure" | "infeasible",
      "task": {"instruction": "...", "source_video": "source/video.mp4"},
      "robot": {"uid": "widowx250s_bridge", "base_pose": [0,0,0, 1,0,0,0]},
      "cameras": [{"name": "main", "width": 256, "height": 256,
                   "intrinsics": [[..]], "extrinsics_base_cam": [[..]],
                   "source": "estimated_from_video"}],       # optional
      "scene": {"support": {...}|null, "arena": {...}|null,
                "props": [...], "objects": [...]},           # see bridge_env
      "provenance": {"method": "...", "agent_model": "...", "notes": "..."}
    }

The agent delivers its OWN action stream (`actions`: {"path": "actions.npy",
"dt": 0.2, "format": "ee_state_abs"} — (T,7) [x y z roll pitch yaw grip_cmd],
absolute end-effector states in the base frame, Bridge convention, 5 Hz,
grip 1 = open / 0 = close). Success is judged by the evaluator's task
predicate on the terminal state after executing those actions in the
delivered scene — the video2sim contract: same task outcome, not the same
path. The withheld real trajectory is additionally replayed in the scene as
a secondary diagnostic ("hidden replay"). `base_pose` is where the agent believes
the robot base is in ITS scene frame; the evaluator re-expresses the scene in
the base frame (translation + yaw only — the gravity axis is shared), which
is the "align on robot base / gravity / table plane, no per-object ICP" step.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

PROFILE = "bridge_widowx"
ENTITY_KINDS = ("box", "cylinder", "sphere", "container", "mesh", "library")


class PackageError(Exception):
    pass


def load_manifest(pkg: Path) -> dict:
    f = Path(pkg) / "protocol.json"
    if not f.exists():
        raise PackageError("protocol.json missing")
    try:
        return json.loads(f.read_text())
    except json.JSONDecodeError as e:
        raise PackageError(f"protocol.json is not valid JSON: {e}")


def _check_entity(e: dict, pkg: Path, what: str, errors: list[str]) -> None:
    if "name" not in e:
        errors.append(f"{what}: missing name")
    k = e.get("kind")
    if k not in ENTITY_KINDS:
        errors.append(f"{what} {e.get('name')}: kind must be one of {ENTITY_KINDS}, got {k!r}")
        return
    if k in ("box", "cylinder", "sphere", "container"):
        hs = e.get("half_size")
        if not (isinstance(hs, (list, tuple)) and len(hs) == 3 and all(float(v) > 0 for v in hs)):
            errors.append(f"{what} {e.get('name')}: half_size must be 3 positive numbers")
    if k == "mesh":
        for key in ("collision_path",):
            p = e.get(key)
            if not p or not (pkg / p).exists():
                errors.append(f"{what} {e.get('name')}: {key} {p!r} not found in package")
        mp = e.get("mesh_path")
        if mp and not (pkg / mp).exists():
            errors.append(f"{what} {e.get('name')}: mesh_path {mp!r} not found in package")
    if k == "library":
        from .bridge_env import LIBRARY_ROOT
        lid = e.get("library_id")
        if not lid or not (LIBRARY_ROOT / "custom/models" / str(lid)).exists():
            errors.append(f"{what} {e.get('name')}: unknown library_id {lid!r}")
    pos = e.get("pos")
    if not (isinstance(pos, (list, tuple)) and len(pos) == 3):
        errors.append(f"{what} {e.get('name')}: pos must be [x, y, z]")
    q = e.get("quat", (1, 0, 0, 0))
    if not (isinstance(q, (list, tuple)) and len(q) == 4 and abs(np.linalg.norm(q) - 1) < 1e-3):
        errors.append(f"{what} {e.get('name')}: quat must be a unit [w, x, y, z]")


def validate_manifest(pkg: Path, m: dict) -> list[str]:
    """Schema + file checks. Returns a list of error strings (empty == ok)."""
    pkg = Path(pkg)
    errors: list[str] = []
    if m.get("physics_profile") != PROFILE:
        errors.append(f"physics_profile must be {PROFILE!r}")
    if m.get("status") not in ("success", "failure", "infeasible"):
        errors.append("status must be success | failure | infeasible")
    if m.get("status") == "infeasible":
        if not (pkg / "report.md").exists():
            errors.append("infeasible package must carry report.md with the diagnosis")
        return errors
    rb = m.get("robot") or {}
    bp = rb.get("base_pose", [0, 0, 0, 1, 0, 0, 0])
    if not (isinstance(bp, (list, tuple)) and len(bp) == 7):
        errors.append("robot.base_pose must be 7 numbers [x y z qw qx qy qz]")
    sc = m.get("scene")
    if not isinstance(sc, dict):
        errors.append("scene missing")
        return errors
    if sc.get("support") is not None:
        s = sc["support"]
        if "z" not in s:
            errors.append("scene.support.z required")
    if sc.get("arena") is not None:
        from .bridge_env import _SIMPLER_ARENA
        if sc["arena"].get("library") not in _SIMPLER_ARENA:
            errors.append(f"scene.arena.library must be one of {list(_SIMPLER_ARENA)}")
    objs = sc.get("objects") or []
    if not objs:
        errors.append("scene.objects must contain at least one dynamic object")
    names = set()
    for o in objs:
        _check_entity(o, pkg, "object", errors)
        if o.get("name") in names:
            errors.append(f"duplicate entity name {o.get('name')!r}")
        names.add(o.get("name"))
    for p in sc.get("props") or []:
        _check_entity(p, pkg, "prop", errors)
        if p.get("name") in names:
            errors.append(f"duplicate entity name {p.get('name')!r}")
        names.add(p.get("name"))
    for c in m.get("cameras") or []:
        for key in ("width", "height", "intrinsics", "extrinsics_base_cam"):
            if key not in c:
                errors.append(f"camera {c.get('name')}: missing {key}")
    act = m.get("actions")
    if act is not None:
        path = act.get("path")
        if not path or not (pkg / path).exists():
            errors.append(f"actions.path {path!r} not found in package")
        else:
            try:
                A = np.load(pkg / path)
                if A.ndim != 2 or A.shape[1] != 7 or len(A) < 2:
                    errors.append(f"actions must be (T>=2, 7) [x y z roll pitch yaw grip]; got {A.shape}")
                elif not np.isfinite(A).all():
                    errors.append("actions contain non-finite values")
                elif A[:, 6].min() < -1e-6 or A[:, 6].max() > 1 + 1e-6:
                    errors.append("actions[:, 6] (gripper command) must be in [0, 1]")
                elif np.abs(A[:, :3]).max() > 1.0:
                    errors.append("actions positions must be in metres, base frame (|x| > 1 m seen)")
            except Exception as e:
                errors.append(f"actions unreadable: {e}")
    elif m.get("status") == "success":
        errors.append("status success requires an action stream (actions.path) that achieves the task")
    if not (pkg / "report.md").exists():
        errors.append("report.md missing")
    return errors


def scene_in_base_frame(m: dict, pkg: Path) -> dict:
    """Resolve package-relative paths and re-express every pose in the robot
    base frame using robot.base_pose (translation + yaw; roll/pitch of the
    declared base are ignored on purpose — gravity is the shared vertical)."""
    pkg = Path(pkg)
    sc = json.loads(json.dumps(m["scene"]))
    bp = np.asarray((m.get("robot") or {}).get("base_pose", [0, 0, 0, 1, 0, 0, 0]), dtype=np.float64)
    yaw = R.from_quat(np.r_[bp[4:7], bp[3]]).as_euler("xyz")[2]
    T_wb = np.eye(4)
    T_wb[:3, :3] = R.from_euler("z", yaw).as_matrix()
    T_wb[:3, 3] = bp[:3]
    T_bw = np.linalg.inv(T_wb)

    def xf(e):
        p = np.asarray(e["pos"], dtype=np.float64)
        q = np.asarray(e.get("quat", (1, 0, 0, 0)), dtype=np.float64)
        Te = np.eye(4)
        Te[:3, :3] = R.from_quat(np.r_[q[1:], q[0]]).as_matrix()
        Te[:3, 3] = p
        Tb = T_bw @ Te
        e["pos"] = Tb[:3, 3].tolist()
        qb = R.from_matrix(Tb[:3, :3]).as_quat()
        e["quat"] = [float(qb[3]), float(qb[0]), float(qb[1]), float(qb[2])]
        for key in ("mesh_path", "collision_path"):
            if e.get(key):
                e[key] = str(pkg / e[key])
        return e

    sc["objects"] = [xf(o) for o in sc.get("objects") or []]
    sc["props"] = [xf(p) for p in sc.get("props") or []]
    if sc.get("support"):
        s = sc["support"]
        c = np.asarray(list(s.get("center", (0.3, 0.0))) + [float(s["z"])], dtype=np.float64)
        cb = T_bw @ np.r_[c, 1.0]
        s["center"] = cb[:2].tolist()
        s["z"] = float(cb[2])
    return sc


def write_package(out: Path, manifest: dict, report: str = "") -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol.json").write_text(json.dumps(manifest, indent=2))
    (out / "report.md").write_text(report or "# report\n\n(no report)\n")
    return out
