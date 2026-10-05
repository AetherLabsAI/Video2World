#!/usr/bin/env python
"""Validate a protocol-v1 package (agent/PROTOCOL.md). Exit 0 = valid.

Usage: validate_protocol.py <package_dir>

Pure numpy — no simulator import, so it runs anywhere, fast.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

QLIM_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
QLIM_HI = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
OBJ_KINDS = {"box", "lshape", "cylinder", "container", "mesh", "ycb"}
PROP_KINDS = {"static_box", "cylinder", "container", "cylinder_container", None}


class Checker:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def err(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    def require(self, cond, msg) -> bool:
        if not cond:
            self.err(msg)
        return bool(cond)


def _file(pkg: Path, rel, c: Checker, what: str) -> Path | None:
    if not isinstance(rel, str) or not rel:
        c.err(f"{what}: path missing or not a string: {rel!r}")
        return None
    p = pkg / rel
    if not p.exists():
        c.err(f"{what}: file not found: {rel}")
        return None
    return p


def validate(pkg: Path) -> Checker:
    c = Checker()
    mf = pkg / "protocol.json"
    if not mf.exists():
        c.err("protocol.json is missing")
        return c
    try:
        m = json.loads(mf.read_text())
    except json.JSONDecodeError as e:
        c.err(f"protocol.json failed to parse: {e}")
        return c

    if m.get("protocol_version") not in ("1.0", "1.1", "2.0", "2.1", "2.2", "3.0"):
        c.err(
            f"protocol_version must be one of '1.0'-'3.0', got {m.get('protocol_version')!r}"
        )
    profile = m.get("physics_profile", "rigid_sapien")
    if profile == "twin_mujoco_v2":
        from ..native_twin import validate as validate_twin
        for error in validate_twin(pkg): c.err(error)
        return c
    if profile == "bridge_widowx":
        # benchmark profile: a scene only, no action stream (the evaluator
        # supplies the withheld real trajectory) — its own checker
        from ..bench.protocol import validate_manifest
        for e in validate_manifest(pkg, m):
            c.err(e)
        return c
    if profile not in ("rigid_sapien", "soft_warp", "genesis", "robodojo"):
        c.err(f"invalid physics_profile: {profile!r}")
    status = m.get("status")
    if status not in ("success", "failure", "infeasible"):
        c.err(f"invalid status: {status!r}")

    task = m.get("task") or {}
    if not (task.get("instruction") or "").strip():
        c.err("task.instruction is empty")

    rep = pkg / "report.md"
    if not rep.exists():
        c.err("report.md is missing (the observation report is mandatory)")
    elif len(rep.read_text()) < 200:
        c.err("report.md is too short (<200 chars) to be a real report")

    if status == "infeasible":
        # unsuitable video: a diagnosis report and the status claim suffice
        return c

    if profile == "robodojo":
        _validate_robodojo(pkg, m, c)
        _validate_rig_delivery(pkg, c)
        return c

    # ---------------------------------------------------------------- actions
    am = m.get("actions") or {}
    ap = _file(pkg, am.get("path"), c, "actions.path")
    actions = None
    if ap is not None:
        actions = np.load(ap)
        if actions.ndim != 2 or actions.shape[1] != 8:
            c.err(f"actions must be (T, 8), got {actions.shape}")
            actions = None
    if actions is not None:
        T = len(actions)
        c.require(
            am.get("T") == T, f"actions.T={am.get('T')} disagrees with array length {T}"
        )
        c.require(np.isfinite(actions).all(), "actions contain NaN/Inf")
        arm = actions[:, :7]
        margin = 1e-3
        if not ((arm >= QLIM_LO - margin) & (arm <= QLIM_HI + margin)).all():
            c.err("arm joint targets exceed the Panda joint limits")
        g = actions[:, 7]
        c.require((np.abs(g) <= 1.0 + 1e-6).all(),
                  "gripper command outside [-1, 1]")
        env = m.get("environment") or {}
        cf = env.get("control_freq")
        if cf and abs(am.get("dt", 0) - 1.0 / cf) > 1e-9:
            c.err(f"actions.dt={am.get('dt')} disagrees with 1/control_freq={1.0 / cf}")

    # ------------------------------------------------------------- soft_warp
    if profile == "soft_warp":
        _validate_soft(pkg, m, actions, c)
        return c
    if profile == "genesis":
        _validate_genesis(pkg, m, actions, c)
        return c

    # ---------------------------------------------------------------- expected
    ex = m.get("expected") or {}
    op = _file(pkg, ex.get("obj_poses"), c, "expected.obj_poses")
    if op is not None and actions is not None:
        obj = np.load(op)
        if obj.shape != (len(actions), 7):
            c.err(f"expected/obj_poses is {obj.shape}, must be ({len(actions)}, 7)")
        else:
            qn = np.linalg.norm(obj[:, 3:], axis=1)
            c.require(
                np.abs(qn - 1).max() < 1e-3, "obj_poses quaternions are not normalized"
            )
            fin = np.asarray(ex.get("final_obj_pose", []), dtype=float)
            if fin.shape != (7,):
                c.err("expected.final_obj_pose must have 7 elements")
            elif np.abs(fin - obj[-1]).max() > 1e-5:
                c.err(
                    "expected.final_obj_pose disagrees with the last row of obj_poses"
                )
    for k in ("tolerance_pos", "tolerance_rot_deg"):
        if not isinstance(ex.get(k), (int, float)):
            c.err(f"expected.{k} is missing or not a number")
    sc = ex.get("success_criteria") or {}
    if sc.get("type") not in ("final_pose", "goal_region", "articulation",
                              "multi_goal"):
        c.err(f"invalid success_criteria.type: {sc.get('type')!r}")
    if sc.get("type") == "multi_goal":
        # An assembly task's semantics are "every piece ended where it belongs".
        # Scoring only the last piece would pass a tower whose lower courses
        # collapsed under it.
        goals = sc.get("goals")
        names = [o.get("name") for o in ((m.get("scene") or {}).get("objects") or [])]
        if not isinstance(goals, list) or not goals:
            c.err("multi_goal criterion needs a non-empty 'goals' list")
        else:
            for g in goals:
                if g.get("object") not in names:
                    c.err(f"multi_goal goal names unknown object {g.get('object')!r}")
                if len(g.get("target_pos", [])) != 3:
                    c.err("multi_goal goal needs target_pos (3)")
                if not isinstance(g.get("pos_radius"), (int, float)):
                    c.err("multi_goal goal needs a numeric pos_radius")
    if sc.get("type") == "articulation" and not (
        isinstance(sc.get("target_qpos"), list)
        and isinstance(sc.get("tolerance"), (int, float))
    ):
        c.err("articulation criterion needs target_qpos and tolerance")
    if ex.get("grasp_transitions") is None:
        c.warn(
            "expected.grasp_transitions not set (record the is_grasped transition count)"
        )

    # ---------------------------------------------------------------- robot
    rb = m.get("robot") or {}
    _file(pkg, rb.get("urdf"), c, "robot.urdf")
    base = np.asarray(rb.get("base_pose", []), dtype=float)
    if base.shape != (7,):
        c.err("robot.base_pose must have 7 elements")
    iq = np.asarray(rb.get("init_qpos", []), dtype=float)
    if iq.shape != (9,):
        c.err("robot.init_qpos must have 9 elements")
    ist = m.get("initial_state") or {}
    if iq.shape == (9,) and np.asarray(ist.get("qpos", []), dtype=float).shape == (9,):
        if np.abs(iq - np.asarray(ist["qpos"])).max() > 1e-9:
            c.err("robot.init_qpos disagrees with initial_state.qpos")
    if np.asarray(ist.get("obj_pose", []), dtype=float).shape != (7,):
        c.err("initial_state.obj_pose must have 7 elements")

    # ------------------------------------------------------------- scene/base
    scene = m.get("scene") or {}
    support = scene.get("support")
    if base.shape == (7,):
        expect_p = np.zeros(3) if support else np.array([-0.615, 0, 0])
        if np.abs(base[:3] - expect_p).max() > 1e-6:
            c.err(
                f"base_pose {base[:3].tolist()} does not match the "
                f"{'support' if support else 'table'}-mode base position {expect_p.tolist()}"
            )
    if support is not None and "z" not in support:
        c.err("scene.support is missing z")
    objs = scene.get("objects") or []
    if len(objs) < 1:
        c.err("scene.objects must hold at least 1 dynamic object")
    if len(objs) > 1:
        # Multi-object scene. objects[0] stays the PRIMARY (every pre-existing
        # field refers to it); the rest need their own initial poses and their
        # own recorded trajectory, or a replay would reset seven of eight
        # objects to their authored spawn poses and still report healthy.
        n = len(objs)
        ist_all = np.asarray((m.get("initial_state") or {}).get("obj_poses", []),
                             dtype=float)
        if ist_all.shape != (n, 7):
            c.err(f"initial_state.obj_poses must be ({n}, 7) for a "
                  f"{n}-object scene, got {list(ist_all.shape)}")
        else:
            first = np.asarray((m.get("initial_state") or {}).get("obj_pose", []),
                               dtype=float)
            if first.shape == (7,) and np.abs(ist_all[0] - first).max() > 1e-9:
                c.err("initial_state.obj_poses[0] disagrees with "
                      "initial_state.obj_pose (the primary object)")
        apath = _file(pkg, ex.get("obj_poses_all"), c, "expected.obj_poses_all")
        if apath is not None and actions is not None:
            allp = np.load(apath)
            if allp.shape != (len(actions), n, 7):
                c.err(f"expected/obj_poses_all is {allp.shape}, must be "
                      f"({len(actions)}, {n}, 7)")
            else:
                fa = np.asarray(ex.get("final_obj_poses", []), dtype=float)
                if fa.shape != (n, 7):
                    c.err(f"expected.final_obj_poses must be ({n}, 7)")
                elif np.abs(fa - allp[-1]).max() > 1e-5:
                    c.err("expected.final_obj_poses disagrees with the last "
                          "row of obj_poses_all")
    for o in objs:
        kind = o.get("kind")
        if kind not in OBJ_KINDS:
            c.err(f"invalid object.kind: {kind!r}")
        if kind in ("mesh", "ycb"):
            _file(pkg, o.get("mesh_path"), c, f"object[{o.get('name')}].mesh_path")
        if kind == "ycb":
            _file(
                pkg,
                o.get("collision_path"),
                c,
                f"object[{o.get('name')}].collision_path",
            )
        if kind == "container" and not (float(o.get("wall", 0.0)) > 0.0):
            c.err("object.kind 'container' needs a positive wall thickness")
        for k in ("init_pos", "init_quat"):
            n = 3 if k == "init_pos" else 4
            if len(o.get(k, [])) != n:
                c.err(f"object.{k} must have {n} elements")
    artics = scene.get("articulations") or []
    if len(artics) > 1:
        c.err(f"at most 1 articulation is supported (env limit), got {len(artics)}")
    for a in artics:
        if a.get("kind") not in ("drawer", "cabinet_door", "kettle_lid", "hinged_lid"):
            c.err(f"invalid articulation.kind: {a.get('kind')!r}")
        if a.get("kind") == "kettle_lid":
            if "pos" not in a or "hinge_azimuth_deg" not in a:
                c.err("kettle_lid articulation needs pos and hinge_azimuth_deg")
        elif a.get("kind") == "hinged_lid":
            if "pos" not in a or len(a.get("body", [])) != 3:
                c.err("hinged_lid articulation needs pos and body (3)")
        elif len(a.get("inner_size", [])) != 3 or "pos" not in a:
            c.err("articulation needs inner_size (3) and pos")
    if artics:
        ap2 = _file(pkg, ex.get("artic_qpos"), c, "expected.artic_qpos")
        if ap2 is not None and actions is not None:
            aq = np.load(ap2)
            if aq.ndim != 2 or len(aq) != len(actions):
                c.err(f"artic_qpos is {aq.shape}, must be ({len(actions)}, dof)")
            else:
                fin_a = np.asarray(ex.get("final_artic_qpos", []), dtype=float)
                if fin_a.shape != (aq.shape[1],):
                    c.err(
                        "expected.final_artic_qpos has a different width than artic_qpos"
                    )
                elif np.abs(fin_a - aq[-1]).max() > 1e-6:
                    c.err("final_artic_qpos disagrees with the last row of artic_qpos")
        if not isinstance(ex.get("tolerance_artic"), (int, float)):
            c.err("expected.tolerance_artic is missing or not a number")
    for p in scene.get("props") or []:
        if p.get("kind") not in PROP_KINDS:
            c.err(f"invalid prop.kind: {p.get('kind')!r}")
        if p.get("kind") is None and p.get("collision_path"):
            _file(pkg, p["collision_path"], c, "prop.collision_path")
        if p.get("kind") == "container" and not (
            len(p.get("inner_size", [])) == 2 and "height" in p and "pos" in p
        ):
            c.err("container prop needs inner_size, height and pos")
        if p.get("kind") == "cylinder" and not (
            float(p.get("radius", 0.0)) > 0.0
            and float(p.get("half_length", 0.0)) > 0.0 and "pos" in p
        ):
            c.err("cylinder prop needs a positive radius, a positive "
                  "half_length and pos")
        if p.get("kind") == "cylinder_container":
            if not (float(p.get("inner_diameter", 0.0)) > 0.0
                    and float(p.get("height", 0.0)) > 0.0 and "pos" in p):
                c.err("cylinder_container prop needs a positive inner_diameter, "
                      "a positive height and pos")
            elif int(p.get("sections", 20)) < 6:
                c.err("cylinder_container prop needs at least 6 sections")

    # ---------------------------------------------------------------- cameras
    cams = m.get("cameras") or []
    if not cams:
        c.err(
            "cameras is empty - the protocol requires at least one (extrinsics are base-relative)"
        )
    for cam in cams:
        K = np.asarray(cam.get("intrinsics", []), dtype=float)
        if K.shape != (3, 3):
            c.err(f"camera[{cam.get('name')}].intrinsics must be 3x3")
        E = np.asarray(cam.get("extrinsics_base_cam", []), dtype=float)
        if E.shape != (4, 4):
            c.err(f"camera[{cam.get('name')}].extrinsics_base_cam must be 4x4")
        else:
            if np.abs(E[3] - np.array([0, 0, 0, 1])).max() > 1e-9:
                c.err("extrinsics_base_cam bottom row must be [0, 0, 0, 1]")
            R = E[:3, :3]
            if (
                np.abs(R @ R.T - np.eye(3)).max() > 1e-5
                or abs(np.linalg.det(R) - 1) > 1e-5
            ):
                c.err("extrinsics_base_cam rotation is not orthonormal / det != 1")
        if not (cam.get("width") and cam.get("height")):
            c.err("camera is missing width/height")

    # ------------------------------------------------------------- provenance
    pv = m.get("provenance") or {}
    for e in pv.get("evidence") or []:
        _file(pkg, e, c, "provenance.evidence")
    if not (pv.get("checks") or {}).get("visual_inspection"):
        c.err(
            "provenance.checks.visual_inspection must be true - "
            "a conclusion drawn from pixels has to be eyeballed"
        )
    else:
        # The claim alone is worthless: a text-only model cannot open an image
        # at all, yet it can still write `true` here. One did — it read a
        # single frame, saved no evidence, asserted visual inspection, and
        # passed this gate. Require the frames the claim refers to.
        ev = pv.get("evidence") or []
        imgs = [
            e
            for e in ev
            if isinstance(e, str) and e.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if len(imgs) < 2:
            c.err(
                f"provenance.checks.visual_inspection is true but "
                f"provenance.evidence lists {len(imgs)} annotated frame(s); "
                "a visual conclusion has to cite the frames it came from"
            )
    if task.get("source_video"):
        _file(pkg, task["source_video"], c, "task.source_video")

    _validate_rig_delivery(pkg, c)
    return c


def _validate_rig_delivery(pkg: Path, c: Checker) -> None:
    """The robodojo/ deliverable: stream, rollout, render, distribution report.

    A robodojo/ directory is an additional deliverable on top of a base
    profile (rigid_sapien + delivery_profile), and it is the WHOLE deliverable
    for `physics_profile: "robodojo"`. Either way the same four artifacts have
    to be there: the other profiles get their render-in-package invariant from
    gate 2 writing verification/render.mp4 itself; this delivery is produced
    by the agent, so the gate checks the same invariant instead.
    """
    dj = pkg / "robodojo"
    if not dj.is_dir():
        return
    dr = None
    if not (dj / "delivery_report.json").exists():
        c.err("robodojo/delivery_report.json is missing")
    else:
        try:
            dr = json.loads((dj / "delivery_report.json").read_text())
        except json.JSONDecodeError as e:
            c.err(f"robodojo/delivery_report.json failed to parse: {e}")
    # an explicit rejection with reasons is a valid, honest outcome —
    # like `infeasible`, it needs nothing beyond the report saying so
    if dr is not None and dr.get("status") == "rejected":
        if not dr.get("reasons"):
            c.err("a rejected robodojo delivery must state its reasons")
    elif dr is not None:
        for req in ("actions.npy", "render.mp4", "rollout.npz"):
            if not (dj / req).exists():
                c.err(f"robodojo/{req} is missing")
        if (dj / "actions.npy").exists():
            da = np.load(dj / "actions.npy")
            if da.ndim != 2 or da.shape[1] != 14:
                c.err(f"robodojo/actions.npy must be (T, 14), got {da.shape}")
            elif not np.isfinite(da).all():
                c.err("robodojo/actions.npy contains non-finite values")
        if dr.get("in_distribution") is not True:
            c.err(
                "robodojo delivery claims delivered but is not "
                f"in-distribution (reasons: {dr.get('reasons')})"
            )


def _validate_robodojo(pkg: Path, m: dict, c: Checker) -> None:
    """`physics_profile: "robodojo"` — the rig IS the simulator.

    The other profiles describe a scene that gate 2 rebuilds in SAPIEN behind
    a Panda, and a robodojo/ delivery rides on top of that. Some episodes have
    no such base: this one is DUAL-ARM with four manipulated objects, and the
    SAPIEN env deliberately holds one Panda and exactly one dynamic body, so a
    base package would have to misrepresent the task to exist at all. When the
    episode is authored directly on the benchmark's rig, that rig is the
    package's physics: the action stream is their 14-dim joint stream, the
    scene is expressed in their world frame, and gate 2 re-runs the rig.

    What that changes here: no Panda fields, no (T, 8) actions, and any number
    of manipulated objects — but the scene still has to be checkable, and the
    success criterion still has to be a stated, per-object target rather than
    an adjective.
    """
    am = m.get("actions") or {}
    ap = _file(pkg, am.get("path"), c, "actions.path")
    actions = None
    if ap is not None:
        actions = np.load(ap)
        if actions.ndim != 2 or actions.shape[1] != 14:
            c.err(f"robodojo actions must be (T, 14), got {actions.shape}")
            actions = None
    if actions is not None:
        c.require(am.get("T") == len(actions),
                  f"actions.T={am.get('T')} disagrees with array length {len(actions)}")
        c.require(np.isfinite(actions).all(), "actions contain NaN/Inf")
        g = actions[:, [6, 13]]
        c.require(bool(((g >= -1e-6) & (g <= 1 + 1e-6)).all()),
                  "gripper dims (6, 13) must be normalized 0..1, got "
                  f"[{g.min():.3f}, {g.max():.3f}]")
        if abs(float(am.get("dt", 0.0)) - 1.0 / 25.0) > 1e-9:
            c.err(f"actions.dt must be 1/25 s (the rig's clock), got {am.get('dt')!r}")

    scene = m.get("scene") or {}
    objs = scene.get("objects") or []
    if not objs:
        c.err("the robodojo profile needs at least 1 manipulated object")
    names = set()
    for o in objs:
        nm = o.get("name")
        if not nm or nm in names:
            c.err(f"object.name missing or duplicated: {nm!r}")
        names.add(nm)
        if o.get("kind") not in ("box", "cylinder", "mesh", "polygon_container"):
            c.err(f"object[{nm}].kind must be box, cylinder, mesh or "
                  f"polygon_container, got {o.get('kind')!r}")
        if o.get("kind") == "polygon_container":
            # A vessel the robot CARRIES: it is a manipulated object, not a
            # prop, so it belongs here rather than under scene.props — and its
            # cavity has to be declared, because "put them IN it" is the
            # success criterion and a bounding box would pass every other check.
            if not (float(o.get("inner_across_flats", 0.0)) > 0.0
                    and float(o.get("height", 0.0)) > 0.0
                    and float(o.get("wall", 0.0)) > 0.0):
                c.err(f"object[{nm}] kind 'polygon_container' needs positive "
                      "inner_across_flats, height and wall")
            n = int(o.get("sections", 6))
            if n < 4 or n % 2:
                c.err(f"object[{nm}].sections must be an even number >= 4, "
                      f"got {n}")
        if o.get("kind") == "mesh":
            _file(pkg, o.get("mesh_path"), c, f"object[{nm}].mesh_path")
            _file(pkg, o.get("collision_path"), c, f"object[{nm}].collision_path")
        if len(o.get("init_pos", [])) != 3:
            c.err(f"object[{nm}].init_pos must have 3 elements")
        if o.get("init_quat") is not None and len(o["init_quat"]) != 4:
            c.err(f"object[{nm}].init_quat must have 4 elements")
    for p in scene.get("props") or []:
        if p.get("kind") not in ("static_box", "cylinder", "container",
                                 "cylinder_container", "mesh", None):
            c.err(f"invalid prop.kind: {p.get('kind')!r}")
        if len(p.get("pos", [])) != 3:
            c.err(f"prop[{p.get('name')}].pos must have 3 elements")

    ex = m.get("expected") or {}
    tgts = ex.get("object_targets")
    if not isinstance(tgts, list) or not tgts:
        c.err("expected.object_targets must list the per-object goals "
              "({object, target_pos, pos_radius}) — a multi-object task has "
              "no single final_obj_pose")
    else:
        for t in tgts:
            if t.get("object") not in names:
                c.err(f"expected.object_targets names an unknown object: "
                      f"{t.get('object')!r}")
            if len(t.get("target_pos", [])) != 3:
                c.err(f"object_targets[{t.get('object')}].target_pos needs 3 elements")
            if not isinstance(t.get("pos_radius"), (int, float)):
                c.err(f"object_targets[{t.get('object')}].pos_radius must be a number")
    if not isinstance(ex.get("tolerance_pos"), (int, float)):
        c.err("expected.tolerance_pos is missing or not a number")
    if ex.get("grasp_transitions") is None:
        c.warn("expected.grasp_transitions not set (record the carry count)")

    # An episode whose terminal state IS its initial state (cover three blocks,
    # then uncover them) has no final-pose criterion worth the name: doing
    # nothing satisfies it. `expected.event_sequence` is the ordered
    # intermediate-state criterion for those, and it is only worth anything if
    # the delivery carries it onto the rig, so both halves are checked.
    seq = ex.get("event_sequence")
    if seq is not None:
        evs = seq.get("events")
        if seq.get("type") != "ordered_events":
            c.err("expected.event_sequence.type must be 'ordered_events', "
                  f"got {seq.get('type')!r}")
        if not isinstance(evs, list) or len(evs) < 2:
            c.err("expected.event_sequence.events must list at least 2 events "
                  "— a one-event sequence has no order to check")
        else:
            for e in evs:
                if e.get("object") not in names or e.get("reference") not in names:
                    c.err(f"event {e.get('name')!r} names an object outside the "
                          f"scene: {e.get('object')!r} / {e.get('reference')!r}")
                if e.get("relation") not in ("over", "clear_of"):
                    c.err(f"event {e.get('name')!r}: relation must be 'over' or "
                          f"'clear_of', got {e.get('relation')!r}")
                if not isinstance(e.get("xy_radius"), (int, float)):
                    c.err(f"event {e.get('name')!r} needs a numeric xy_radius")
                # A vertical band, in any of its four forms. The maxima say
                # "and it is DOWN"; a pour needs the mirror ("raised above the
                # rim and inverted"), so requiring max_z_m specifically would
                # make the criterion for this episode's defining state
                # inexpressible — see robodojo_deliver._event_hold.
                zb = [k for k in ("max_z_m", "max_z_above_ref_m",
                                  "min_z_m", "min_z_above_ref_m")
                      if isinstance(e.get(k), (int, float))]
                if not zb:
                    c.err(f"event {e.get('name')!r} needs at least one numeric "
                          "vertical bound (max_z_m / max_z_above_ref_m / "
                          "min_z_m / min_z_above_ref_m)")
        dr_path = pkg / "robodojo/delivery_report.json"
        if dr_path.exists():
            try:
                dr = json.loads(dr_path.read_text())
            except json.JSONDecodeError:
                dr = {}
            if dr.get("status") != "rejected":
                rig = dr.get("rig_events")
                if not isinstance(rig, list) or len(rig) != len(evs or []):
                    c.err("expected.event_sequence declares "
                          f"{len(evs or [])} events but robodojo/"
                          "delivery_report.json carries rig_events="
                          f"{len(rig) if isinstance(rig, list) else None} — "
                          "gate 2 re-runs the rig and can only check the "
                          "sequence the delivery recorded")
                elif [e.get("name") for e in rig] != [e.get("name") for e in evs]:
                    c.err("robodojo/delivery_report.json rig_events names "
                          "disagree with expected.event_sequence")

    cams = m.get("cameras") or []
    if not cams:
        c.err("cameras is empty - the protocol requires at least one")
    for cam in cams:
        K = np.asarray(cam.get("intrinsics", []), dtype=float)
        if K.shape != (3, 3):
            c.err(f"camera[{cam.get('name')}].intrinsics must be 3x3")
        E = np.asarray(cam.get("extrinsics_world_cam", []), dtype=float)
        if E.shape != (4, 4):
            c.err(f"camera[{cam.get('name')}].extrinsics_world_cam must be 4x4 "
                  "(the rig's frame is a world frame, not a robot base frame)")
        else:
            R = E[:3, :3]
            if (np.abs(R @ R.T - np.eye(3)).max() > 1e-5
                    or abs(np.linalg.det(R) - 1) > 1e-5):
                c.err("extrinsics_world_cam rotation is not orthonormal / det != 1")
        if not (cam.get("width") and cam.get("height")):
            c.err("camera is missing width/height")

    pv = m.get("provenance") or {}
    for e in pv.get("evidence") or []:
        _file(pkg, e, c, "provenance.evidence")
    if not (pv.get("checks") or {}).get("visual_inspection"):
        c.err("provenance.checks.visual_inspection must be true - "
              "a conclusion drawn from pixels has to be eyeballed")
    else:
        imgs = [e for e in (pv.get("evidence") or [])
                if isinstance(e, str) and e.lower().endswith((".png", ".jpg", ".jpeg"))]
        if len(imgs) < 2:
            c.err(f"provenance.checks.visual_inspection is true but "
                  f"provenance.evidence lists {len(imgs)} annotated frame(s); "
                  "a visual conclusion has to cite the frames it came from")
    task = m.get("task") or {}
    if task.get("source_video"):
        _file(pkg, task["source_video"], c, "task.source_video")
    if not (pkg / "robodojo").is_dir():
        c.err("the robodojo profile requires a robodojo/ delivery directory")


def _validate_soft(pkg: Path, m: dict, actions, c: Checker) -> None:
    """soft_warp profile: cloth spec + particle trajectory acceptance."""
    soft = (m.get("scene") or {}).get("soft_bodies") or []
    if len(soft) != 1:
        c.err(f"soft_warp needs exactly 1 soft_body, got {len(soft)}")
        return
    sb = soft[0]
    if sb.get("kind") != "cloth_grid":
        c.err(f"invalid soft_body.kind: {sb.get('kind')!r}")
    for k in ("nx", "ny", "spacing", "origin"):
        if k not in sb:
            c.err(f"soft_body is missing {k}")
    rb = m.get("robot") or {}
    _file(pkg, rb.get("urdf"), c, "robot.urdf")
    if np.asarray(rb.get("base_pose", []), dtype=float).shape != (7,):
        c.err("robot.base_pose must have 7 elements")
    if np.asarray(rb.get("init_qpos", []), dtype=float).shape != (9,):
        c.err("robot.init_qpos must have 9 elements")
    for p in (m.get("scene") or {}).get("props") or []:
        kind = p.get("kind", "mesh" if p.get("collision_path") else None)
        if kind not in ("static_box", "cylinder", "mesh"):
            c.err(f"invalid prop.kind for the soft profile: {kind!r}")
        elif kind == "mesh":
            _file(pkg, p.get("collision_path"), c, "soft prop.collision_path")
        elif kind == "static_box" and len(p.get("half_size", [])) != 3:
            c.err("static_box prop needs half_size (3)")
        elif kind == "cylinder" and not ("radius" in p and "half_length" in p):
            c.err("cylinder prop needs radius and half_length")
        if "pos" not in p:
            c.err("soft prop is missing pos")
    ex = m.get("expected") or {}
    pf = _file(pkg, ex.get("particles"), c, "expected.particles")
    if pf is not None and actions is not None:
        P = np.load(pf)
        n = int(sb.get("nx", 0)) * int(sb.get("ny", 0))
        if P.shape != (len(actions), n, 3):
            c.err(f"particles is {P.shape}, must be ({len(actions)}, {n}, 3)")
        elif not np.isfinite(P).all():
            c.err("particles contain NaN/Inf")
    if not isinstance(ex.get("tolerance_particles"), (int, float)):
        c.err("expected.tolerance_particles is missing or not a number")
    sc = ex.get("success_criteria") or {}
    if sc.get("type") == "soft_centroid":
        if not (
            len(sc.get("target_pos", [])) == 3
            and isinstance(sc.get("pos_radius"), (int, float))
        ):
            c.err("soft_centroid criterion needs target_pos (3) and pos_radius")
    elif sc.get("type") == "soft_clear":
        if not (
            len(sc.get("clear_pos", [])) >= 2
            and isinstance(sc.get("clear_radius"), (int, float))
        ):
            c.err("soft_clear criterion needs clear_pos and clear_radius")
    else:
        c.err(
            f"soft profile success_criteria.type must be soft_centroid or soft_clear, "
            f"got {sc.get('type')!r}"
        )
    pv = m.get("provenance") or {}
    for e in pv.get("evidence") or []:
        _file(pkg, e, c, "provenance.evidence")
    if not (pv.get("checks") or {}).get("visual_inspection"):
        c.err("provenance.checks.visual_inspection must be true")
    if (m.get("task") or {}).get("source_video"):
        _file(pkg, m["task"]["source_video"], c, "task.source_video")


GENESIS_ENTITY_TYPES = {
    "rigid",
    "articulated",
    "sph_liquid",
    "mpm_elastoplastic",
    "mpm_sand",
    "pbd_cloth",
}
GENESIS_PARTICLE_TYPES = {"sph_liquid", "mpm_elastoplastic", "mpm_sand", "pbd_cloth"}


def _validate_genesis(pkg: Path, m: dict, actions, c: Checker) -> None:
    """genesis profile: generalized entity list + two-tier state acceptance."""
    ents = (m.get("scene") or {}).get("entities") or []
    if not ents:
        c.err("the genesis profile needs at least 1 entity")
    names = set()
    for e in ents:
        if e.get("type") not in GENESIS_ENTITY_TYPES:
            c.err(f"invalid entity.type: {e.get('type')!r}")
        if not e.get("name") or e["name"] in names:
            c.err(f"entity.name missing or duplicated: {e.get('name')!r}")
        names.add(e.get("name"))
        if e.get("type") == "articulated":
            _file(pkg, e.get("urdf"), c, f"entity[{e.get('name')}].urdf")
        mf = (e.get("morph") or {}).get("file")
        if mf and not str(mf).startswith("meshes/"):
            _file(pkg, mf, c, f"entity[{e.get('name')}].morph.file")
    rb = m.get("robot") or {}
    if rb.get("urdf"):
        _file(pkg, rb["urdf"], c, "robot.urdf")
    if np.asarray(rb.get("init_qpos", []), dtype=float).size < 7:
        c.err("robot.init_qpos needs at least 7 elements")
    ex = m.get("expected") or {}
    sf = _file(pkg, ex.get("states"), c, "expected.states")
    if sf is not None and actions is not None:
        S = np.load(sf)
        for n in names:
            if n not in S.files:
                c.err(f"expected.states has no trajectory for entity {n!r}")
            elif len(S[n]) != len(actions):
                c.err(
                    f"entity {n!r} trajectory length {len(S[n])} != actions T {len(actions)}"
                )
    for k in ("tolerance_rigid", "tolerance_particles"):
        if not isinstance(ex.get(k), (int, float)):
            c.err(f"expected.{k} is missing or not a number")
    et = ex.get("entity_types") or {}
    if set(et) != names:
        c.err("expected.entity_types does not match the scene.entities roster")
    sc = ex.get("success_criteria") or {}
    t = sc.get("type")
    if t == "entity_final_pos":
        if not (
            sc.get("entity") in names
            and len(sc.get("target_pos", [])) == 3
            and isinstance(sc.get("pos_radius"), (int, float))
        ):
            c.err(
                "entity_final_pos criterion needs entity, target_pos (3) and pos_radius"
            )
    elif t == "entity_joint":
        if not (
            sc.get("entity") in names
            and isinstance(sc.get("target_qpos"), list)
            and isinstance(sc.get("tolerance"), (int, float))
        ):
            c.err("entity_joint criterion needs entity, target_qpos and tolerance")
    elif t == "particles_centroid":
        if not (
            sc.get("entity") in names
            and len(sc.get("target_pos", [])) == 3
            and isinstance(sc.get("pos_radius"), (int, float))
        ):
            c.err(
                "particles_centroid criterion needs entity, target_pos (3) and pos_radius"
            )
    else:
        c.err(
            f"invalid genesis criterion type: {t!r}"
            "（entity_final_pos | entity_joint | particles_centroid）"
        )
    pv = m.get("provenance") or {}
    for e2 in pv.get("evidence") or []:
        _file(pkg, e2, c, "provenance.evidence")
    if not (pv.get("checks") or {}).get("visual_inspection"):
        c.err("provenance.checks.visual_inspection must be true")
    if (m.get("task") or {}).get("source_video"):
        _file(pkg, m["task"]["source_video"], c, "task.source_video")


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    pkg = Path(argv[1])
    if not pkg.is_dir():
        print(f"not a directory: {pkg}")
        return 2
    c = validate(pkg)
    for w in c.warnings:
        print(f"[warn] {w}")
    if c.errors:
        for e in c.errors:
            print(f"[FAIL] {e}")
        print(f"\nFAIL {pkg.name}: {len(c.errors)} error(s)")
        return 1
    print(
        f"OK {pkg.name}: valid protocol package"
        + (f" ({len(c.warnings)} warning(s))" if c.warnings else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
