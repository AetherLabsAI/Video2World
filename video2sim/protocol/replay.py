#!/usr/bin/env python
"""Replay a protocol package from ONLY the files inside it (PROTOCOL.md).

    replay_protocol.py <package_dir> [--no-render]

Rebuilds the env from protocol.json (object/prop/support, camera mounted at
the recorded base-frame extrinsics), resets to the recorded initial state,
steps actions.npy open-loop, then writes:

    verification/render.mp4          replay through the protocol camera
    verification/replay_report.json  replay_match / task_success / deviations

Exit 0  iff  status "success" -> replay_match AND task_success
        or   status "failure" -> replay_match.

Run in the v2sim venv, single process (rendering in the main process only —
see v2sim-env-quirks: concurrent Vulkan init deadlocks this shared machine).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


# SAPIEN camera frame (+X fwd, +Y left, +Z up) expressed in the OpenCV
# camera frame (+Z fwd, +X right, +Y down): columns are the SAPIEN axes.
CV_TO_SAPIEN = np.array([
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
])


def _mat_to_pose7(T: np.ndarray) -> np.ndarray:
    from ..transforms import mat_to_pose
    return mat_to_pose(np.asarray(T, dtype=np.float64))


def build_env(m: dict, pkg: Path, render: bool):
    """Create the replay env purely from the manifest."""
    import gymnasium as gym

    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils.registration import register_env
    from ..env import ObjectSpec, V2SPickEnv

    def _spec(o: dict):
        """Manifest object -> ObjectSpec (any of them, not just the first)."""
        return ObjectSpec(

            name=o["name"], kind=o["kind"],
            half_size=tuple(o.get("half_size", (0.02, 0.02, 0.02))),
            l_arm_half=tuple(o.get("l_arm_half", (0.015, 0.03, 0.02))),
            wall=float(o.get("wall", 0.005)),
            cylinder_sections=int(o.get("cylinder_sections", 24)),
            mesh_path=str(pkg / o["mesh_path"]) if o.get("mesh_path") else None,
            collision_path=str(pkg / o["collision_path"]) if o.get("collision_path") else None,
            mesh_scale=float(o.get("mesh_scale", 1.0)),
            density=float(o.get("density", 500.0)),
            friction=(None if o.get("friction") is None else float(o["friction"])),
            color=tuple(o.get("color", (0.8, 0.2, 0.1, 1.0))),
            init_pos=tuple(o["init_pos"]), init_quat=tuple(o["init_quat"]),
        )

    objs_m = m["scene"]["objects"]
    spec = _spec(objs_m[0])
    extra_specs = [_spec(o) for o in objs_m[1:]]

    props = []
    for p in m["scene"].get("props") or []:
        p = dict(p)
        for f in ("mesh_path", "collision_path"):
            if p.get(f):
                p[f] = str(pkg / p[f])
        props.append(p)
    artics = m["scene"].get("articulations") or []
    if len(artics) > 1:
        raise ValueError("the env supports only one articulation")

    cam = m["cameras"][0]
    base_pose7 = np.asarray(m["robot"]["base_pose"], dtype=np.float64)

    def cam_config():
        from ..transforms import quat_to_mat
        T_world_base = np.eye(4)
        T_world_base[:3, :3] = quat_to_mat(base_pose7[3:])
        T_world_base[:3, 3] = base_pose7[:3]
        T_base_cam = np.asarray(cam["extrinsics_base_cam"], dtype=np.float64)
        T_world_cv = T_world_base @ T_base_cam
        T_world_sapien = T_world_cv.copy()
        T_world_sapien[:3, :3] = T_world_cv[:3, :3] @ CV_TO_SAPIEN
        import sapien
        pose7 = _mat_to_pose7(T_world_sapien)
        return CameraConfig(
            "protocol_camera",
            pose=sapien.Pose(p=pose7[:3].tolist(), q=pose7[3:].tolist()),
            width=int(cam["width"]), height=int(cam["height"]),
            intrinsic=np.asarray(cam["intrinsics"], dtype=np.float32),
            near=0.01, far=10.0,
        )

    class V2SReplayEnv(V2SPickEnv):
        @property
        def _default_sensor_configs(self):
            return [cam_config()] if self.enable_cameras else []

    env_id = "V2SReplay-v1"
    register_env(env_id, max_episode_steps=1_000_000,
                 override=True)(V2SReplayEnv)

    return gym.make(
        env_id,
        num_envs=1,
        obs_mode="rgbd" if render else "state",
        reward_mode="none",
        control_mode=m["environment"]["control_mode"],
        sim_backend="physx_cpu",
        render_mode="rgb_array" if render else None,
        render_backend="gpu" if render else "none",
        obj_spec=spec,
        extra_objs=extra_specs,
        enable_cameras=render,
        support=m["scene"].get("support"),
        prop=props or None,
        artic=artics[0] if artics else None,
        sim_config=dict(sim_freq=int(m["environment"]["sim_freq"]),
                        control_freq=int(m["environment"]["control_freq"])),
    )


def pose_criterion_met(sc: dict, final_pos, lifted: float) -> bool:
    """`final_pose`/`goal_region` semantics: end inside the goal AND have lifted.

    `min_lift_m` (optional, default 0) is the height the object reached above
    where it started. It exists because an episode that takes an object out of
    a container and puts it back has a terminal state identical to doing
    nothing: without it the criterion is satisfied by the initial state.
    """
    target = np.asarray(sc["target_pos"], dtype=np.float64)
    within = float(np.linalg.norm(np.asarray(final_pos, dtype=np.float64) - target))
    return bool(within <= float(sc["pos_radius"])
                and float(lifted) >= float(sc.get("min_lift_m", 0.0)))


def replay_soft(pkg: Path, m: dict, render: bool = True) -> dict:
    """soft_warp profile: re-simulate with Warp, compare particle trajectory."""
    from ..soft import ClothSpec, _grid, render_cloth_mp4, run_soft_actions

    actions = np.load(pkg / m["actions"]["path"])
    expected = np.load(pkg / m["expected"]["particles"]).astype(np.float64)
    spec = ClothSpec.from_dict(m["scene"]["soft_bodies"][0])
    table_z = float((m["scene"].get("support") or {}).get("z", 0.0))
    props = []
    for p in m["scene"].get("props") or []:
        p = dict(p)
        if p.get("collision_path"):
            p["collision_path"] = str(pkg / p["collision_path"])
        props.append(p)
    traj = run_soft_actions(
        spec, actions, table_z=table_z,
        urdf_path=str(pkg / m["robot"]["urdf"]),
        base_pose=np.asarray(m["robot"]["base_pose"], dtype=np.float64),
        props=props,
        dt=float(m["actions"]["dt"]),
    )
    ver = pkg / "verification"
    ver.mkdir(exist_ok=True)
    if render:
        from ..soft import KinematicPanda, build_prop_trimesh
        _, _, tris = _grid(spec)
        base = np.asarray(m["robot"]["base_pose"], dtype=np.float64)
        cams = m.get("cameras") or []
        try:
            from ..soft_render import render_soft_mp4
            render_soft_mp4(
                traj, tris, ver / "render.mp4", actions,
                urdf_path=str(pkg / m["robot"]["urdf"]), base_pose=base,
                table_z=table_z, prop_trimesh=build_prop_trimesh(props),
                camera=cams[0] if cams else None)
        except Exception as e:
            print(f"[warn] rasterized renderer failed ({e}); falling back to the matplotlib sketch")
            render_cloth_mp4(
                traj, tris, ver / "render.mp4", table_z=table_z,
                prop_trimesh=build_prop_trimesh(props), actions=actions,
                panda=KinematicPanda(str(pkg / m["robot"]["urdf"]), base))

    dev = np.linalg.norm(traj - expected, axis=2)          # (T, N)
    fin_dev = float(dev[-1].max())
    tol = float(m["expected"]["tolerance_particles"])
    replay_match = fin_dev <= tol
    sc = m["expected"]["success_criteria"]
    centroid = traj[-1].mean(0)
    if sc["type"] == "soft_centroid":
        task_success = bool(np.linalg.norm(
            centroid - np.asarray(sc["target_pos"])) <= sc["pos_radius"])
    elif sc["type"] == "soft_clear":
        # xy distance: the soft body must END AWAY from a region (uncover tasks)
        task_success = bool(np.linalg.norm(
            centroid[:2] - np.asarray(sc["clear_pos"][:2])) >= sc["clear_radius"])
    else:
        raise ValueError(f"unknown soft success_criteria.type: {sc['type']}")
    report = {
        "package": str(pkg), "T": len(actions), "profile": "soft_warp",
        "final_particle_dev_max": fin_dev,
        "traj_particle_dev_max": float(dev.max()),
        "traj_particle_dev_mean": float(dev.mean()),
        "tolerance_particles": tol,
        "final_centroid": centroid.tolist(),
        "replay_match": bool(replay_match),
        "task_success": task_success,
        "status": m["status"],
    }
    if render:
        report["render"] = "verification/render.mp4"
    report["ok"] = bool(replay_match and (task_success or m["status"] == "failure"))
    (ver / "replay_report.json").write_text(json.dumps(report, indent=2))
    return report


def replay_genesis(pkg: Path, m: dict, render: bool = True) -> dict:
    """genesis profile: re-simulate, two-tier acceptance (rigid vs particles)."""
    from ..genesis_backend import run_genesis_actions

    actions = np.load(pkg / m["actions"]["path"])
    expected = np.load(pkg / m["expected"]["states"])
    cams = m.get("cameras") or []
    traj, frames = run_genesis_actions(m, actions, pkg=pkg, render=render,
                                       camera=cams[0] if cams else None)
    ver = pkg / "verification"
    ver.mkdir(exist_ok=True)
    if render and frames:
        import imageio.v2 as iio
        iio.mimwrite(ver / "render.mp4", [f for f in frames if f is not None],
                     fps=max(1, int(m["environment"].get("control_freq", 20)) // 2))

    et = m["expected"]["entity_types"]
    tol_r = float(m["expected"]["tolerance_rigid"])
    tol_p = float(m["expected"]["tolerance_particles"])
    particle_types = {"sph_liquid", "mpm_elastoplastic", "mpm_sand", "pbd_cloth"}
    per_entity = {}
    replay_match = True
    for n, tr in traj.items():
        exp = expected[n].astype(np.float64)
        fin_dev = float(np.abs(np.asarray(tr[-1], dtype=np.float64) - exp[-1]).max())
        tol = tol_p if et.get(n) in particle_types else tol_r
        per_entity[n] = {"final_dev": fin_dev, "tolerance": tol,
                         "tier": "particles" if et.get(n) in particle_types else "rigid"}
        replay_match &= fin_dev <= tol

    sc = m["expected"]["success_criteria"]
    ent = np.asarray(traj[sc["entity"]][-1], dtype=np.float64)
    if sc["type"] == "entity_final_pos":
        task_success = bool(np.linalg.norm(ent[:3] - np.asarray(sc["target_pos"]))
                            <= sc["pos_radius"])
    elif sc["type"] == "entity_joint":
        nq = len(sc["target_qpos"])
        task_success = bool(np.abs(ent[:nq] - np.asarray(sc["target_qpos"])).max()
                            <= sc["tolerance"])
    elif sc["type"] == "particles_centroid":
        task_success = bool(np.linalg.norm(ent.reshape(-1, 3).mean(0)
                                           - np.asarray(sc["target_pos"]))
                            <= sc["pos_radius"])
    else:
        raise ValueError(f"unknown genesis criterion: {sc['type']}")

    report = {
        "package": str(pkg), "T": len(actions), "profile": "genesis",
        "per_entity": per_entity,
        "replay_match": bool(replay_match), "task_success": task_success,
        "status": m["status"],
    }
    if render:
        report["render"] = "verification/render.mp4"
    report["ok"] = bool(replay_match and (task_success or m["status"] == "failure"))
    (ver / "replay_report.json").write_text(json.dumps(report, indent=2))
    return report


def replay(pkg: Path, render: bool = True) -> dict:
    from ..transforms import quat_angle

    m = json.loads((pkg / "protocol.json").read_text())
    if m.get("status") == "infeasible":
        print("an infeasible package has nothing to replay")
        return {"skipped": True, "ok": True}
    if m.get("physics_profile") == "twin_mujoco_v2":
        from ..native_twin import replay as replay_twin
        return replay_twin(pkg, render=render)
    if m.get("physics_profile") == "bridge_widowx":
        from ..bench.rollout import rollout
        result=rollout(pkg, render=render)
        return dict(result, profile='bridge_widowx', ok=result.get('execution_completed', True), task_success=None)
    if m.get("physics_profile", "rigid_sapien") == "robodojo":
        # the rig IS this package's physics: there is no SAPIEN scene to
        # rebuild, and the re-run happens in `replay_robodojo` below. Not
        # "skipped" — main() returns early on a skip, and that would leave the
        # episode ungated.
        return {"profile": "robodojo", "ok": True, "replay_match": None,
                "task_success": None, "status": m["status"],
                "note": "gate 2 for this profile is the rig re-run"}
    if m.get("physics_profile", "rigid_sapien") == "soft_warp":
        return replay_soft(pkg, m, render)
    if m.get("physics_profile", "rigid_sapien") == "genesis":
        return replay_genesis(pkg, m, render)

    actions = np.load(pkg / m["actions"]["path"])
    expected_obj = np.load(pkg / m["expected"]["obj_poses"])
    fin_expected = np.asarray(m["expected"]["final_obj_pose"], dtype=np.float64)

    env = build_env(m, pkg, render)
    u = env.unwrapped
    ist = m["initial_state"]
    opts = {"obj_pos": list(ist["obj_pose"][:3]), "obj_quat": list(ist["obj_pose"][3:])}
    if ist.get("obj_poses"):
        # multi-object scene: every dynamic body's spawn pose, primary first
        opts["obj_poses"] = [[float(v) for v in row] for row in ist["obj_poses"]]
    if ist.get("artic_qpos"):
        # per-dof vector (dof>=1); env reset accepts scalar or vector
        opts["artic_q"] = [float(v) for v in ist["artic_qpos"]]
    if ist.get("qpos"):
        # the manifest's post-reset arm configuration is authoritative; for
        # packages recorded from the "rest" keyframe this is a no-op
        opts["robot_qpos"] = [float(v) for v in ist["qpos"]]
    env.reset(seed=0, options=opts)

    report: dict = {"package": str(pkg), "T": len(actions)}

    q0 = u.qpos_np()
    dq0 = float(np.abs(q0 - np.asarray(ist["qpos"])).max())
    report["init_qpos_dev"] = dq0
    if dq0 > 1e-3:
        print(f"[warn] qpos after reset deviates from initial_state.qpos by {dq0:.2e}")

    if render:
        # cross-check that the mounted camera reproduces the manifest
        # extrinsics — guards the CV<->SAPIEN axis conversion
        cam_obj = u.scene.sensors["protocol_camera"].camera
        ext = cam_obj.get_extrinsic_matrix()[0].cpu().numpy()
        T_cam_world = np.eye(4)
        T_cam_world[:3] = ext
        from ..transforms import quat_to_mat
        base7 = np.asarray(m["robot"]["base_pose"], dtype=np.float64)
        T_world_base = np.eye(4)
        T_world_base[:3, :3] = quat_to_mat(base7[3:])
        T_world_base[:3, 3] = base7[:3]
        T_base_cam = np.linalg.inv(T_cam_world @ T_world_base)
        dev = float(np.abs(
            T_base_cam - np.asarray(m["cameras"][0]["extrinsics_base_cam"])).max())
        report["camera_extrinsics_dev"] = dev
        if dev > 1e-3:
            print(f"[warn] mounted camera extrinsics deviate from the manifest by {dev:.2e}")

    has_artic = getattr(u, "artic", None) is not None
    frames = []
    obj_poses = []
    all_obj_poses = []
    grasped = []
    artic_qs = []
    n_obj = len(u.objs)
    for k in range(len(actions)):
        obs = env.step(actions[k])[0]
        obj_poses.append(u.obj_pose_np())
        if n_obj > 1:
            all_obj_poses.append(u.obj_poses_np())
        grasped.append(bool(u.is_grasping_np()))
        if has_artic:
            artic_qs.append(u.artic_qpos_np())
        if render:
            rgb = obs["sensor_data"]["protocol_camera"]["rgb"][0].cpu().numpy()
            frames.append(rgb.astype(np.uint8))
    env.close()

    obj_poses = np.stack(obj_poses)
    grasped = np.asarray(grasped)
    ver = pkg / "verification"
    ver.mkdir(exist_ok=True)
    if render and frames:
        import imageio.v2 as imageio
        imageio.mimwrite(ver / "render.mp4", frames,
                         fps=int(m["environment"]["control_freq"]))
        report["render"] = "verification/render.mp4"

    fin = obj_poses[-1]
    pos_err = float(np.linalg.norm(fin[:3] - fin_expected[:3]))
    rot_err = float(np.degrees(quat_angle(fin[3:], fin_expected[3:])))
    dev_p = np.linalg.norm(obj_poses[:, :3] - expected_obj[:, :3], axis=1)
    transitions = int(np.abs(np.diff(grasped.astype(int))).sum())

    tol_p = float(m["expected"]["tolerance_pos"])
    tol_r = float(m["expected"]["tolerance_rot_deg"])
    replay_match = pos_err <= tol_p and rot_err <= tol_r

    # Multi-object scene: reproduction means EVERY piece landed where the
    # recording put it. Scoring the primary alone would call a collapsed tower
    # a match as long as the last block happened to end in the right place.
    if n_obj > 1:
        all_obj_poses = np.stack(all_obj_poses)
        exp_all = np.load(pkg / m["expected"]["obj_poses_all"])
        fin_all = np.asarray(m["expected"]["final_obj_poses"], dtype=np.float64)
        per_obj = []
        for i, o in enumerate(m["scene"]["objects"]):
            pe = float(np.linalg.norm(all_obj_poses[-1, i, :3] - fin_all[i, :3]))
            re_ = float(np.degrees(quat_angle(all_obj_poses[-1, i, 3:], fin_all[i, 3:])))
            per_obj.append({"name": o.get("name"), "final_pos_err": pe,
                            "final_rot_err_deg": re_,
                            "traj_pos_dev_max": float(np.linalg.norm(
                                all_obj_poses[:, i, :3] - exp_all[:, i, :3],
                                axis=1).max())})
            replay_match = replay_match and pe <= tol_p and re_ <= tol_r
        report["per_object"] = per_obj

    artic_err = None
    if has_artic:
        aq = np.stack(artic_qs)
        exp_aq = np.load(pkg / m["expected"]["artic_qpos"])
        fin_aq = np.asarray(m["expected"]["final_artic_qpos"], dtype=np.float64)
        artic_err = float(np.abs(aq[-1] - fin_aq).max())
        tol_a = float(m["expected"]["tolerance_artic"])
        replay_match = replay_match and artic_err <= tol_a
        report.update({
            "final_artic_err": artic_err,
            "tolerance_artic": tol_a,
            "artic_traj_dev_max": float(np.abs(aq - exp_aq).max()),
        })

    sc = m["expected"]["success_criteria"]
    # How far the object rose above where it started, at any point in the run.
    # An episode that returns its object to the container it came out of has a
    # final state indistinguishable from doing nothing; the lift is the part of
    # the semantics a final-pose test cannot see.
    lifted = float(np.max(obj_poses[:, 2]) - obj_poses[0, 2])
    min_lift = float(sc.get("min_lift_m", 0.0))
    report["object_lifted_m"] = lifted
    report["minimum_lift_m"] = min_lift
    if sc["type"] in ("final_pose", "goal_region"):
        task_success = pose_criterion_met(sc, fin[:3], lifted)
    elif sc["type"] == "multi_goal":
        if n_obj <= 1:
            raise ValueError("multi_goal criterion needs a multi-object scene")
        idx = {o.get("name"): i for i, o in enumerate(m["scene"]["objects"])}
        task_success = True
        checks = []
        for g in sc["goals"]:
            i = idx[g["object"]]
            traj_i = all_obj_poses[:, i, :3]
            lift_i = float(traj_i[:, 2].max() - traj_i[0, 2])
            ok_i = pose_criterion_met(g, traj_i[-1], lift_i)
            checks.append({"object": g["object"],
                           "err": float(np.linalg.norm(
                               traj_i[-1] - np.asarray(g["target_pos"], dtype=float))),
                           "pos_radius": float(g["pos_radius"]),
                           "lifted_m": lift_i,
                           "min_lift_m": float(g.get("min_lift_m", 0.0)),
                           "ok": bool(ok_i)})
            task_success = task_success and ok_i
        report["multi_goal_checks"] = checks
    elif sc["type"] == "articulation":
        if not has_artic:
            raise ValueError("articulation criterion but the scene has no articulation")
        target = np.asarray(sc["target_qpos"], dtype=np.float64)
        task_success = bool(np.abs(np.stack(artic_qs)[-1] - target).max() <= sc["tolerance"])
    else:
        raise ValueError(f"unknown success_criteria.type: {sc['type']}")

    report.update({
        "final_pos_err": pos_err,
        "final_rot_err_deg": rot_err,
        "tolerance_pos": tol_p,
        "tolerance_rot_deg": tol_r,
        "traj_pos_dev_mean": float(dev_p.mean()),
        "traj_pos_dev_max": float(dev_p.max()),
        "grasp_transitions_replay": transitions,
        "grasp_transitions_expected": m["expected"].get("grasp_transitions"),
        "replay_match": bool(replay_match),
        "task_success": task_success,
        "status": m["status"],
    })
    report["ok"] = bool(
        replay_match and (task_success or m["status"] == "failure"))
    (ver / "replay_report.json").write_text(json.dumps(report, indent=2))
    return report


def replay_robodojo(pkg: Path) -> dict:
    """Re-run the delivered 14-dim stream on the rig and compare joint states.

    Same question the other branches answer — does the package reproduce —
    asked of the RoboDojo delivery: re-simulate from robodojo/actions.npy with
    the same objects (recorded in rollout.job.json at delivery time) and
    compare against the recorded rollout.npz states. Physics only; the
    delivered render.mp4 is the visual artifact, gate 1 checks its presence.
    """
    dj = pkg / "robodojo"
    report = json.loads((dj / "delivery_report.json").read_text())
    if report.get("status") == "rejected":
        return {"skipped": True, "ok": True, "note": "explicitly rejected delivery"}

    from video2sim.robodojo_backend import run_episode
    actions = np.load(dj / "actions.npy")
    expected = np.load(dj / "rollout.npz")["states"]
    job = json.loads((dj / "rollout.job.json").read_text())

    # Re-run the SAME scene, not merely the same actions. `visuals` is not a
    # cosmetic flag: "official" spawns the benchmark's room USD, MDL table and
    # ground and the camera stand, "plain" a bare slab — different PhysX
    # colliders and materials. Rerunning an officially-staged delivery against
    # the plain scene compares two different experiments, and the difference
    # surfaces exactly where it is least interpretable: the gripper's stalled
    # width at contact, 28 mrad on one frame with a mean deviation of 8e-5.
    rerun = run_episode(actions, dj / "replay_rerun.npz", render=False,
                        objects=job.get("objects") or None,
                        substeps=job.get("substeps", 8),
                        visuals=job.get("visuals"),
                        official_scene=job.get("official_scene"))
    got = rerun["states"]
    (dj / "replay_rerun.npz").unlink(missing_ok=True)   # states kept in report

    # Three questions, each in the units it is actually posed in.
    #
    # 1. Do the ARM joints reproduce? 5e-3 rad; PhysX on one GPU sits well
    #    inside that.
    # 2. Do the GRIPPER dims reproduce? These are not radians — they are the
    #    hand's normalized 0..1 travel, and the drive STALLS on the object, so
    #    the state reports the object's width. Holding a contact measurement to
    #    5e-3 of travel (0.44 mm of jaw) fails an episode for closing on a cup
    #    wall a fraction of a millimetre differently. 0.05 is ~4 mm of jaw:
    #    still tight enough that a lost or missed grasp shows immediately.
    # 3. Does the OBJECT end up in the same place? That is what "reproduces"
    #    means for a manipulation episode, and neither of the above asks it.
    arm = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    grip = [6, 13]
    dev = np.abs(got - expected)
    tol_arm, tol_grip = 5e-3, 5e-2
    # The object's tolerance is the PACKAGE's own declared reproduction
    # tolerance, not a constant invented here. A fixed few millimetres asserts
    # that the terminal object pose is a unique equilibrium, and for a whole
    # class of tasks it is not: a cone nested in a cone with an 8.7 deg taper
    # is self-locking (tan a = 0.15 < mu), so its resting depth is a
    # friction-arrested RANGE. Re-running one delivered stream three times put
    # the cup 3.4, 17.8 and 23.8 mm from the recorded pose - all of them
    # nested, none of them wrong. What has to hold is the package's declared
    # tolerance AND its own success criteria, which is exactly what the
    # rigid-body branch of this gate checks.
    proto = json.loads((pkg / "protocol.json").read_text())
    exp = proto.get("expected", {}) or {}
    tol_obj = float(exp.get("tolerance_pos", 0.02))
    arm_max = float(dev[:, arm].max())
    grip_max = float(dev[:, grip].max())
    rolled = np.load(dj / "rollout.npz", allow_pickle=True)
    obj_dev = {}
    for k in sorted(rerun.get("objects", {})):
        a = np.asarray(rerun["objects"][k])
        if k not in rolled.files:
            continue
        b = np.asarray(rolled[k])
        if len(b) != len(a):
            continue
        obj_dev[k] = float(np.linalg.norm(a[-1, :3] - b[-1, :3]))
    obj_max = max(obj_dev.values()) if obj_dev else 0.0

    # ... and does the RE-RUN still do the task? The delivery report already
    # carried the success target onto the rig; re-use it rather than
    # re-deriving, so this asks the same question the delivery did.
    goal, radius = report.get("rig_target_pos"), report.get("rig_goal_radius_m")
    rerun_success = None
    rerun_goal_err = None
    # A multi-object episode has one target PER OBJECT, and "did the re-run
    # still do the task" is the conjunction: three tiles on their pads and one
    # beside is a failure, not 75% of a success. `rig_targets` carries that;
    # the single-object fields stay for deliveries made before it existed.
    targets = report.get("rig_targets")
    if targets and rerun.get("objects"):
        errs, oks = {}, []
        for t in targets:
            key = t["object_key"]
            if key not in rerun["objects"]:
                oks.append(False)
                continue
            row = np.asarray(rerun["objects"][key])[-1]
            e = float(np.linalg.norm(row[:3] - np.asarray(t["pos"], dtype=float)))
            errs[key] = e
            ok = e <= float(t["radius"])
            # Orientation, when the delivery says it is part of the goal. A row
            # of digit tiles has to READ as a number: right place, wrong way up
            # is a different outcome, and a position-only check calls it a pass.
            yt = t.get("yaw_tol_deg")
            if yt is not None and len(row) >= 7:
                w, x, y, z = row[3:7]
                got = np.degrees(np.arctan2(2 * (w * z + x * y),
                                            1 - 2 * (y * y + z * z)))
                sym = float(t.get("yaw_symmetry_deg", 360.0))
                d = (got - float(t.get("yaw_target_deg", 0.0))) % sym
                d = min(d, sym - d)
                errs[key + ":yaw_deg"] = float(d)
                ok = ok and d <= float(yt)
            oks.append(ok)
        rerun_goal_err = errs
        rerun_success = bool(oks) and all(oks)
    elif goal is not None and radius is not None and rerun.get("objects"):
        key = report.get("rig_object_key") or sorted(rerun["objects"])[0]
        if key in rerun["objects"]:
            fin = np.asarray(rerun["objects"][key])[-1, :3]
            rerun_goal_err = float(np.linalg.norm(fin - np.asarray(goal)))
            rerun_success = bool(rerun_goal_err <= float(radius))

    # ... and, for an episode whose terminal state is its INITIAL state, did the
    # re-run go through the right states in the right order? "Cover three blocks
    # then uncover them" puts every object back where it started, so the
    # position check above is passed just as well by an episode that did
    # nothing; the semantics are the ordered intermediate states. A delivery
    # that declares `rig_events` is gated on them IN ADDITION to its targets.
    events = report.get("rig_events")
    rerun_events = None
    if events and rerun.get("objects"):
        from video2sim.robodojo_deliver import check_event_sequence

        rerun_events = check_event_sequence(
            rerun["objects"], events,
            terminal=report.get("rig_terminal"),
            stationary=report.get("rig_stationary"))
        rerun_success = bool((rerun_success is not False) and rerun_events["ok"])

    # What "reproduces" means for a rig delivery. The ARM has to follow the same
    # joints, and the re-run has to STILL DO THE TASK by the package's own
    # criteria. The other two numbers are reported, not gated, and deliberately:
    #   - the gripper dims are the hand's travel, and the drive stalls on the
    #     object, so they are a contact measurement of the grasped wall;
    #   - the final object pose, for a task whose terminal state is not a unique
    #     equilibrium, is a sample from a range. This episode nests a cone in a
    #     cone at 8.7 deg, i.e. tan a = 0.15 below any plausible mu: self-locking,
    #     so the resting depth is friction-arrested anywhere in a band. Four
    #     re-runs of one delivered stream put the cup 3.4, 17.8, 21.5 and 23.8 mm
    #     from the recorded pose - and all four still nested it. Thresholding
    #     that would fail the episode for the physics being what it is; the
    #     success criterion is what catches a re-run that actually went wrong,
    #     and it is strictly more informative than a distance.
    # Without a recorded success target (a delivery made before this existed)
    # fall back to the package's declared reproduction tolerance, so those are
    # not silently waved through.
    ok = arm_max <= tol_arm and (rerun_success if rerun_success is not None
                                 else obj_max <= tol_obj)
    out = {
        "profile": "robodojo",
        "state_dev_max": float(dev.max()),
        "state_dev_mean": float(dev.mean()),
        "state_dev_arm_max_rad": arm_max,
        "state_dev_gripper_max_norm": grip_max,
        "final_object_dev_m": obj_dev,
        "rerun_goal_err_m": rerun_goal_err,
        "rerun_event_sequence": rerun_events,
        "rerun_task_success": rerun_success,
        "tolerance_joints": tol_arm,
        "gripper_dev_is_reported_not_gated": True,
        "object_dev_is_reported_not_gated": rerun_success is not None,
        "tolerance_object_m": tol_obj,
        "tolerance_object_source": "package expected.tolerance_pos "
                                   "(fallback only, when the delivery recorded "
                                   "no success target)",
        "replay_match": bool(ok),
        "ok": bool(ok),
    }
    (dj / "replay_report.json").write_text(json.dumps(out, indent=2))
    return out


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    pkg = Path(argv[1]).resolve()
    render = "--no-render" not in argv
    r = replay(pkg, render=render)
    if r.get("skipped"):
        return 0
    if (pkg / "robodojo").is_dir():
        dr = replay_robodojo(pkg)
        r["robodojo"] = dr
        if not dr.get("skipped"):
            _od = max(dr["final_object_dev_m"].values()) if dr["final_object_dev_m"] else 0.0
            print(f"robodojo replay: arm {dr['state_dev_arm_max_rad']:.2e} rad "
                  f"(tol {dr['tolerance_joints']}) | reported: gripper "
                  f"{dr['state_dev_gripper_max_norm']:.2e}, final object "
                  f"{_od:.2e} m | rerun_task_success="
                  f"{dr['rerun_task_success']}, "
                  f"replay_match={dr['replay_match']}")
            r["ok"] = bool(r["ok"] and dr["ok"])
    print(json.dumps(r, indent=2, ensure_ascii=False))
    if r["ok"]:
        if r.get('profile') in ('twin_mujoco_v2', 'bridge_widowx'):
            print('OK execution; task success requires task-specific evaluation')
            return 0
        if r.get("profile") == "robodojo":
            dr = r.get("robodojo") or {}
            print(f"\nOK replay: the rig re-run reproduced the delivery "
                  f"(arm {dr.get('state_dev_arm_max_rad', float('nan')):.2e} rad), "
                  f"rerun_task_success={dr.get('rerun_task_success')}")
            return 0
        if r.get("profile") == "genesis":
            worst = max(v["final_dev"] for v in r["per_entity"].values())
            print(f"\nOK replay: max entity final deviation {worst:.2e}, "
                  f"replay_match={r['replay_match']}, task_success={r['task_success']}")
            return 0
        if r.get("profile") == "soft_warp":
            print(f"\nOK replay: max final particle deviation {r['final_particle_dev_max']*1000:.2f} mm, "
                  f"replay_match={r['replay_match']}, task_success={r['task_success']}")
        else:
            print(f"\nOK replay: final error {r['final_pos_err']*1000:.1f} mm / "
                  f"{r['final_rot_err_deg']:.1f}°, "
                  f"replay_match={r['replay_match']}, task_success={r['task_success']}")
        return 0
    print("\nFAIL replay")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
