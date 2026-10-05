"""Evaluate one simulator package on one benchmark sample.

    v2s bench <package> --sample <dir> [--reference <human_package>] [--out report.json]

Protocol (per sample):
  1. build the scene from the package alone (bridge_widowx profile), robot
     base at the origin — alignment on base / gravity / support plane happens
     in `scene_in_base_frame`; no per-object ICP;
  2. initialise the arm on the real episode's first EE state, settle;
  3. execute the withheld real trajectory (absolute EE targets, the recorded
     gripper command);
  4. record every object's 6-D trajectory, the terminal state, run status;
  5. render through the real (hidden) camera at canonical frames.

Metrics written to the report:
  build              validate + build + reset + run + render all succeeded
  chamfer            surface point cloud of the delivered scene vs the reference
                     scene (reference = the human-built sim; declared as such)
  lpips              rendered frames vs the real video frames at k in {0, T/2, T-1}
  ape                object trajectory vs the reference sim's trajectory under
                     the same actions (reference = human sim; NOT ground truth)
  replay_success     phi_i on the terminal state (object correspondence by
                     initial position)
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np

from . import dataset, metrics
from .protocol import PackageError, load_manifest, scene_in_base_frame, validate_manifest

CANONICAL_FRACS = (0.0, 0.5, 1.0)


# ------------------------------------------------------------------ rollout
def run_hidden_trajectory(scene: dict, camera: dict | None, proprio: np.ndarray,
                          action: np.ndarray, render: bool = True,
                          mode: str = "absolute", settle_tail_s: float = 1.0,
                          obs_mode: str = "rgb") -> dict:
    """Build `scene`, execute the hidden trajectory, return trajectories + frames."""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  (registers envs)
    from . import bridge_env  # noqa: F401  (registers V2SBridge-v1)
    from .widowx_replay import WidowXRig

    env = gym.make("V2SBridge-v1", obs_mode=obs_mode if render else "state", num_envs=1,
                   sim_backend="physx_cpu", render_backend="gpu" if render else "none",
                   scene_spec=scene, camera=camera if render else None,
                   enable_cameras=render)
    try:
        obs, _ = env.reset(seed=0)
        u = env.unwrapped
        rig = WidowXRig(env)
        init = rig.init_from_state(proprio[0])
        if not init["ok"]:
            raise RuntimeError(f"initial EE state unreachable (ik residual {init['ik_err']:.3g})")
        # settle again after the arm moved (it may have touched something)
        u._settle(0.3)
        init_poses = u.all_obj_poses_np()
        names = list(init_poses)
        T = len(action)
        traj = {n: [] for n in names}
        ee = []
        grasp = {n: [] for n in names}
        frames = {}
        want = sorted({int(round(f * (T - 1))) for f in CANONICAL_FRACS})

        def grab(k):
            if render:
                o = env.unwrapped.get_obs()
                frames[k] = o["sensor_data"]["eval_camera"]["rgb"][0].cpu().numpy().astype(np.uint8)

        if 0 in want:
            grab(0)
        for k in range(T):
            if mode == "absolute":
                rig.step_to_state(proprio[min(k + 1, len(proprio) - 1)], grip_cmd=float(action[k][6]))
            else:
                rig.step(action[k])
            for n in names:
                traj[n].append(u.obj_pose_np(n))
                grasp[n].append(u.is_grasping_np(n))
            ee.append(rig.ee_state())
            if (k + 1) in want and (k + 1) != 0:
                grab(k + 1)
        # let the terminal state come to rest, then check it did
        u._settle(settle_tail_s)
        final = u.all_obj_poses_np()
        vel = {n: u.obj_velocity_np(n) for n in names}
        contacts = {}
        ents = list(u.objs) + list(u.props)
        for i, a in enumerate(ents):
            for b in ents[i + 1:]:
                try:
                    f = u.contact_force(a, b)
                except Exception:
                    f = 0.0
                if f > 0.05:
                    contacts[f"{a}|{b}"] = float(f)
        return {
            "names": names, "T": T,
            "init_poses": {n: p.tolist() for n, p in init_poses.items()},
            "traj": {n: np.stack(v) for n, v in traj.items()},
            "grasp": {n: np.asarray(v) for n, v in grasp.items()},
            "ee": np.stack(ee),
            "final_poses": {n: p.tolist() for n, p in final.items()},
            "final_velocity": vel, "contacts": contacts,
            "frames": frames, "init": init,
        }
    finally:
        env.close()


# ---------------------------------------------------------------- surfaces
def scene_point_cloud(scene: dict, n_per_entity: int = 4000, include_support: bool = True,
                      roi: tuple[float, float] = (0.0, 0.9)) -> np.ndarray:
    """Sample surface points of every entity at its INITIAL pose (base frame)."""
    import trimesh
    from scipy.spatial.transform import Rotation as R
    from .bridge_env import library_model_dir, _visual_file

    pts = []

    def mesh_of(e):
        k = e["kind"]
        if k == "box":
            return trimesh.creation.box(extents=2 * np.asarray(e["half_size"], dtype=float))
        if k == "cylinder":
            return trimesh.creation.cylinder(radius=float(e["half_size"][0]),
                                             height=2 * float(e["half_size"][2]))
        if k in ("sphere", "container"):
            from .silhouette import load_mesh
            return load_mesh(e)
        if k == "mesh":
            m = trimesh.load(e.get("mesh_path") or e["collision_path"], force="mesh")
            m.apply_scale(float(e.get("scale", 1.0)))
            return m
        if k == "library":
            d = library_model_dir(e["library_id"])
            try:
                m = trimesh.load(_visual_file(d), force="mesh")
            except Exception:
                m = trimesh.load(str(d / "collision.obj"), force="mesh")
            m.apply_scale(float(e.get("scale", 1.0)))
            return m
        raise ValueError(k)

    for e in (scene.get("objects") or []) + (scene.get("props") or []):
        m = mesh_of(e)
        p, _ = trimesh.sample.sample_surface(m, n_per_entity, seed=0)   # deterministic: self-chamfer == 0
        q = np.asarray(e.get("quat", (1, 0, 0, 0)), dtype=float)
        Rm = R.from_quat(np.r_[q[1:], q[0]]).as_matrix()
        pts.append(p @ Rm.T + np.asarray(e["pos"], dtype=float))
    if include_support and scene.get("support"):
        s = scene["support"]
        cx, cy = s.get("center", (0.3, 0.0))
        lx, ly = s.get("size", (1.0, 1.0))
        n = n_per_entity * 2
        xy = np.random.default_rng(0).uniform([-lx / 2, -ly / 2], [lx / 2, ly / 2], (n, 2))
        pts.append(np.c_[xy + [cx, cy], np.full(n, float(s["z"]))])
    if not pts:
        return np.zeros((0, 3))
    P = np.concatenate(pts)
    # region of interest: the reachable workspace in front of the robot
    r = np.linalg.norm(P[:, :2], axis=1)
    return P[(r >= roi[0]) & (r <= roi[1])]


# ----------------------------------------------------------------- phi
def eval_phi(phi: dict, final: dict[str, list], init: dict[str, list], names_map: dict,
             contacts: dict, scene: dict) -> dict:
    """Success predicate on the terminal state. `names_map` maps reference
    object names (used in phi) to names in the evaluated scene."""
    def pose(ref_name):
        n = names_map.get(ref_name)
        if n is None or n not in final:
            return None
        return np.asarray(final[n]), np.asarray(init[n])

    t = phi["type"]
    out = {"type": t}
    if t == "moved":
        r = pose(phi["source"])
        if r is None:
            return {**out, "success": False, "reason": "source object not matched"}
        d = float(np.linalg.norm(r[0][:3] - r[1][:3]))
        out.update(dist_m=d, success=d >= float(phi.get("min_dist", 0.05)))
    elif t == "in_region":
        r = pose(phi["source"])
        if r is None:
            return {**out, "success": False, "reason": "source object not matched"}
        c = np.asarray(phi["center"], dtype=float)
        d = float(np.linalg.norm(r[0][:len(c)] - c))
        ok = d <= float(phi["radius"])
        if "max_z" in phi:
            ok = ok and float(r[0][2]) <= float(phi["max_z"])
        out.update(dist_m=d, success=bool(ok))
    elif t == "on_top":
        s, tg = pose(phi["source"]), pose(phi["target"])
        if s is None or tg is None:
            return {**out, "success": False, "reason": "source/target not matched"}
        dxy = float(np.linalg.norm(s[0][:2] - tg[0][:2]))
        dz = float(s[0][2] - tg[0][2])
        key = f"{names_map[phi['source']]}|{names_map[phi['target']]}"
        key2 = f"{names_map[phi['target']]}|{names_map[phi['source']]}"
        touching = key in contacts or key2 in contacts
        ok = dxy <= float(phi.get("xy_radius", 0.08)) and dz > 0 and (touching or not phi.get("require_contact", True))
        out.update(dxy_m=dxy, dz_m=dz, touching=touching, success=bool(ok))
    elif t == "lifted":
        r = pose(phi["source"])
        if r is None:
            return {**out, "success": False, "reason": "source object not matched"}
        dz = float(r[0][2] - r[1][2])
        out.update(dz_m=dz, success=dz >= float(phi.get("min_dz", 0.05)))
    else:
        raise ValueError(f"unknown phi type {t}")
    return out


# ------------------------------------------------------------------ driver
def evaluate(pkg: Path, sample: Path, reference: Path | None = None, render: bool = True,
             mode: str = "absolute", out: Path | None = None, save_frames: bool = True) -> dict:
    pkg, sample = Path(pkg), Path(sample)
    hid = dataset.load_hidden(sample)
    rep: dict = {"package": str(pkg), "sample": hid["meta"]["sample_id"],
                 "category": hid["meta"].get("category"), "mode": mode,
                 "build": False, "stage": "validate"}
    t0 = time.time()
    gt_traj = None
    try:
        m = load_manifest(pkg)
        errs = validate_manifest(pkg, m)
        rep["validate_errors"] = errs
        rep["status"] = m.get("status")
        if errs:
            raise PackageError("; ".join(errs))
        if m.get("status") == "infeasible":
            rep["stage"] = "infeasible"
            rep["replay_success"] = False
            return _finish(rep, out, t0)
        scene = scene_in_base_frame(m, pkg)
        rep["stage"] = "run"
        cam = hid["camera"]
        if cam is None and render:
            rep["warning"] = "no hidden camera.json — LPIPS skipped"
        roll = run_hidden_trajectory(scene, cam, hid["proprio"], hid["action"],
                                     render=render and cam is not None, mode=mode)
        rep["build"] = True
        rep["stage"] = "metrics"
        rep["init_ik"] = {k: v for k, v in roll["init"].items() if k != "qpos"}
        rep["objects"] = roll["names"]
        rep["final_poses"] = roll["final_poses"]
        rep["init_poses"] = roll["init_poses"]
        rep["final_velocity"] = roll["final_velocity"]
        rep["quiescent"] = bool(max(roll["final_velocity"].values(), default=0.0) < 5e-3)
        rep["contacts"] = roll["contacts"]
        rep["grasp_steps"] = {n: int(g.sum()) for n, g in roll["grasp"].items()}

        # ---- LPIPS vs real frames
        if roll["frames"]:
            real = dataset.load_frames(sample)
            ks = sorted(roll["frames"])
            a = np.stack([metrics.resize_like_bridge(roll["frames"][k], real.shape[1]) for k in ks])
            b = np.stack([real[min(k, len(real) - 1)] for k in ks])
            d = metrics.lpips_batch(a, b)
            rep["lpips"] = {"mean": float(d.mean()), "per_frame": {str(k): float(v) for k, v in zip(ks, d)}}
            if save_frames and out is not None:
                _save_sheet(a, b, Path(out).with_suffix(".png"))

        # ---- reference-relative metrics: quasi ground truth from the video
        # (model-based fits, independent of any simulator) when the sample
        # has it, otherwise the human-built simulator.
        gt_scene = (sample / "hidden" / "scene_gt.json")
        gt_obj = (sample / "hidden" / "objects_6d.npz")
        names_map = {n: n for n in roll["names"]}
        if gt_scene.exists() and gt_obj.exists():
            g = json.loads(gt_scene.read_text())
            gz = np.load(gt_obj)
            gt_traj = {n: gz[n] for n in g["object_frame0"]}
            g_scene = {"support": g.get("support"), "props": g["props"],
                       "objects": [dict(o, pos=g["object_frame0"][o["name"]][:3], quat=g["object_frame0"][o["name"]][3:])
                                   for o in g["objects"]]}
            pa = scene_point_cloud(scene)
            pb = scene_point_cloud(g_scene)
            rep["chamfer"] = {**metrics.chamfer(pa, pb), "reference": "video_model_fit", "n_a": len(pa), "n_b": len(pb)}
            names_map = metrics.match_objects(
                {n: np.asarray(p) for n, p in roll["init_poses"].items()},
                {n: np.asarray(p[0]) for n, p in gt_traj.items()})
            rep["object_match"] = names_map
            sym = {o["name"]: o.get("symmetry") for o in g["objects"]}
            ape = {}
            for rn, an in names_map.items():
                ape[rn] = None if an is None else metrics.pose_ape(roll["traj"][an], gt_traj[rn], symmetry=sym.get(rn))
            valid = [v for v in ape.values() if v]
            rep["ape"] = {"per_object": ape, "reference": "video_model_fit",
                          "trans_ape_cm": float(np.mean([v["trans_ape_cm"] for v in valid])) if valid else None,
                          "rot_ape_deg": float(np.mean([v["rot_ape_deg"] for v in valid])) if valid else None}
        elif reference is not None:
            ref_m = load_manifest(reference)
            ref_scene = scene_in_base_frame(ref_m, reference)
            pa = scene_point_cloud(scene)
            pb = scene_point_cloud(ref_scene)
            rep["chamfer"] = {**metrics.chamfer(pa, pb), "reference": "human_sim", "n_a": len(pa), "n_b": len(pb)}
            ref_roll_path = Path(reference) / "verification" / f"rollout_{hid['meta']['sample_id']}_{mode}.npz"
            ref_roll = _cached_reference_rollout(ref_roll_path, ref_scene, hid, mode)
            names_map = metrics.match_objects(
                {n: np.asarray(p) for n, p in roll["init_poses"].items()},
                {n: np.asarray(p) for n, p in ref_roll["init_poses"].items()})
            rep["object_match"] = names_map
            sym = {o["name"]: o.get("symmetry") for o in ref_m["scene"].get("objects", [])}
            ape = {}
            for rn, an in names_map.items():
                ape[rn] = None if an is None else metrics.pose_ape(roll["traj"][an], ref_roll["traj"][rn], symmetry=sym.get(rn))
            valid = [v for v in ape.values() if v]
            rep["ape"] = {"per_object": ape, "reference": "human_sim",
                          "trans_ape_cm": float(np.mean([v["trans_ape_cm"] for v in valid])) if valid else None,
                          "rot_ape_deg": float(np.mean([v["rot_ape_deg"] for v in valid])) if valid else None}

        # ---- phi under the withheld real trajectory ("hidden replay")
        if hid["phi"] is not None:
            r = eval_phi(hid["phi"], roll["final_poses"], roll["init_poses"], names_map,
                         roll["contacts"], scene)
            rep["phi_hidden"] = r
            rep["hidden_replay_success"] = bool(r["success"])
        # ---- semantic success: the package's OWN actions in its OWN scene,
        #      judged by the evaluator's phi (video2sim contract: same task
        #      outcome, not the same path)
        act = m.get("actions")
        if act and act.get("path"):
            A = np.load(pkg / act["path"])
            rep["actions_T"] = int(len(A))
            sem = run_agent_actions(scene, A, render=False)
            rep["semantic"] = {"init_poses": sem["init_poses"], "final_poses": sem["final_poses"],
                               "quiescent": bool(max(sem["final_velocity"].values(), default=0.0) < 5e-3),
                               "contacts": sem["contacts"], "grasp_steps": {n: int(g.sum()) for n, g in sem["grasp"].items()}}
            if hid["phi"] is not None:
                r2 = eval_phi(hid["phi"], sem["final_poses"], sem["init_poses"], names_map, sem["contacts"], scene)
                rep["phi"] = r2
                rep["replay_success"] = bool(r2["success"])
            if gt_traj is not None:
                # terminal-state agreement with the tracked real outcome
                fin = {}
                for rn, an in names_map.items():
                    if an is None or rn not in gt_traj:
                        continue
                    fin[rn] = float(np.linalg.norm(np.asarray(sem["final_poses"][an][:3]) - np.asarray(gt_traj[rn][-1][:3])) * 100)
                rep["semantic"]["final_pos_err_cm"] = fin
        else:
            rep["phi"] = {"type": "none", "success": False, "reason": "package delivers no actions"}
            rep["replay_success"] = False
        rep["stage"] = "done"
    except Exception as e:
        rep["error"] = f"{type(e).__name__}: {e}"
        rep["traceback"] = traceback.format_exc()[-2000:]
        rep["replay_success"] = False
    return _finish(rep, out, t0)


def run_agent_actions(scene: dict, actions: np.ndarray, render: bool = False, camera: dict | None = None) -> dict:
    """Execute the package's OWN action stream — (T,7) absolute EE states
    [x y z roll pitch yaw grip_cmd] in the base frame at 5 Hz — in its scene.
    Row 0 is the initial EE state (reached by IK before the first step)."""
    A = np.asarray(actions, dtype=np.float64)
    states = A.copy()
    states[:, 6] = 1.0                       # init: gripper open unless told otherwise
    states[0, 6] = A[0, 6]
    return run_hidden_trajectory(scene, camera, states, A, render=render, mode="absolute")


def _cached_reference_rollout(path: Path, ref_scene: dict, hid: dict, mode: str) -> dict:
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return {"init_poses": z["init_poses"].item(), "traj": z["traj"].item()}
    roll = run_hidden_trajectory(ref_scene, None, hid["proprio"], hid["action"], render=False, mode=mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, init_poses=roll["init_poses"], traj=roll["traj"])
    return roll


def _save_sheet(a: np.ndarray, b: np.ndarray, path: Path) -> None:
    from PIL import Image
    rows = [np.concatenate([x, y], 1) for x, y in zip(a, b)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, 0)).save(path)


def _finish(rep: dict, out: Path | None, t0: float) -> dict:
    rep["seconds"] = round(time.time() - t0, 1)
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(rep, indent=2, default=_json_default))
    return rep


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="v2s bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--reference", default=None, help="human-built package for chamfer/APE")
    ap.add_argument("--mode", default="absolute", choices=["absolute", "delta"])
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rep = evaluate(Path(a.package), Path(a.sample), Path(a.reference) if a.reference else None,
                   render=not a.no_render, mode=a.mode, out=Path(a.out) if a.out else None)
    keys = ["sample", "status", "build", "stage", "replay_success", "quiescent"]
    print({k: rep.get(k) for k in keys})
    if "lpips" in rep:
        print("lpips", round(rep["lpips"]["mean"], 4))
    if "chamfer" in rep:
        print("chamfer_cm", round(rep["chamfer"]["chamfer_m"] * 100, 2))
    if "ape" in rep:
        print("ape", rep["ape"]["trans_ape_cm"], rep["ape"]["rot_ape_deg"])
    if "phi" in rep:
        print("phi", rep["phi"])
    if "error" in rep:
        print("ERROR", rep["error"])
    return 0 if rep.get("build") or rep.get("stage") == "infeasible" else 1


if __name__ == "__main__":
    raise SystemExit(main())
