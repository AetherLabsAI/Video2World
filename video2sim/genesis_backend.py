"""Genesis backend (protocol physics_profile "genesis", v3.0).

One scene API covering the modalities the other two profiles cannot:
fluids (SPH), granular/plastic (MPM — via entity type passthrough), cloth
(PBD, native robot-link pinning), articulated objects with FREE roots
(bottle caps, scissors) and any number of rigid bodies — alongside the
Franka replaying the same (T, 8) ``pd_joint_pos`` action stream as every
other profile.

Determinism (measured on this machine, scripts/35): rigid path bit-exact,
particle solvers ~1e-5 m run-to-run (GPU atomics) — hence the two-tier
acceptance in replay: `tolerance_rigid` vs `tolerance_particles`.

Rendering: Genesis Rasterizer (madrona) only — no ray tracing, no Vulkan.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _np(x):
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def _squeeze_env(a: np.ndarray) -> np.ndarray:
    """Genesis returns (n_envs, ...) for some accessors — drop the env dim."""
    a = np.asarray(a)
    return a[0] if a.ndim >= 2 and a.shape[0] == 1 else a


GRIP_MAX_WIDTH = 0.08


def grip_norm_to_width(a: float) -> float:
    return max(0.0, 2 * ((float(a) + 1) / 2 * 0.05 - 0.01))


class GenesisWorld:
    """Manifest-driven scene: build once, replay an action stream, record
    per-entity state each control step."""

    def __init__(self, manifest: dict, pkg: Path | None = None,
                 render: bool = False, camera: dict | None = None):
        import genesis as gs
        if not getattr(gs, "_initialized", False):
            gs.init(backend=gs.gpu, logging_level="warning")
            gs._initialized = True
        self.gs = gs
        env = manifest["environment"]
        self.steps_per_ctrl = int(env.get("steps_per_control", 5))
        dt = float(env.get("dt", 1.0 / (env.get("control_freq", 20) * self.steps_per_ctrl)))
        # Optional per-solver options straight from the manifest, so a package
        # that needs a tighter/finer particle grid than the genesis default
        # (dx = 1/64 m leaks straight through an 8 mm container wall) carries
        # that with it and replays identically.
        solver_kw = {}
        for key, opt in (("mpm", "MPMOptions"), ("sph", "SPHOptions"),
                         ("pbd", "PBDOptions")):
            if env.get(key):
                solver_kw[f"{key}_options"] = getattr(gs.options, opt)(**env[key])
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=dt, substeps=int(env.get("substeps", 10))),
            show_viewer=False,
            renderer=gs.renderers.Rasterizer(),
            **solver_kw,
        )
        self.scene.add_entity(gs.morphs.Plane(
            pos=(0, 0, float((manifest["scene"].get("support") or {}).get("z", 0.0)))))

        rb = manifest["robot"]
        base = np.asarray(rb.get("base_pose", [0, 0, 0, 1, 0, 0, 0]), dtype=float)
        if rb.get("urdf"):
            morph = gs.morphs.URDF(file=str(pkg / rb["urdf"]) if pkg else rb["urdf"],
                                   pos=tuple(base[:3]), quat=tuple(base[3:]), fixed=True)
        else:
            morph = gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml",
                                   pos=tuple(base[:3]), quat=tuple(base[3:]))
        self.robot = self.scene.add_entity(morph)

        self.entities: dict[str, tuple[str, object]] = {}
        for e in manifest["scene"].get("entities") or []:
            self.entities[e["name"]] = (e["type"], self._add_entity(e, pkg))

        self.cam = None
        if render:
            self.cam = self._add_camera(camera, manifest)
        self.scene.build()

        # The MJCF panda drives its fingers through a tendon that genesis
        # approximates with a NON-PD-reducible joint actuator: a plain
        # control_dofs_position target is ignored and the fingers collapse
        # shut (measured: 0.0386 -> 0.0001 m over 60 steps while commanded
        # open).  Install explicit PD gains so the width channel works in
        # both directions.  Only ever exercised by episodes that OPEN the
        # gripper, which is why the closed-gripper smoke never caught it.
        self._install_gripper_pd(
            kp=float(env.get("gripper_kp", self.GRIPPER_KP)),
            kv=float(env.get("gripper_kv", self.GRIPPER_KV)),
            fmax=float(env.get("gripper_fmax", self.GRIPPER_FMAX)))

        iq = np.asarray(rb.get("init_qpos", []), dtype=float)
        if iq.size:
            self.robot.set_dofs_position(iq[: self.robot.n_dofs])
        for e in manifest["scene"].get("entities") or []:
            if e["type"] == "articulated" and e.get("init_q") is not None:
                self.entities[e["name"]][1].set_dofs_position(
                    np.asarray(e["init_q"], dtype=float))

    GRIPPER_KP = 2000.0
    GRIPPER_KV = 200.0
    GRIPPER_FMAX = 100.0

    def _install_gripper_pd(self, kp=None, kv=None, fmax=None) -> None:
        """Give the two finger DOFs real position gains (see __init__).

        Episodes that must hold a heavy/awkward payload can raise the gains
        from the manifest (``environment.gripper_kp`` etc.) — the grip force
        is kp * interference, and the stock 2000 N/m only reaches ~12 N on a
        6 mm interference, far under the panda's ~70 N spec."""
        n = self.robot.n_dofs
        if n < 9:
            return
        kp = self.GRIPPER_KP if kp is None else kp
        kv = self.GRIPPER_KV if kv is None else kv
        fmax = self.GRIPPER_FMAX if fmax is None else fmax
        fi = [n - 2, n - 1]
        try:
            self.robot.set_dofs_kp(np.full(2, kp), fi)
            self.robot.set_dofs_kv(np.full(2, kv), fi)
            self.robot.set_dofs_force_range(np.full(2, -fmax),
                                            np.full(2, fmax), fi)
        except Exception as e:                      # pragma: no cover
            print(f"[genesis] failed to set gripper PD gains ({e}); the width channel may not track")

    # ------------------------------------------------------------ builders
    def _morph(self, m: dict, pkg: Path | None, fixed: bool):
        gs = self.gs
        kind = m["kind"]
        common = dict(pos=tuple(m.get("pos", (0, 0, 0))),
                      euler=tuple(m.get("euler", (0, 0, 0))))
        if kind == "box":
            return gs.morphs.Box(size=tuple(m["size"]), fixed=fixed, **common)
        if kind == "cylinder":
            return gs.morphs.Cylinder(radius=float(m["radius"]),
                                      height=float(m["height"]), fixed=fixed, **common)
        if kind == "sphere":
            return gs.morphs.Sphere(radius=float(m["radius"]), fixed=fixed, **common)
        if kind == "mesh":
            f = str(pkg / m["file"]) if pkg and not str(m["file"]).startswith("meshes/") \
                else m["file"]
            return gs.morphs.Mesh(file=f, scale=float(m.get("scale", 1.0)),
                                  fixed=fixed, **common)
        raise ValueError(f"unknown morph.kind: {kind!r}")

    def _add_entity(self, e: dict, pkg: Path | None):
        gs = self.gs
        t = e["type"]
        if t == "rigid":
            morph = self._morph(e["morph"], pkg, e.get("fixed", False))
            mkw = {}
            if e.get("friction") is not None:
                mkw["friction"] = float(e["friction"])
            if e.get("density") is not None:           # kg/m^3, else genesis default
                mkw["rho"] = float(e["density"])
            mkw.update(e.get("material_params") or {})  # coup_softness, ...
            if mkw:
                return self.scene.add_entity(morph=morph,
                                             material=gs.materials.Rigid(**mkw))
            return self.scene.add_entity(morph)
        if t == "articulated":
            f = str(pkg / e["urdf"]) if pkg else e["urdf"]
            return self.scene.add_entity(gs.morphs.URDF(
                file=f, pos=tuple(e.get("pos", (0, 0, 0))),
                quat=tuple(e.get("quat", (1, 0, 0, 0))),
                fixed=e.get("fixed", False)))
        mp = dict(e.get("material_params") or {})       # per-episode calibration
        if t == "sph_liquid":
            return self.scene.add_entity(material=gs.materials.SPH.Liquid(**mp),
                                         morph=self._morph(e["morph"], pkg, False))
        if t == "mpm_elastoplastic":
            return self.scene.add_entity(material=gs.materials.MPM.ElastoPlastic(**mp),
                                         morph=self._morph(e["morph"], pkg, False))
        if t == "mpm_sand":          # Drucker-Prager: a real friction angle, so
            return self.scene.add_entity(   # a payload holds until the vessel
                material=gs.materials.MPM.Sand(**mp),   # tips past repose
                morph=self._morph(e["morph"], pkg, False))
        if t == "pbd_cloth":
            return self.scene.add_entity(material=gs.materials.PBD.Cloth(**mp),
                                         morph=self._morph(e["morph"], pkg, False))
        raise ValueError(f"unknown entity.type: {t!r}")

    def _add_camera(self, camera: dict | None, manifest: dict):
        if camera:
            from .transforms import quat_to_mat
            base = np.asarray(manifest["robot"].get("base_pose",
                                                    [0, 0, 0, 1, 0, 0, 0]), dtype=float)
            Tb = np.eye(4)
            Tb[:3, :3] = quat_to_mat(base[3:])
            Tb[:3, 3] = base[:3]
            T = Tb @ np.asarray(camera["extrinsics_base_cam"], dtype=float)
            pos = T[:3, 3]
            lookat = pos + T[:3, :3] @ np.array([0, 0, 1.0])   # OpenCV z-forward
            fy, h = camera["intrinsics"][1][1], camera["height"]
            fov = float(np.degrees(2 * np.arctan(h / (2 * fy))))
            return self.scene.add_camera(res=(int(camera["width"]), int(camera["height"])),
                                         pos=tuple(pos), lookat=tuple(lookat), fov=fov)
        return self.scene.add_camera(res=(640, 480), pos=(1.3, -1.1, 0.9),
                                     lookat=(0.4, 0.0, 0.2), fov=40)

    # ------------------------------------------------------------ stepping
    def step_action(self, action: np.ndarray):
        """One control step: 7 arm targets + normalized gripper -> n_dofs."""
        a = np.asarray(action, dtype=float)
        target = np.empty(self.robot.n_dofs)
        target[:7] = a[:7]
        if self.robot.n_dofs >= 9:
            w = grip_norm_to_width(a[7])
            target[7:9] = w / 2
        self.robot.control_dofs_position(target)
        for _ in range(self.steps_per_ctrl):
            self.scene.step()

    def entity_state(self, name: str) -> np.ndarray:
        t, ent = self.entities[name]
        if t == "rigid":
            p = _squeeze_env(_np(ent.get_links_pos()))
            q = _squeeze_env(_np(ent.get_links_quat()))
            return np.concatenate([p.reshape(-1), q.reshape(-1)])
        if t == "articulated":
            q = _squeeze_env(_np(ent.get_dofs_position()))
            p = _squeeze_env(_np(ent.get_links_pos()))
            return np.concatenate([q.reshape(-1), p.reshape(-1)])
        return _squeeze_env(_np(ent.get_particles_pos() if hasattr(ent, "get_particles_pos")
                                else ent.get_state().pos)).reshape(-1, 3)

    def render_frame(self):
        return _np(self.cam.render()[0]) if self.cam is not None else None


def run_genesis_actions(manifest: dict, actions: np.ndarray, pkg: Path | None = None,
                        render: bool = False, camera: dict | None = None,
                        render_stride: int = 2):
    """Replay the stream; returns ({entity: stacked states}, frames)."""
    world = GenesisWorld(manifest, pkg, render=render, camera=camera)
    names = list(world.entities)
    traj: dict[str, list] = {n: [] for n in names}
    frames = []
    for k, a in enumerate(np.asarray(actions, dtype=float)):
        world.step_action(a)
        for n in names:
            traj[n].append(world.entity_state(n))
        if render and k % render_stride == 0:
            frames.append(world.render_frame())
    return {n: np.stack(v) for n, v in traj.items()}, frames
