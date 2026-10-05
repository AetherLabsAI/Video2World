> Vendored from video2sim. Video2World ships the package validator
> (`video2sim/protocol/validate.py`) and the simulation bridges its evaluators use; the `v2s` CLI and `replay.py`
> referenced below belong to the full video2sim distribution.

# Protocol package specification (v3.0)

A **protocol package** is a self-contained directory: given nothing but that
directory, a machine can reproduce the manipulation task in simulation.
`video2sim/protocol/replay.py` is the operational definition of this document —
a package is *valid* if `v2s validate` passes, and *reproducible* if
`v2s replay` passes.

## Layout

```
<package>/
  protocol.json            manifest (the schema below)
  actions.npy              (T, 8) float64 action stream
  report.md                the agent's observation report (human-readable)
  expected/
    obj_poses.npy          (T, 7) recorded object trajectory [x y z qw qx qy qz]
    qpos.npy               (T, 9) recorded joint positions (reference)
  assets/
    robot/                 robot URDF + every mesh it references (self-contained)
    objects/               dynamic object visual/collision meshes (if any)
    props/                 static prop meshes (if any)
  source/
    video.mp4              copy of the input video
    frames/                annotated evidence frames
  verification/
    replay_report.json     written by `v2s replay`
    render.mp4             replay rendered through the protocol camera
```

## Conventions

- **World frame.** In `support` mode the robot base sits at the world origin
  (reproducing how the real arm is mounted), so **world == base frame**. In
  table mode (`support: null`) the base is at `(-0.615, 0, 0)`.
- **Poses** are 7 numbers, `[x y z qw qx qy qz]` (wxyz quaternion), everywhere.
- **Camera extrinsics** are `T_base_cam`, a 4x4 matrix in the **OpenCV camera
  frame** (+Z forward, +X right, +Y down), such that `X_base = T_base_cam @
  X_cam`. Relative to the **base**, not the world (identical in support mode).
- **Actions** are `pd_joint_pos`: 7 arm joint targets in radians (not
  normalized) plus one gripper command normalized to [-1, 1], which maps to
  [-0.01, 0.04] m per finger (opening = 2 x per-finger target). The stream
  starts at the first control step after `env.reset` and **includes** any
  settle and pre-roll phase: replay is `reset` followed by stepping every row
  in order, with no other information.

## protocol.json

```jsonc
{
  "protocol_version": "1.0",
  "name": "wedge_back",                    // short slug
  "created": "2026-08-03",
  "status": "success",                     // success | failure | infeasible
  // infeasible = the video is unsuitable (grasp off-frame, target occluded at
  //   start or end, loop action, ...). actions/expected/assets may then be
  //   omitted, but report.md must carry the diagnosis.
  // failure = the pipeline ran end to end but the task semantics were not
  //   achieved. The package must still be complete — an honest failure is
  //   reproducible too.

  "task": {
    "instruction": "move the orange wedge ~12 cm backwards",  // YOUR reading of
                                             // the video, not the dataset label
    "source_video": "source/video.mp4",
    "source_meta": {"lab": "RAIL", "fps": 15, "resolution": [1280, 720]},
    "phases": [                            // semantic phases on the action clock
      {"name": "approach", "t": [0.0, 2.5], "video_frames": [0, 40]},
      {"name": "grasp",    "t": [2.5, 3.0], "video_frames": [40, 55]}
    ]
  },

  "environment": {
    "simulator": "mani_skill-3.0.1/sapien-3.0.3",
    "env_id": "V2SPick-v1",
    "sim_freq": 100,
    "control_freq": 20,
    "control_mode": "pd_joint_pos"
  },

  "robot": {
    "uid": "panda",
    "urdf": "assets/robot/panda_v2.urdf",  // package-relative
    "base_pose": [0,0,0, 1,0,0,0],         // world frame; must match scene mode
    "init_qpos": [/* 9 */]                 // configuration after reset, before
                                           // the first action
  },

  "cameras": [
    {
      "name": "main",
      "width": 320, "height": 240,
      "intrinsics": [[fx,0,cx],[0,fy,cy],[0,0,1]],
      "extrinsics_base_cam": [/* 4x4 row-major, T_base_cam, OpenCV frame */],
      "source": "estimated_from_video | dataset_calibration | sim_default"
    }
  ],

  "scene": {
    "support": {"z": 0.19, "center": [0.45, 0.0], "size": [1.4, 1.4]},  // or null
    "objects": [                           // dynamic; exactly 1 in this env
      {
        "name": "wedge", "kind": "mesh",   // box | lshape | cylinder | mesh | ycb
        "half_size": [0.02,0.02,0.02],
        "mesh_path": "assets/objects/wedge.obj",
        "collision_path": null,
        "mesh_scale": 1.0, "density": 500.0,
        "color": [0.9,0.45,0.1,1.0],
        "init_pos": [0.45,0.0,0.19], "init_quat": [1,0,0,0]
      }
    ],
    "props": [                             // static scene props
      // static_box | cylinder | container | cylinder_container | mesh (no kind)
      {"kind": "container", "name": "bin", "inner_size": [0.09,0.09],
       "height": 0.12, "wall": 0.004, "pos": [0.5,0.1,0.19]},
      // solid round pillar, axis local +z, pos = the body CENTRE. Pair it
      // with a cylinder_container (which has no floor of its own) when a
      // vessel's cavity is shallower than the thing it stands on.
      {"kind": "cylinder", "name": "flask_body", "radius": 0.0525,
       "half_length": 0.101, "pos": [0.5,0.0,0.101]}
    ],
    "articulations": [                     // passive, contact-driven only
      {"kind": "drawer",                   // drawer | cabinet_door | kettle_lid
       "name": "drawer", "inner_size": [0.22,0.3,0.1], "wall": 0.008,
       "travel": 0.15, "init_q": 0.12,     // doors use open_limit_deg / hinge /
       "friction": 0.6, "damping": 3.0,    //   init_q_deg
       "pos": [0.35,-0.25,0.0]}
      // kettle_lid (dof=3): a press-to-pop lid — shell + lid (horizontal hinge
      //   with a torsion spring) + centre button (return spring) + latch hook
      //   (driven by a ball-and-ramp cam). The latch holds structurally (load
      //   perpendicular to the hook axis); the springs are joint drives, i.e.
      //   part of the modelled object, not actuation.
    ]                                      // at most 1; the robot may only move
  },                                       // it through contact

  "actions": {"path": "actions.npy", "dt": 0.05, "T": 400},

  "initial_state": {
    "obj_pose": [/* 7, passed to reset options, before settling */],
    "qpos": [/* 9, same as robot.init_qpos */]
  },

  "expected": {
    "obj_poses": "expected/obj_poses.npy",
    "qpos": "expected/qpos.npy",
    "artic_qpos": "expected/artic_qpos.npy",  // (T, dof); required with articulations
    "final_artic_qpos": [/* dof */],
    "tolerance_artic": 0.02,                  // m for prismatic, rad for revolute
    "final_obj_pose": [/* 7 */],
    "tolerance_pos": 0.02,                 // replay position tolerance (m)
    "tolerance_rot_deg": 20.0,             // replay rotation tolerance (deg)
    "grasp_transitions": 2,                // is_grasped transitions while recording
    "success_criteria": {                  // task semantics, independent of fidelity
      "type": "final_pose",                // final_pose | goal_region | articulation
      "target_pos": [/* 3 */], "pos_radius": 0.05
      // articulation: {"type": "articulation", "target_qpos": [..], "tolerance": 0.02}
    }
  },

  "provenance": {
    "method": "claude-code-agent",
    "agent_model": "...",
    "evidence": ["source/frames/grasp_overlay.png", "verification/track_overlay.mp4"],
    "checks": {
      "visual_inspection": true,           // pixel-derived conclusions were eyeballed
      "grasp_transitions_verified": true,  // transition count + release-height check
      "geometry_cross_checked": true       // sizes / surface height measured two ways
    },
    "notes": "free text: how things were measured, traps hit, what is unresolved"
  }
}
```

**Set the success criterion from what the task means, not from what you
recorded.** Copying the recorded final pose into `target_pos` makes the
criterion self-fulfilling — it passes even if nothing happened.

## augment/ — the data-expansion interface (expected with `status: success`)

A delivered episode should also generalise: `augment/layout_space.json`
declares WHAT may be perturbed (per-entity xy/yaw ranges in the base frame,
clearance and workspace constraints, and the success criteria in RELATIONAL
form — targets reference entities, e.g. `{"target": {"prop": "pot",
"offset": [0,0,0.05]}}` — so the goal follows the perturbed layout), and
`augment/policy.py` is code-as-policy:

    def plan(layout: dict, ctx) -> (PoseTraj, GripperTraj)

`layout` maps entity name → settled pose7 for ONE sampled layout; the return
is an EE keyframe trajectory + gripper widths that achieves the same task
semantics there. `ctx` (video2sim.augment.AugmentContext) carries the
manifest, the recorded reference episode and grasp helpers
(`top_grasp_quat(yaw)` closes ACROSS the axis given by yaw; `entity_yaw`;
`keyframes(rows)`).

Alongside the policy, `augment/reward.py` makes the episode trainable:

    def reward(prev: dict, curr: dict, ctx) -> float     # dense, per step
    def success(curr: dict, ctx) -> bool                 # optional; defaults to
                                                         # the relational criteria

Both see only the STATE DICT (t, objects/props poses, ee, gripper_width,
grasped, layout) — never the simulator — so the same function scores online
RL steps (`video2sim.rl.PackageEnv`: reset samples the layout space, step
returns this reward) and recorded streams offline. Derive every target from
`ctx`/state (`ctx.goal_pos()` follows the perturbed layout); absolute
coordinates are the classic wrong reward. Verify without training:

    v2s reward-check <package>

scores the reward over the package's own augment episodes plus a fresh
do-nothing rollout, and accepts it only if it is finite and bounded,
successful rollouts out-return the no-op by a clear margin, and each
success's terminal step is the peak of its cumulative return (a reward that
peaks mid-episode and decays is pointing somewhere other than success —
this check caught exactly that in the reference reward's first draft).

    v2s augment <package> --n 20 [--seed S] [--render-first K]

samples layouts, runs the policy on each in the package's own scene, judges
the relational criteria + terminal quiescence, writes
`augment/episodes/<k>/` (actions/qpos/obj_poses/meta, provenance) and
`augment/report.json` with the success band. Declare the ranges from what the
video actually pins down (table extent, reachable workspace) and report the
measured success rate honestly — a policy that only works on the recorded
layout scores exactly what it is. Reference example:
`outputs/agent_rail_marker_pot/augment/` (20/20 at ±20 cm translation, free
yaw).

The interface is backend-generic. `rigid_sapien` and `bridge_widowx` packages
run natively (bridge policies plan EE-link poses driven through the WidowX rig
at 5 Hz; multi-object episodes save `obj_poses.npz`; reference:
`outputs/bench_claude_v2/folding_table_035170/augment/`, 6/6 stacking).
`robodojo` is a delivery target, not a physics profile — augment in
the source profile, then

    v2s augment <package> --deliver robodojo --deliver-n K

wraps successful variants as packages and retargets them through the rig,
inheriting the parent delivery's staging (heading/offset/arm) so variants land
in-distribution like the parent did.

`video2sim.rl.PackageEnv` turns a package with augment + reward files into a
gym environment on both native profiles: actions are the profile's own
convention ((8,) pd_joint_pos on rigid_sapien; the Bridge dataset's (7,)
delta-EE + gripper command on bridge_widowx), each reset samples a fresh
layout from the declared space, and `terminated` is the package's relational
success criteria.

## Acceptance

1. `v2s validate <package>` — schema, file existence, array shapes and internal
   consistency (`actions.T` matches the trajectories, `init_qpos` agrees with
   `initial_state`, `base_pose` matches the scene mode, ...).
2. `v2s replay <package>` — rebuild the scene from the package alone, step the
   actions, render `verification/render.mp4` through the protocol camera and
   write `replay_report.json`:
   - `replay_match`: the final object pose is within tolerance of
     `expected.final_obj_pose` (reproduction fidelity);
   - `task_success`: the final state satisfies `success_criteria` (semantics);
   - deviation statistics against `expected/obj_poses.npy` over the whole run.

   `status: "success"` requires both. `"failure"` requires only `replay_match`.
   With articulations, `replay_match` additionally requires the final joint
   positions to be within `tolerance_artic`.

**The terminal state must be at rest.** If anything is still moving in the last
frames, the replay has a branch point and a matching number is a sampling
artifact, not determinism. Assert quiescence before packaging rather than
widening a tolerance.

## physics_profile: `robodojo`

Normally a RoboDojo delivery rides on top of a `rigid_sapien` package: that
package is the physics, and `robodojo/` is an extra deliverable retargeted from
it. `"physics_profile": "robodojo"` is for the episodes where no such base can
exist. The SAPIEN env holds one Panda and **exactly one** dynamic body, so a
DUAL-ARM episode, or one with several manipulated objects, would have to be
misdescribed in order to have a base package at all. Declare the rig as the
physics instead: the actions are the benchmark's own (T, 14) joint stream at
25 Hz, the scene is expressed in the rig's world frame, and gate 2 re-runs the
rig rather than SAPIEN.

```jsonc
{
  "physics_profile": "robodojo",
  "actions": {"path": "robodojo/actions.npy", "dt": 0.04, "T": 827},
  "scene": {
    "objects": [                  // any number; RIG world frame, z up from 0
      {"name": "six", "kind": "mesh",         // box | cylinder | mesh
       "mesh_path": "assets/objects/digit_6.obj",
       "collision_path": "assets/objects/digit_6_collision.obj",
       "mesh_scale": 1.0, "mass": 0.02,
       "init_pos": [-0.404, -0.085, 0.765], "init_quat": [1, 0, 0, 0],
       "color": [0.13, 0.70, 0.82],
       "static_friction": 1.0, "dynamic_friction": 1.5}
    ],
    "props": [                    // static scenery, same shape vocabulary
      {"name": "pad1", "kind": "mesh", "mesh_path": "assets/objects/pad.obj",
       "pos": [-0.129, -0.084, 0.765], "quat": [1, 0, 0, 0]}
    ]
  },
  "expected": {
    "object_targets": [           // ONE PER OBJECT — a multi-object task has no
      {"object": "six",           //   single final_obj_pose
       "target": {"prop": "pad1", "offset": [0, 0, 0.0033]},
       "target_pos": [-0.129, -0.084, 0.768], "pos_radius": 0.02}
    ],
    "tolerance_pos": 0.02,
    "grasp_transitions": 12
  },
  "cameras": [{"name": "cam_head", "width": 640, "height": 480,
               "intrinsics": [[288.13, 0, 320], [0, 288.13, 240], [0, 0, 1]],
               "extrinsics_world_cam": [[...4x4...]]}]   // WORLD, not base
}
```

`expected.event_sequence` is the criterion for an episode whose terminal state
is not the point — "cover three blocks then uncover them" ends where it began,
and a pour's defining state is transient. It is
`{"type": "ordered_events", "events": [...], "terminal": [...],
"stationary": [...]}`, each event a predicate on one recorded object relative
to another, matched on its RISING EDGE and in order:

```jsonc
{"name": "pour", "object_key": "cup", "reference_key": "vase",
 "relation": "over",              // or "clear_of"
 "xy_radius": 0.035,              // within (over) / beyond (clear_of)
 "min_z_above_ref_m": 0.10,       // a vertical BAND: any of max_z_m,
                                  // max_z_above_ref_m, min_z_m,
                                  // min_z_above_ref_m; at least one required
 "min_tilt_deg": 90.0,            // the object's own +z against world +z
 "min_hold_frames": 5}
```

The vertical bound is required because an object being CARRIED over a target
would otherwise count as having been put there; the `min_*` forms exist
because a pour is the mirror of that — raised above the rim, not down on the
table — and `min_tilt_deg` because "passed over the vase's mouth" is true of a
pick-and-place that never turned the cup over. The delivery must carry the
same events into `robodojo/delivery_report.json` as `rig_events`, since gate 2
re-runs the rig and can only check what the delivery recorded.

`robodojo/` carries the delivery exactly as for the add-on form:
`actions.npy` (T, 14), `rollout.npz`, a composed `render.mp4` you have watched,
and `delivery_report.json` with `in_distribution: true`. For gate 2 to ask
whether the RE-RUN still did the task, the report additionally carries
`rig_targets`: `[{object_key, pos, radius}]`, one per recorded object
(`obj_object_<i>_traj`, indexed by position in the spawn list). Success is the
conjunction — three objects home and one beside is a different outcome, not
75 % of this one.

Authoring helpers: `robodojo.assemble_dual_action_stream` builds the 14-dim
stream from two per-arm halves; `robodojo_retarget.ArmIK` solves world-frame
top-down poses with the pad FACE as the tool point and rejects self-colliding
postures (`min_clearance`, which the rig's `enabled_self_collisions=True`
makes necessary); `robodojo_deliver.rig_objects_from_scene` turns the manifest
scene into spawn specs — the same translation the augmentation uses, so a
parent and its variants are staged by one code path.

Augmentation works on this profile natively, with one difference the profile
forces: `plan(layout, ctx)` returns the **(T, 14) action stream** rather than
an EE keyframe trajectory, because a single-arm keyframe list cannot express a
two-armed episode. Everything else — the layout space, the relational
criteria, `v2s reward-check` — is unchanged. A variant is a full Isaac rollout,
so budget minutes rather than seconds per sample.

## physics_profile: `soft_warp`

Top-level field `physics_profile`, default `"rigid_sapien"` (everything above).
`"soft_warp"` runs a separate backend: spring-mass cloth on NVIDIA Warp
(`video2sim/soft.py`), with the robot as a kinematic FK proxy (the
`pd_joint_pos` stream is assumed perfectly tracked) and grasping modelled as
**pinch attachment** — particles inside the closing jaws are rigidly attached
to the TCP frame and released when it opens.

```jsonc
{
  "physics_profile": "soft_warp",
  "environment": {"simulator": "warp-1.16.0", "control_mode": "pd_joint_pos_kinematic"},
  "cameras": [],                          // optional here (headless render)
  "scene": {
    "support": {"z": 0.0},                // surface height (world == base)
    "objects": [],
    "props": [                            // static colliders (wp.Mesh BVH)
      {"kind": "static_box", "half_size": [..], "pos": [..], "quat"?: [..]},
      {"kind": "cylinder", "radius": .., "half_length": .., "pos": [..]},  // axis = local +z
      {"kind": "mesh", "collision_path": "assets/props/bowl.obj", "scale": 1.0, "pos": [..]}
    ],
    "soft_bodies": [{                     // exactly 1
      "kind": "cloth_grid", "name": "cloth",
      "nx": 16, "ny": 16, "spacing": 0.0125,
      "origin": [0.35, -0.10, 0.005],     // position of the (0,0) corner particle
      "yaw_deg": 0.0, "mass_total": 0.05,
      "iters": 20, "substeps": 8,
      "attach_width": 0.02, "release_width": 0.035,
      "self_collision": true, "self_dist": 0.6,
      "self_friction": 0.7,       // cloth-on-cloth tangential damping (0 = off)
      "table_stick_speed": 0.05   // m/s; below it table contact sticks (0 = off)
    }]
  },
  "expected": {
    "particles": "expected/particles.npy",   // (T, N, 3) float32
    "tolerance_particles": 0.005,            // max final particle deviation (m)
    "final_centroid": [/* 3 */],
    "success_criteria": {"type": "soft_centroid",       // centroid lands in a ball
                         "target_pos": [/* 3 */], "pos_radius": 0.03}
    // or soft_clear (uncover-style): the centroid must END AWAY from a region
    //   {"type": "soft_clear", "clear_pos": [/* >=2 */], "clear_radius": 0.08}
  }
}
```

Replay re-simulates the whole stream and compares the maximum final particle
deviation against `tolerance_particles`. The solver is double-buffered Jacobi
PBD, so replay is deterministic on the same GPU and wheel. Rendering uses
pyrender/EGL (full robot visual meshes posed by FK, cloth, props, and the
protocol camera when present), falling back to a matplotlib sketch.

## physics_profile: `genesis`

A third backend (`video2sim/genesis_backend.py`) covering what the other two
cannot: any number of rigid bodies, **articulated objects with free roots**
(bottle caps, scissors), SPH fluids, MPM elasto-plastics and PBD cloth with
native robot-link pinning. The robot is the bundled MJCF Franka; the action
stream format is unchanged. Rendering is the Genesis rasterizer — no ray
tracing, no Vulkan.

```jsonc
{
  "protocol_version": "3.0",
  "physics_profile": "genesis",
  "environment": {"simulator": "genesis-1.3.1", "control_freq": 20,
                  "steps_per_control": 5, "substeps": 10},
  "robot": {"urdf": null,                  // null = built-in MJCF panda
            "base_pose": [/* 7 */], "init_qpos": [/* >=7 */]},
  "scene": {
    "support": {"z": 0.0},
    "entities": [                          // any number, any mix
      {"name": "cube", "type": "rigid",
       "morph": {"kind": "box", "size": [..], "pos": [..], "euler"?: [..]}},
      {"name": "bottle", "type": "articulated", "urdf": "assets/objects/bottle.urdf",
       "pos": [..], "fixed": false, "init_q": [..]},     // joints ON the object
      {"name": "water", "type": "sph_liquid", "morph": {...}},
      {"name": "dough", "type": "mpm_elastoplastic", "morph": {...}},
      {"name": "cloth", "type": "pbd_cloth", "morph": {"kind": "mesh", ...}}
    ]                                      // morph.kind: box|cylinder|sphere|mesh
  },
  "expected": {
    "states": "expected/genesis_states.npz",   // one (T, ...) trajectory per entity
    "entity_types": {"cube": "rigid", ...},    // selects the acceptance tier
    "tolerance_rigid": 1e-5,       // rigid/articulated: bit-exact in practice
    "tolerance_particles": 0.002,  // particle solvers drift ~5e-5 m on GPU atomics
    "success_criteria": {          // entity_final_pos | entity_joint | particles_centroid
      "type": "entity_final_pos", "entity": "cube",
      "target_pos": [/* 3 */], "pos_radius": 0.05}
  }
}
```

State encoding: `rigid` = link positions and quaternions flattened;
`articulated` = dofs followed by link positions; particle entities = `(N, 3)`.

## Known limits

- **soft_warp**: a single rectangular cloth sheet; pinch attachment is an
  approximation of finger friction; self-collision is particle repulsion, not
  exact cloth contact; stiffness and mass need per-episode calibration
  (inverting them from video, PhysTwin-style, is the obvious next step).
- **rigid_sapien articulations**: one drawer (prismatic), cabinet door
  (revolute) or kettle lid (3-dof latch mechanism) per scene, always passive.
  Multi-joint cabinets, knobs and double doors are not supported.
- **genesis**: cross-machine determinism is unverified; dynamic granular media
  in the contact chain amplify tiny differences (measured: 1.7 m divergence
  between two runs of the same stream) — freeze such media to their settled
  configuration when the episode does not depend on their motion.
- **Mixing**: the soft profile accepts static props but not dynamic rigid
  bodies or articulations alongside the cloth. The genesis profile mixes
  freely.
