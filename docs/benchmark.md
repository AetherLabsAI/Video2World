# Benchmark protocol

## What an agent receives and delivers

| Item | Content | Read by |
|---|---|---|
| Video | `data/samples/<sample>/video.mp4` | the agent |
| Brief | `v2w/agents/profiles/<profile>.md`, plus the public task contract and robot kit of the track | the agent |
| Package | `protocol.json` (scene, robot, camera frame, task), `actions.npy`, meshes, `report.md` | the evaluators |
| Ground truth | `gt_pkg/`, `hidden/`, `meta.json` in the sample directory | the evaluators only |

The package format is the video2sim protocol ([video2sim/PROTOCOL.md](../video2sim/PROTOCOL.md)).
Unless a brief says otherwise, poses are delivered in the video camera's OpenCV frame; the evaluator maps them to the
evaluation frame with the hidden camera extrinsics. The bundled driver (`v2w.agents.run`) stages only the video and the
brief into an isolated working directory and audits the agent log for accesses to the sample directory; a custom
`--agent-cmd` must enforce the same isolation.

## Evaluation

Every configuration runs three resumable steps under `runs/<experiment>/<family>/<sample>/`:

1. **agent** produces `pkg/`;
2. **eval** validates the package, builds the scene in the track's simulator and executes the package's own actions
   (ManiSkill/SAPIEN for FurnitureBench, DROID and the hand tracks; Isaac Sim for RoboDojo and the in-house episodes;
   MuJoCo for the reconstructed twins and cloth);
3. **metrics** compares the built scene and the executed trajectories with the GT and writes `metrics.json`.

A missing, invalid or unbuildable package is a method failure (`build = false`). An evaluator crash is recorded with
`error_kind = "evaluator"`; it blocks the score rather than counting against the method.

## Metrics

Which metrics apply to a configuration is declared by its GT metric contract in `data/manifest.json`, never by the
submission.

**Geometry (G)**: scene Chamfer distance (visible GT surfaces against the built scene after an object-only alignment,
entities weighted by their involvement in the demonstrated motion), object shape Chamfer distance in the object frame,
and object size error; deformable configurations use visible scene points, initial surface and world AABB size instead.

**Dynamics (D)**, always on the package's own execution: absolute translation and symmetry-aware rotation error of the
manipulated object against the demonstration, relative translation error over a fixed stride; observable-axis
rotation errors where only part of the rotation is observable; point position / displacement errors for point-robot
episodes; surface-trajectory Chamfer distance for cloth.

**Function (F)**: `task_success`, the task's success predicate on the final simulator state, and
`progress.progress`, the fraction of the task's ordered stages achieved.

## V2WScore

```
S = build · (F + G + D) / 3            per configuration
F = α · success + (1 − α) · progress     α = 0.5
quality(error) = max(0, 1 − error / τ)
G, D = mean quality over the applicable configured metrics
```

Configurations are averaged within a family and families get equal weight; the score is reported ×100. Tolerances
(`v2w/config/score.json`, `v2w-score/1`):

| Metric | τ |
|---|---:|
| scene CD, object shape CD, object size error (and their deformable counterparts) | 10 cm |
| translation APE / RPE | 20 cm / 10 cm |
| rotation APE, observable-axis errors | 90° |
| surface-trajectory CD | 20 cm |
| point position / displacement | 20 cm / 10 cm |

The scope is fixed from the GT metric contracts before any result is read. A build failure stays in the denominator
with S = 0; a missing record, an evaluator error or a missing applicable measurement makes the report `blocked` instead
of shrinking the denominator. `G_object` and `G_scene` regroup the geometry metrics for reporting only.

## Human-assisted reference (HAR)

Each configuration has a reference package built with access to the GT and recorded human assistance, executed and
scored with the same code as the agents. Its metric records ship in `data/har/`; `v2w har` scores them.
It is a reference row, not an upper bound.
