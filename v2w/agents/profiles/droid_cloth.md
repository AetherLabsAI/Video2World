# Reconstruct this towel-folding scene and execute your own fold (profile droid_cloth, droid-cloth-camera/1)

Input: only `{video}` ({W} x {H}, {T} video frames). It shows a DROID Franka arm with a Robotiq two-finger gripper
folding a towel on a table. Infer the initial cloth, table, and fingertip motion from the video. Read its frame rate
with ffprobe: action dt is in real seconds, and time zero is the first video frame. You have no robot logs,
calibration, depth, GT scene, or reference actions.

Write your package in `{out}`:
- protocol.json
- scene.json
- actions.npz
- report.md
- source/video.mp4 (a copy of the input; already staged)

Protocol example (replace names and estimates as appropriate):
```json
{{"protocol_version":"3.0", "physics_profile":"droid-cloth-camera/1",
  "coordinate_frame":"first_camera_opencv", "status":"success",
  "scene":{{"path":"scene.json"}},
  "actions":{{"path":"actions.npz", "format":"tcp_matrix_grip", "dt":0.13333333333333333}}}}
```
All positions, cloth corners, table point/normal and actions use the FIRST VIDEO CAMERA OpenCV frame (+x right,
+y down, +z forward), metres. Do not estimate or supply a robot-base transform: the evaluator alone applies the hidden
camera alignment.

scene.json contains exactly table and cloth:
```json
{{"table":{{"point":[0,0,0.8], "normal":[0,-1,0]}},
 "cloth":{{"corners":[[-0.15,0,0.65],[0.15,0,0.65],[0.15,0,0.95],[-0.15,0,0.95]],
 "grid":[15,15], "mass":0.04, "friction":1.0, "young":1000000,
 "poisson":0.3, "thickness":0.0003}}}}
```
Numbers above are illustrative, not calibrated scene estimates. Supply four convex perimeter corners of the initial
sheet, either winding, approximately parallel to the table. The table normal is unit length and points UP from the
surface toward the cloth; gravity follows minus this normal. The cloth is a bilinear elastic sheet; the table is a
static plane. Scene inputs are data only; no XML, custom code, equalities, attachment flags, or prescribed cloth
trajectories are accepted.

Resource bounds: edges 0.02..2 m; grid dimensions 3..41 with at most 1024 vertices; mass 0.001..5 kg; friction 0..5;
Young's modulus 1e3..1e7 Pa; Poisson ratio 0..0.49; thickness 1e-4..0.01 m; coordinates within 10 m of the camera.
Material choices affect the physical result. These bounds are not fitted values.

Use numpy.savez for actions.npz, containing exactly:
- tcp_pose: finite (T,4,4) rigid transforms of the fingertip in the camera frame. Rotation columns are the fingertip
  axes: +z wrist-to-tip, +y finger opening direction, +x completes a right-handed frame. Origin is the distal tip plane
  midway between the pads; pads extend 3.7 cm BACK along local -z.
- gripper_closure: (T,), exactly 0=open or 1=closed.
Each row lasts actions.dt seconds. There must be 2..7200 rows; dt is between 1/15 and 1 second and is an integer
multiple of 1/15; total duration at most 120 seconds. The evaluator resamples commands onto a 15 Hz control clock
without duration normalization, then steps MuJoCo at 0.5 ms. It holds the final commanded pose when comparing against
later video times. Do not speed up the video clock to improve trajectory scores.

The simulated end effector is a floating pair of Robotiq-sized pads; arm links are not simulated. Pads are 3.7 cm long,
2.2 cm wide, 0.8 cm thick, with an 8.5 cm open gap. The hand follows your fingertip poses. At an open-to-closed command,
the evaluator attaches cloth vertices within the pad capture volume (6 mm margin in pad width and tip depth, within the
open gap), at their current local positions. Opening releases them. Candidates cannot supply their own attachments.
Cloth motion thereafter comes from simulation. A 0.5 s settling period precedes the actions and a 1.5 s tail follows.
End your actions with an explicit release.

Task success: terminal observed-to-simulated cloth surface median distance <= 3 cm, footprint CHANGE ratio
(sim terminal/initial area)/(observed terminal/initial area) within 0.75..1.25, and the initial state must not already
satisfy the fold criterion. Footprints are measured in the hidden table plane. Shape, size, centre and trajectory
errors are scored separately from task success; only visible cloth surfaces are compared.

## Acceptance
Validate before finishing (format only, not task success):
`PYTHONPATH={repo} {py_eval} -m v2w.tracks.cloth.contract {out}` must exit 0.
`status: infeasible` (with the diagnosis in report.md) is a valid delivery.
Workspace rule: write only inside `{out}` and `/tmp/{name}/`. The only file about this episode you may read is `{video}`.
Do NOT search for or download this dataset's ground truth or calibration; everything metric must come from the pixels.
