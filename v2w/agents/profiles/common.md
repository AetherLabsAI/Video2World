
## protocol.json (schema is the `bridge_widowx` package; `v2s validate {out}` must exit 0)

```jsonc
{{
  "protocol_version": "3.0", "physics_profile": "bridge_widowx", "name": "{name}", "status": "success",
  "task": {{"instruction": "YOUR reading of what happens in the video", "source_video": "source/video.mp4"}},
  "robot": {{"uid": "widowx250s_bridge", "base_pose": [0,0,0, 1,0,0,0]}},      // nominal: the robot is NOT part of this profile
  "cameras": [{{"name": "main", "width": {W}, "height": {H},
               "intrinsics": [[fx,0,cx],[0,fy,cy],[0,0,1]],                    // YOUR OWN estimate at {W}x{H}; no calibration is provided or visible
               "extrinsics_base_cam": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],  // MUST stay identity: base frame == camera frame
               "source": "estimated_from_video"}}],
  "scene": {{"support": {{"z": <plane, see below>, "center": [x, y], "size": [1.2, 1.2], "color": [0.6,0.6,0.6,1]}} | null,
            "arena": null, "props": [ /* static things */ ], "objects": [ /* the manipulated object(s), at least one */ ]}},
  "provenance": {{"method": "...", "agent_model": "...", "notes": "..."}}
}}
```
entity: {{"name": "...", "kind": "box"|"cylinder"|"sphere"|"container"|"mesh", "half_size": [hx,hy,hz], "mesh_path": "assets/x.obj",
          "collision_path": "assets/x_collision.obj", "scale": 1.0, "pos": [x,y,z], "quat": [w,x,y,z], "density": 300, "color": [r,g,b,a], "symmetry": null|"axis"|"box"}}
All poses are in the CAMERA frame (OpenCV: +x right, +y down, +z forward, metres) — this profile does not use a robot base frame.
The support plane is therefore NOT z=0: give `support.z` as the plane's offset along the camera z axis at the image centre ray is not
meaningful, so instead give the support as `null` and put the plane into `props` as a thin box if you need it.

## Trajectory (REQUIRED for this profile)
`{out}/expected/obj_poses.npy` — float32 array (T, 7) = [x y z qw qx qy qz] of the manipulated object's pose in the CAMERA frame
for every video frame (T = number of frames in `{video}`), estimated from the pixels (tracking, geometry, your judgement).
Also write `{out}/expected/README.md` with how you obtained it.

## Actions (REQUIRED — the video2sim contract: the TASK, not the path)
`{out}/actions.npy` — float64 array (T', 7), each row `[x y z roll pitch yaw grip]`: YOUR OWN action stream that performs
the task. Absolute gripper TCP poses in the CAMERA frame (same OpenCV frame as everything else), 5 Hz;
R_tcp = Rz(yaw)·Ry(pitch)·Rx(roll); `grip` is a COMMAND, 1 = open, 0 = close (not a width). Row 0 is the initial TCP state
(the evaluator IK-drives the arm there before stepping). Declare it in protocol.json:
`"actions": {{"path": "actions.npy", "dt": 0.2, "format": "ee_state_abs"}}`.

The evaluator EXECUTES your actions in YOUR scene on the real rig (numerical IK -> joint PD) and judges the terminal state
with its own success predicate for this episode — the task outcome must match the video (the object ends where it ends,
on/in what it ends on/in); how you get there is yours to plan. You do NOT have to copy the video's motion frame by frame,
but the video's own trajectory is the best evidence for a stream that works. Practical rig facts: keep per-step moves
<= 3 cm / <= 0.3 rad and end at rest; a grasp only holds if, when `grip` transitions 1 -> 0, the TCP is actually AT the
object (the fingers physically close; commanding close 5 cm away grabs nothing, so put the grasp rows exactly on your
object's surface); the arm holds still while the fingers travel (~1.5 s), you need not pad extra rows for the closing.
As a secondary diagnostic the evaluator also replays the withheld REAL trajectory in your scene, so put things where
they really are — a scene that only works for your own path is a weak scene.

## Acceptance
`v2s validate {out}` must exit 0 (run it with: `PYTHONPATH={ours} {py_eval} -m video2sim.cli validate {out}` — `status: success` REQUIRES the action stream above).
Before claiming success, self-check your own numbers: row 0 and every grasp row of `actions.npy` should coincide with your
declared object pose to within ~1 cm; the release rows should sit at the task's goal; no step should jump more than 3 cm.
Put every measurement on annotated frames in `{out}/source/frames/`. Declare in report.md what you could not observe.
`status: infeasible` (with the diagnosis in report.md) is a valid delivery.

Workspace rule: write only inside `{out}` and `/tmp/{name}/` (both on local disk). The only file about this episode you may read is `{video}`.
Do NOT search for or download this dataset's ground truth, meshes or calibration; everything metric must come from the pixels.
