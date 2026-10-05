# Reconstruct this hand-object video and perform its task with two Wuji hands

Input: `{video}` ({W} × {H}, {T} frames). Deliver a scene package at `{out}` containing the manipulated object, visible support and other relevant objects, and synchronized actions for a left and a right Wuji hand. Both hands are available even if only one human hand is visible. An unused hand must receive a stationary command stream.

Express all geometry, frame-zero poses, predicted object trajectories and palm poses in the frame-zero camera coordinate system. Estimate camera intrinsics. For a moving camera, recover camera motion from static scene features. Use protocol_version 3.0, physics_profile bridge_widowx, and identity cameras[0].extrinsics_base_cam for camera-frame delivery. Include source/video.mp4, protocol.json, meshes, expected object poses, and report.md describing assumptions. Sizes/positions are meters; object and camera pose quaternions are wxyz. Static entities go in scene.props; manipulated parts go in scene.objects.

A drawer needs a static frame, a moving drawer and a prismatic joint; a hinged door needs a revolute joint. Declare scene.joints with name, parent, child, type, axis, point and limits. Axis and point are in the parent local frame; q=0 is the frame-zero pose. Prismatic limits are meters; revolute limits are radians. Preserve cavities and physical clearance in collision geometry. Do not attach objects to robot links or prescribe their motion during simulation.

The robot is wuji_bimanual_floating: two independently controlled hands, each with six wrist DOFs and twenty finger joints. Both are dynamic, force-limited robots, with finite tracking error. They collide with the scene, themselves (except immediate mechanical neighbors) and the other hand. Each wrist's translation drive is capped at 200 N per axis and rotation at 100 N·m; actual finger effort/velocity limits come from its installed Wuji URDF. The evaluator checks actual velocity, position limits, contact penetration and hand intersection throughout the execution.

Palm coordinates: +z points along extended fingers, +x is the palm-facing/curling direction, and the thumb is on +y for the right hand and -y for the left hand. Extended tips are about 19 cm from the wrist. Finger order is thumb, index, middle, ring, pinky; four joints per finger. Finger 2–5 joint 2 controls abduction. Deliver radians within the corresponding URDF limits.

Deliver four finite arrays with the same number of rows:

- actions.npy and right_actions.npy: (N,7), [x,y,z,roll,pitch,yaw,grip], with R=Rz(yaw)Ry(pitch)Rx(roll). These are LEFT and RIGHT palm poses. Grip is informational: 1 means open, 0 closed; the finger joint commands determine posture.
- finger_joints.npy and right_finger_joints.npy: (N,20), finger1_joint1..4 through finger5_joint1..4.

Use one common dt, a positive multiple of 1/300 second. Example action declaration (left chosen as the top-level primary alias; either side is allowed):

```json
{{"path":"actions.npy", "finger_joints_path":"finger_joints.npy", "format":"tcp_abs_rpy_grip", "dt":0.03333333333333333,
  "bimanual_version":"wuji-bimanual-actions/1.0", "primary_hand":"left",
  "hands":{{"left":{{"path":"actions.npy","finger_joints_path":"finger_joints.npy"}},
             "right":{{"path":"right_actions.npy","finger_joints_path":"right_finger_joints.npy"}}}}}}
```

Each hand may additionally declare initial_finger_joints_path (20-vector) and prelude_offset_in_palm (3-vector, norm ≤25 cm). Default startup uses the canonical open posture and an 8 cm backward palm offset, with a fixed 1.2 s approach. Both hands share one clock; no independent adaptive waits are inserted. Any remaining force-bearing hand/object contact prevents a released-object verdict. Report proposed task completion and all modeling uncertainties; predicted object trajectories are declarations, not simulation inputs.
