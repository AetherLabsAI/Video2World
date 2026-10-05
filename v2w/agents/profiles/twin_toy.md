# Reconstruct this video as a native MuJoCo scene
Input: only `{video}` ({W}x{H}, {T} frames). Deliver at `{out}`.
Read `{out}/robot/README.md` and `robot_spec.json`. This is the source xArm7
collision URDF: seven arm joints plus six linked revolute gripper joints.
Use its body/assets/actuators, five mimic equalities and four linkage adjacency
exclusions. Only the xarm_base placement is estimated from the video; preserve
all internal mechanics. TCP is link_tcp, 0.172 m from link7. Controls are finite
(T,8): joint1..7 and drive_joint in radians (0 open, 0.85 closed). The native
controller limits drive_joint command slew to 2 rad/s. Use timestep 0.00005 s and mimic solref 0.0001 1; the runtime rejects mimic error above 0.001 rad at any physics step.
Model objects as dynamic passive bodies/flex, receiver as physical colliding
plate/walls, Earth gravity in the first-camera OpenCV frame. No attachments,
mocap, object actuators, state overrides or object coupling constraints.
protocol.json: protocol_version=3.0, physics_profile=twin_mujoco_v2,
status=success, coordinate_frame=first_camera_opencv, model_path=scene.xml,
robot.uid=xarm7_gripper, actions.path=actions.npy, actions.format=actuator_ctrl,
actions.dt=your control timestep. roles.object.bodies, roles.robot.bodies and
roles.target.bodies identify disjoint physical subtrees; roles.object.flex is
optional. Include source/video.mp4 and report.md. Follow robot/README.md's
complete scene.xml include recipe: compiler meshdir=".", robot_assets.xml
inside asset, robot_body.xml inside worldbody, robot_actuators.xml inside
actuator, robot_equalities.xml inside equality, robot_contacts.xml inside contact.
Run `PYTHONPATH={ours} {py} -m video2sim.cli validate {out}` then
`PYTHONPATH={ours} {py} -m video2sim.cli replay {out} --no-render`.
CLI success means execution completed; benchmark task scoring is separate.
Write only inside `{out}` and `/tmp/{name}`. Read no dataset GT, sample metadata,
calibration or reference trajectories. Estimate the scene and controls from pixels.

Task: Pack the soft toy into the visible box. Reconstruct a passive articulated or elastic object and a physical box floor and four walls.
No expected object trajectory is required or executed. Controls act only on the supplied robot.
