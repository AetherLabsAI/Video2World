# Task: reconstruct the manipulated OBJECT, what it interacts with, and a ROBOT action stream that performs the same task, from this HAND-OBJECT video (hand track, setting A: Franka arm + gripper)

Input: exactly one file, `{video}` ({W}x{H}, {T} frames at 30 Hz). A HUMAN HAND -- no robot -- manipulates ONE object on a
table. Depending on the clip it is one of:
  (a) a pick-and-place: the hand carries a small object (a toy, a spoon, a spatula, ...) and puts it INTO or ONTO a receiving
      object (a bowl, a plate, a tray) that stays where it is;
  (b) a lift: the hand grasps a household object (a can, a bottle, a box, ...) among other objects on the table and lifts it;
      the clip ends with the object still in the hand -- there is no put-down;
  (c) an articulated action: the hand opens or closes the DOOR of a small appliance (a microwave oven) by its handle; the door
      swings about a hinge, the appliance body does not move.
Other objects on the table do NOT move. You have NOTHING else: no calibration, no depth, no object list, no hand model.

Camera: either a STATIC third-person RGB camera (a RealSense at 640x480, or a top-down view; if the picture has black bands
above and below, they are letterbox padding of a wider source and the picture occupies the rows in between), or a HEAD-MOUNTED
camera (Project Aria, rectified to a 640x640 pinhole) that MOVES with the wearer's head. In the moving case EVERYTHING you
deliver must be expressed in the camera frame OF FRAME 0: track the static scene to recover the camera motion (the table,
the objects that do not move) and re-express every later measurement in the frame-0 camera frame. State in report.md which
case you are in and how you handled it.

Deliverable: a scene package at `{out}` whose scored content is the MANIPULATED object: its shape (mesh or primitive), size,
pose at frame 0 and per-frame trajectory, all in the (frame-0) camera frame; PLUS the object it interacts with, delivered as a
`prop` and NAMED BY WHAT IT IS (`bowl`, `plate`, `tray`, `holder`, `microwave`, ...): the evaluator finds your receiving
prop by name and scores where the object ends RELATIVE to it, so a missing or oddly named receiving prop loses that column.
Deliver the table as a thin `box` prop if you can place it, and the other visible objects as props (sizes and poses).

The DOOR case (c): deliver the appliance as TWO entities -- the body as a static prop (name containing `microwave` or
`cavity`) and the door as the manipulated `object` (name containing `door` or `gate`) -- and declare the hinge so the
evaluator can build a real revolute joint:
```jsonc
"scene": {{ ..., "joints": [{{"name": "door_hinge", "parent": "<body prop name>", "child": "<door object name>",
                              "type": "revolute", "axis": [ax, ay, az], "point": [px, py, pz], "limits": [0.0, 1.6]}}] }}
```
`axis` (unit vector) and `point` (a point on the hinge line) are in the BODY prop's OWN local frame (the frame in which you
declared the body's pos/quat); the joint coordinate is 0 at frame 0 and positive in the direction the door moves in the
video (flip `axis` if needed); `limits` are in radians. Without a declared joint the door is a free rigid body and simply falls.

## Embodiment for `actions.npy` (setting A: a Franka Panda ARM with its original two-finger gripper)
Your actions drive a real 7-DoF Franka Panda arm with the standard Franka Hand (two parallel finger pads, ~2 cm wide, opening
up to 8 cm). The evaluator places the robot base on the table plane next to the workspace (a spot from which the demonstrated
motion is reachable; you do not choose it and you do not know it) and runs numerical IK from your TCP rows to joint position
targets, so a row the arm cannot reach is executed as closely as the arm can. TCP frame: origin midway between the two
fingertip pads; +z from the palm THROUGH the fingertips (the approach direction); +y along the line between the two finger
pads (the opening direction); +x completes the right-handed frame. Rows are `[x y z roll pitch yaw grip]` at 5 Hz as below, in
the (frame-0) camera frame, `grip` 1 = open, 0 = closed. Holding works like this: when `grip` goes 1 -> 0 the position-driven
fingers close with a force limit and stop on the object, and the object is held ONLY if a finger pad is actually touching it --
so put the grasp rows ON the object's surface with the pads straddling it (closing 3 cm away holds nothing). The arm holds still
while the fingers travel (up to 1.5 s). It stays held until `grip` goes back to 1; a door declared with a hinge is carried along
its hinge while held. Plan for a real arm: no more than a 3 cm / 0.3 rad move per row, approach and retreat along the TCP z
axis, keep the wrist orientation in a range a 7-DoF arm standing beside the table can realise (a top-down or side pinch is
fine; a pinch from underneath the table is not).

What the evaluator scores as the TASK (same outcome as the video, not the same path): (a) the object ends INSIDE / ON the
receiving prop -- its surface centroid within the prop's rim and above its floor -- and, if the person let go of it in the
video, released and at rest; (b) the object ends at least 10 cm above the table (still in the gripper is fine, the person
never lets go either); (c) the door ends within 10 degrees of the opening angle it reaches in the video. In every case the
scene must be at rest 1 s after your last row. As a secondary diagnostic the evaluator also replays the REAL hand's
motion (withheld) in your scene: where your objects really are decides whether that hand finds them.

Scale cues (no calibration is available): an adult hand is ~8-9 cm across the palm, the index finger is ~7-8 cm long, a
thumb-index pinch spans up to ~10 cm; use the hand as your ruler and state your focal-length estimate in report.md.
The trajectory file `{out}/expected/obj_poses.npy` must cover every one of the {T} video frames (frame-0 camera frame); for
the door case it is the DOOR's pose per frame.
