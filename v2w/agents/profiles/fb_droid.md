# Task: reconstruct the manipulated OBJECT, the object it interacts with, and the motion from this video (DROID, tabletop pick-and-place)

Input: exactly one file, `{video}` ({W}x{H}, letterboxed with SQUARE pixels: a wider camera image scaled to the full width and
centred, with black padding bands above and below; {T} frames at 7.5 Hz). It shows a Franka Panda / FR3 (7-DoF, white links)
fitted with a Robotiq 2F-85 TWO-FINGER PARALLEL gripper (black) on a table, seen from ONE fixed external camera. The robot
picks up ONE object and moves it to / onto / into another object or a spot next to another object. Other things on the
table do NOT move. You have NOTHING else: no instruction text, no robot logs, no calibration, no depth, no object list —
what the task is, you read from the video.

Deliverable: a scene package at `{out}` whose scored content is
  1. the MANIPULATED object (`scene.objects`, exactly ONE entry): its shape (mesh or primitive), size, pose at frame 0
     and per-frame trajectory, all in the camera frame. Hollow things (a cup, a bowl) must be modelled hollow (mesh with
     an inner wall, or `container`) — a gripper can only pinch a wall or a rim, and a solid cylinder cannot be grasped.
  2. the object the manipulated one ends on / in / next to, as a static `prop` with its size and pose — the outcome is
     judged by where the manipulated object ends relative to the scene, so the receiving object must be where it really
     is. Other static items may be props too.
  3. the table as `support` (or a thin static prop) with the correct plane orientation — objects rest on it under gravity.
Task Success is judged by where the manipulated object ends (within 5 cm of the demonstrated end position, released,
quiescent) after YOUR actions run in YOUR scene.

Scale cues (public robot geometry, use these rather than guessing): the Robotiq 2F-85 opens to 8.5 cm between the
finger pads, its pads are ~3.7 cm long and 2.2 cm wide, and the wrist flange is ~17.5 cm from the fingertips (it is a
long gripper); link 7 of the Franka is ~7 cm in diameter. The camera intrinsics are NOT available: estimate focal length
and principal point yourself (the robot's known geometry moving in depth is the strongest cue) and state your uncertainty
in report.md.

TCP frame for `actions.npy` (this profile): origin midway between the two fingertip PADS (the grasp point you can see);
+z points from the wrist THROUGH the fingertips (the approach direction — straight down when the hand points down);
+y along the line between the two fingertips (the opening direction); +x completes the right-handed frame.
Track the visible gripper through the video to get the poses — the hand is in view for the whole episode.
