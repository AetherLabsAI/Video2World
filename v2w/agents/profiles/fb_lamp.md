# Task: reconstruct the manipulated PART and its motion from this video (FurnitureBench, object fidelity)

Input: exactly one file, `{video}` ({W}x{H}, letterboxed with SQUARE pixels: the camera image is the 16:9 frame scaled
to the full width and centred, so the picture occupies rows ~98-350 and the bands above/below are black padding;
{T} frames at 5 Hz, i.e. every 2nd frame of a 10 Hz capture). It shows a Franka Emika Panda (7-DoF, white links)
with its standard TWO-FINGER PARALLEL gripper on a black workbench in front of a green screen, picking up ONE white
3D-printed furniture part and moving it. Other parts of the same furniture sit on the bench and do NOT move.
You have NOTHING else: no robot logs, no calibration, no depth, no part list.

Deliverable: a scene package at `{out}` whose scored content is the MANIPULATED part: its shape (mesh or primitive),
size, pose at frame 0, and per-frame trajectory, all in the camera frame. Static parts may be delivered as `props`.

Scale cues (public robot geometry, use these rather than guessing): the Panda hand is 8 cm across at maximum opening
and its finger pads are ~2 cm long; the wrist (panda_hand) is ~10 cm from the fingertips; link 7 is ~7 cm in diameter.
The camera intrinsics are NOT available: estimate focal length and principal point yourself (the robot's known
geometry moving in depth is the strongest cue) and state your uncertainty in report.md.

TCP frame for `actions.npy` (this profile): origin midway between the two fingertip PADS (the grasp point you can see);
+z points from the wrist THROUGH the fingertips (the approach direction — straight down when the hand points down);
+y along the line between the two fingertips (the opening direction); +x completes the right-handed frame.
Track the visible gripper through the video to get the poses — the hand is in view for the whole episode.
