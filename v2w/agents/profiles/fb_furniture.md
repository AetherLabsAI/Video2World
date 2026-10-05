# Task: reconstruct the manipulated PART, the part it is mounted on, and the motion from this video (FurnitureBench, furniture assembly)

Input: exactly one file, `{video}` ({W}x{H}, letterboxed with SQUARE pixels: the camera image is the 16:9 frame scaled
to the full width and centred, so the picture occupies rows ~98-350 and the bands above/below are black padding;
{T} frames at 5 Hz, i.e. every 2nd frame of a 10 Hz capture). It shows a Franka Emika Panda (7-DoF, white links)
with its standard TWO-FINGER PARALLEL gripper on a black workbench in front of a green screen, picking up ONE white
3D-printed furniture part and INSERTING it into / MOUNTING it onto ANOTHER, larger white part of the same furniture
that stays static on the bench (a table top with corner holes that a leg is screwed into; a drawer box that a
container tray is slid into; a cabinet body that a door is fitted onto). Other parts of the furniture may lie on
the bench and do NOT move. You have NOTHING else: no robot logs, no calibration, no depth, no part list.

Deliverable: a scene package at `{out}` whose scored content is
  1. the MANIPULATED part: its shape (mesh or primitive), size, pose at frame 0, and per-frame trajectory, all in the
     camera frame (`scene.objects`, exactly one entry);
  2. the RECEIVING part it ends up in / on, delivered as a static `prop` with its size and pose. Task success is judged
     by where the manipulated part ends RELATIVE TO THIS PROP (in the prop's own frame), so the prop MUST be present
     and MUST carry a recognisable name: use `table_top` for a table top / plate with holes, `drawer_box` for a
     drawer box / housing, `cabinet_body` for a cabinet body / shell. If you split the receiving part into several
     primitives, give every piece a name that starts with that word (e.g. `drawer_box_left_wall`). Without such a
     prop the evaluator has nothing to measure the outcome against and Task Success is lost.
Other static parts and the bench may be delivered as further `props` (any other names). Put the manipulated part's
initial pose where it really starts (on the bench or already in the hand) and the receiving part where it really
stands: the evaluator also replays the withheld REAL trajectory in your scene.

Scale cues (public robot geometry, use these rather than guessing): the Panda hand is 8 cm across at maximum opening
and its finger pads are ~2 cm long; the wrist (panda_hand) is ~10 cm from the fingertips; link 7 is ~7 cm in diameter.
The camera intrinsics are NOT available: estimate focal length and principal point yourself (the robot's known
geometry moving in depth is the strongest cue) and state your uncertainty in report.md.

TCP frame for `actions.npy` (this profile): origin midway between the two fingertip PADS (the grasp point you can see);
+z points from the wrist THROUGH the fingertips (the approach direction — straight down when the hand points down);
+y along the line between the two fingertips (the opening direction); +x completes the right-handed frame.
Track the visible gripper through the video to get the poses — the hand is in view for the whole episode.
The outcome that counts is the manipulated part ending in / on the receiving part the way the video shows it (e.g. the
leg standing in the corner hole, the tray slid into the box as far as the video pushes it, the door seated on the body)
and then released; plan your action stream to bring YOUR part to that place in YOUR scene and open the gripper there.
