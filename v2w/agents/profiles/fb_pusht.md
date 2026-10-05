# Task: reconstruct the pushed T block and the pushing motion from this video (Push-T, single fixed camera)

Input: exactly one file, `{video}` ({W}x{H}, {T} frames at 5 Hz). It shows a UFACTORY xArm7 arm (7-DoF, white links)
fitted with a straight 20 cm PUSHER ROD (a thin vertical stick, NO gripper) on a black table, pushing ONE light
wooden T-shaped block (a bar with a stem, ~3 cm tall) across the table onto a goal. The yellow T you see on the
table is a RENDERED GOAL OVERLAY drawn into the video, not a physical object: it never moves, it does not collide,
and it must NOT be delivered as an object (leave it out, or at most a zero-height static prop). The camera is a
fixed oblique view; the image is a 240x240 crop of a wider camera and looks vertically stretched — estimate the
focal lengths per axis (fx != fy). You have NOTHING else: no robot logs, no calibration, no depth, no block CAD.

Deliverable: a scene package at `{out}` whose scored content is
  1. the T BLOCK: its shape (a mesh of TWO boxes — bar and stem — or an equivalent mesh), size, pose at frame 0 and
     per-frame trajectory in the camera frame (`scene.objects`, exactly one entry). Because the block is NOT convex,
     give `collision_path` an OBJ with two groups (`o bar` / `o stem`); the evaluator builds one convex collision
     per group. A single convex hull would fill the notch where the rod works and the push would behave wrongly.
  2. the table as `support` (or a thin static prop) so the block has something to slide on.
No receiving part exists in this family: Task Success is judged by where the block ends (its centre within 3 cm of
the goal and its planar orientation within 15 deg of the goal's), quiescent, after YOUR actions run in YOUR scene.

Scale cues (public robot geometry, use these rather than guessing): the rod is 20 cm long and ~1 cm thick, mounted
15 mm below the xArm7 flange; link 7 (the last cylindrical link above the rod) is ~7 cm in diameter; the block's
bar is ~15 cm long. The block is light and slides with moderate friction; it does not tip.

TCP frame for `actions.npy` (this profile): origin at the TIP of the pusher rod (the point that touches the block);
+z points from the flange THROUGH the rod towards the tip (straight down when the rod is vertical); +x, +y complete
a right-handed frame (the rod is a cylinder: yaw is free, keep it 0). There is no gripper: the `grip` column is
ignored by the evaluator — write 1 in every row. The rod tip rides ~0.5 cm above the table while pushing (the block
is 3 cm tall, so the tip contacts its side wall); a stream whose tip is above the block top pushes nothing.
Track the visible rod tip through the video to get the poses — the rod is in view for the whole episode.
The outcome that counts is the block ending on the goal marker the way the video shows it (bar and stem where the
yellow overlay is); plan your action stream to push YOUR block there in YOUR scene, pushing on its side walls, and
end with the rod lifted or stopped. Pushing is contact physics: a rod that passes through the block's pose without
contact moves nothing, so put the rod tip where the block's side wall is in YOUR scene.
