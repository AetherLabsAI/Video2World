"""MuJoCo cloth executor: a floating Robotiq-sized pad pair follows fingertip poses over an elastic sheet on a table plane.

Scene (robot base frame): {"table": {"point", "normal"} or {"z"}, "cloth": {"corners" (4x3) or "center"/"yaw"/"size",
"grid": [nx, ny], "mass", "young", "poisson", "thickness", "friction"}, "gravity": "table_normal" (optional)}.
The grasp belongs to the evaluator: when the gripper closes, cloth vertices inside the pad capture volume are connected
to the hand where they are; opening releases them. Gravity along minus the table normal avoids solver creep on a
tilted table; the table carries the scene friction because MuJoCo takes contact parameters from the higher-priority
geom. Physics runs at 0.5 ms; a simulation that MuJoCo silently reset after an instability is refused.
"""
import os
import numpy as np
os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco

PAD_LEN, PAD_WIDTH, PAD_THICK = 0.037, 0.022, 0.004     # Robotiq 2F-85 pad: along approach, across, thickness
OPEN_GAP = 0.085                                        # 2F-85 stroke
CAPTURE_MARGIN = 0.006

TPL = """
<mujoco model="cloth_exec">
  <option timestep="{dt}" integrator="implicitfast" solver="Newton" iterations="50" ls_iterations="20" gravity="{g}" noslip_iterations="10"/>
  <size memory="1G"/>
  <visual><global offwidth="960" offheight="720"/></visual>
  <worldbody>
    <light pos="0.4 0 2" dir="0 0 -1"/>
    <camera name="view" pos="{cx} {cy_cam} {cz_cam}" xyaxes="1 0 0 0 0.6 0.8"/>
    <geom name="table" type="plane" size="3 3 0.1" pos="{tpx} {tpy} {tpz}" quat="{tq}" friction="{mu} 0.01 0.001"
          solref="0.002 1" solimp="0.99 0.999 0.0001" priority="1" rgba="0.92 0.92 0.9 1"/>
    <body name="mount" mocap="true" pos="{hx} {hy} {hz}"/>
    <body name="hand" pos="{hx} {hy} {hz}">
      <freejoint/>
      <inertial pos="0 0 0" mass="1.0" diaginertia="0.003 0.003 0.003"/>
      <body name="left_pad" pos="0 {half} {pz}">
        <joint name="left_pad" type="slide" axis="0 -1 0" range="0 {stroke}" damping="40" armature="0.01"/>
        <geom name="left_pad" type="box" size="{pw} {pt} {pl}" rgba="0.15 0.15 0.18 1" friction="2 0.05 0.001"
              solimp="0.98 0.999 0.0005" solref="0.004 1"/>
      </body>
      <body name="right_pad" pos="0 -{half} {pz}">
        <joint name="right_pad" type="slide" axis="0 1 0" range="0 {stroke}" damping="40" armature="0.01"/>
        <geom name="right_pad" type="box" size="{pw} {pt} {pl}" rgba="0.15 0.15 0.18 1" friction="2 0.05 0.001"
              solimp="0.98 0.999 0.0005" solref="0.004 1"/>
      </body>
    </body>
    {flexopen}
      <contact internal="false" selfcollide="none" solimp="0.95 0.99 0.0001" solref="0.004 1" friction="{mu} 0.01 0.001"/>
      <elasticity young="{young}" poisson="{poisson}" thickness="{thick}" elastic2d="both"/>
    </flexcomp>
  </worldbody>
  <equality>
    <weld body1="hand" body2="mount" solref="0.004 1" solimp="0.95 0.99 0.0001"/>
{grips}
  </equality>
  <actuator>
    <position name="left_pad" joint="left_pad" ctrlrange="0 {stroke}" kp="400" kv="20" forcerange="-60 60"/>
    <position name="right_pad" joint="right_pad" ctrlrange="0 {stroke}" kp="400" kv="20" forcerange="-60 60"/>
  </actuator>
</mujoco>
"""


def _plane_quat(n):
    """Quaternion (wxyz) rotating +z onto the unit normal n."""
    n = np.asarray(n, float); n = n / np.linalg.norm(n); z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, n); s = np.linalg.norm(v); c = float(z @ n)
    if s < 1e-12: return np.array([1.0, 0, 0, 0])
    ang = np.arctan2(s, c); v = v / s
    return np.r_[np.cos(ang / 2), np.sin(ang / 2) * v]


def build(scene, tcp0, dt=0.0005):
    c = scene['cloth']; nx, ny = c['grid']; Lx, Ly = c['size']; t = scene['table']
    grips = '\n'.join('    <connect name="grip_%d" body1="cloth_%d" body2="hand" anchor="0 0 0" active="false" '
                      'solref="0.006 1" solimp="0.9 0.95 0.001"/>' % (i, i) for i in range(nx * ny))
    half = OPEN_GAP / 2 + PAD_THICK
    tp = t.get('point', [0.0, 0.0, t['z']]); tq = _plane_quat(t.get('normal', [0, 0, 1]))
    if 'rotation' in c: cq = quat_wxyz(np.asarray(c['rotation'], float))
    else: cq = np.r_[np.cos(c['yaw'] / 2), 0.0, 0.0, np.sin(c['yaw'] / 2)]
    fmt = lambda q: ' '.join('%.9f' % v for v in q)
    nrm = np.asarray(t.get('normal', [0, 0, 1]), float); nrm = nrm / np.linalg.norm(nrm)
    g = -9.81 * nrm if scene.get('gravity') == 'table_normal' else np.array([0.0, 0.0, -9.81])
    common = 'dim="2" mass="%s" radius="0.0015" rgba="0.18 0.18 0.2 1"' % c['mass']
    if 'corners' in c:
        # bilinear grid spanned by the 4 measured corners, x-major (nx, ny) like the grid type, so scoring densifies it the same
        C4 = np.asarray(c['corners'], float); u = np.linspace(0, 1, nx); v = np.linspace(0, 1, ny)
        pts = [((1 - a) * (1 - b)) * C4[0] + (a * (1 - b)) * C4[1] + (a * b) * C4[2] + ((1 - a) * b) * C4[3] for a in u for b in v]
        el = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                p00, p10, p01, p11 = i * ny + j, (i + 1) * ny + j, i * ny + j + 1, (i + 1) * ny + j + 1
                el += [p00, p10, p11, p00, p11, p01]
        flexopen = '<flexcomp name="cloth" type="direct" %s point="%s" element="%s">' % (
            common, ' '.join('%.6f %.6f %.6f' % tuple(q) for q in pts), ' '.join(map(str, el)))
    else:
        flexopen = ('<flexcomp name="cloth" type="grid" count="%d %d 1" spacing="%.6f %.6f 0.01" pos="%.6f %.6f %.6f" quat="%s" %s>'
                    % (nx, ny, Lx / (nx - 1), Ly / (ny - 1), c['center'][0], c['center'][1], c['center'][2], fmt(cq), common))
    xml = TPL.format(dt=dt, g=' '.join('%.6f' % v for v in g), tpx=tp[0], tpy=tp[1], tpz=tp[2], tq=fmt(tq),
                     hx=tcp0[0, 3], hy=tcp0[1, 3], hz=tcp0[2, 3], half=half, pz=-PAD_LEN / 2, stroke=OPEN_GAP / 2,
                     pw=PAD_WIDTH / 2, pt=PAD_THICK, pl=PAD_LEN / 2, flexopen=flexopen, mu=c.get('friction', 1.0),
                     young=c['young'], poisson=c.get('poisson', 0.3), thick=c['thickness'], grips=grips,
                     cx=c['center'][0], cy_cam=c['center'][1] - 0.75, cz_cam=tp[2] + 0.55)
    return mujoco.MjModel.from_xml_string(xml)


def quat_wxyz(R):
    q = np.empty(4); mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).ravel()); return q


def execute(scene, tcp, grip, fps, dt=0.0005, settle_s=0.5, tail_s=1.5, render=None, render_fps=15):
    """tcp: (T,4,4) fingertip poses; grip: (T,) closure 0..1. Returns dict with the executed cloth vertices sampled at every
    demo frame (T, Nv, 3), the grasp/release frames, and how many vertices the grasp captured."""
    tcp = np.asarray(tcp, float); grip = np.asarray(grip, float).ravel(); T = len(tcp)
    m = build(scene, tcp[0], dt); d = mujoco.MjData(m)
    mid = m.body('mount').mocapid[0]; hid = m.body('hand').id
    nv = scene['cloth']['grid'][0] * scene['cloth']['grid'][1]
    eq = [m.equality('grip_%d' % i).id for i in range(nv)]
    d.mocap_pos[mid] = tcp[0][:3, 3]; d.mocap_quat[mid] = quat_wxyz(tcp[0][:3, :3])
    mujoco.mj_forward(m, d)
    # the free hand starts where its mocap target is
    d.qpos[:3] = tcp[0][:3, 3]; d.qpos[3:7] = quat_wxyz(tcp[0][:3, :3]); mujoco.mj_forward(m, d)
    thr = 0.5 * float(grip.max()) if grip.max() > 0 else 1.0
    closed = grip > thr
    stroke = OPEN_GAP / 2
    rend = mujoco.Renderer(m, 720, 960) if render else None; frames = []
    last_t = [d.time]

    def step_checked():
        mujoco.mj_step(m, d)
        if d.time < last_t[0] - 1e-9:
            raise RuntimeError('MuJoCo reset the simulation after an instability at t=%.4f s; result refused' % last_t[0])
        last_t[0] = d.time

    for _ in range(int(settle_s / dt)): step_checked()

    def verts(): return d.flexvert_xpos.reshape(-1, 3).copy()

    def attach():
        V = verts(); c = d.xpos[hid]; Rm = d.xmat[hid].reshape(3, 3); loc = (V - c) @ Rm
        inside = ((np.abs(loc[:, 0]) < PAD_WIDTH / 2 + CAPTURE_MARGIN) & (np.abs(loc[:, 1]) < OPEN_GAP / 2) &
                  (loc[:, 2] > -PAD_LEN - CAPTURE_MARGIN) & (loc[:, 2] < CAPTURE_MARGIN))
        got = np.nonzero(inside)[0]
        for v in got:
            e = eq[int(v)]; m.eq_data[e, 0:3] = 0.0; m.eq_data[e, 3:6] = loc[v]; d.eq_active[e] = 1
        return got

    initial = verts()
    out = np.zeros((T, nv, 3), np.float32); captured = []; grasp_frame = release_frame = None
    steps = max(1, int(round(1.0 / (fps * dt)))); every = max(1, int(round(1.0 / (render_fps * dt))))
    step = 0; held = False
    for k in range(T):
        A, B = tcp[k], tcp[min(k + 1, T - 1)]
        qa, qb = quat_wxyz(A[:3, :3]), quat_wxyz(B[:3, :3])
        for s in range(steps):
            u = (s + 1) / steps
            d.mocap_pos[mid] = (1 - u) * A[:3, 3] + u * B[:3, 3]
            d.mocap_quat[mid] = qa if u < 0.5 else qb
            d.ctrl[:] = stroke if closed[k] else 0.0
            step_checked(); step += 1
            if rend is not None and step % every == 0:
                rend.update_scene(d, camera='view'); frames.append(rend.render())
        if closed[k] and not held:
            captured = attach(); held = True; grasp_frame = k
        elif not closed[k] and held:
            for e in eq: d.eq_active[e] = 0
            held = False; release_frame = k
        out[k] = verts()
    for _ in range(int(tail_s / dt)):
        d.ctrl[:] = 0.0; step_checked(); step += 1
        if rend is not None and step % every == 0:
            rend.update_scene(d, camera='view'); frames.append(rend.render())
    final = verts()
    if render and frames:
        import imageio_ffmpeg
        w = imageio_ffmpeg.write_frames(render, (frames[0].shape[1], frames[0].shape[0]), fps=render_fps, quality=7)
        w.send(None)
        for f in frames: w.send(np.ascontiguousarray(f))
        w.close()
    if rend is not None: rend.close()
    return dict(initial=initial, vertices=out, final=final, grid=list(scene['cloth']['grid']), captured=int(len(captured)), grasp_frame=grasp_frame, release_frame=release_frame,
                grasp_threshold=thr, dt=dt, fps=fps, finite=bool(np.isfinite(out).all()))
