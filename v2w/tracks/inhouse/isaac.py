"""Isaac Sim execution of an in-house candidate on the AgiBot G1 + OmniPicker robot.

Runs inside Isaac Sim's interpreter (``python -m v2w.tracks.inhouse.isaac rigid|wallet ...``) and imports only numpy,
scipy and PIL from outside Isaac. Rigid-body state is written only during initialization; after time zero every
interaction is driven by the native joint drives and PhysX.

  rigid   one rigid target (convex hull, or convex decomposition with --collision decomposition), 60 Hz physics,
          20 Hz commands, target contact sensor and a read-only PhysX audit
  wallet  rigid or elastic (particle-cloth shell) wallet target, 120 Hz GPU physics, surface + pad trajectories
"""
import argparse
import json
import math
import os
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

ARM_JOINTS = [f'idx2{i}_arm_l_joint{i}' for i in range(1, 8)] + [f'idx6{i}_arm_r_joint{i}' for i in range(1, 8)]
OTHER_JOINTS = ['idx01_body_joint1', 'idx02_body_joint2', 'idx11_head_joint1', 'idx12_head_joint2',
                'idx41_gripper_l_outer_joint1', 'idx81_gripper_r_outer_joint1']
MIMIC_JOINTS = ['idx31_gripper_l_inner_joint1', 'idx71_gripper_r_inner_joint1']


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')


# ---------------------------------------------------------------- Kit process isolation
def kit_config(output):
    """Per-process writable Kit caches and a bounded CPU worker count; no physics settings."""
    threads = int(os.environ.get('V2W_ISAAC_THREADS', '8'))
    if not 1 <= threads <= 256:
        raise ValueError('V2W_ISAAC_THREADS must be in [1, 256]')
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    threads = min(threads, available)
    parent = Path(os.environ.get('V2W_ISAAC_CACHE', str(Path(output) / 'kit_runtime'))).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='kit-', dir=parent))
    paths = {name: str(root / name) for name in ['cache', 'data', 'logs', 'DerivedDataCache', 'shadercache', 'nv_shadercache']}
    for path in paths.values():
        Path(path).mkdir()
    settings = {'/app/cachePath': paths['cache'], '/app/dataPath': paths['data'], '/app/logPath': paths['logs'],
                '/UJITSO/datastore/localCachePath': paths['DerivedDataCache'], '/app/tokens/omni_cache': paths['cache'],
                '/rtx/shaderDb/shaderCachePath': paths['shadercache'],
                '/rtx/shaderDb/driverShaderCachePath': paths['nv_shadercache']}
    config = dict(headless=True, enable_cameras=True, multi_gpu=False, limit_cpu_threads=threads,
                  extra_args=['--' + key + '=' + value for key, value in settings.items()])
    return config, dict(threads=threads, settings=settings)


def kit_verify(expected, settings):
    observed = {key: settings.get(key) for key in expected['settings']}
    threads = [settings.get('/plugins/carb.tasking.plugin/threadCount'), settings.get('/plugins/omni.tbb.globalcontrol/maxThreadCount')]
    if any(observed[k] != v for k, v in expected['settings'].items()) or any(t != expected['threads'] for t in threads):
        raise RuntimeError('Kit ignored the cache/thread isolation settings')


# ---------------------------------------------------------------- scene authoring (after SimulationApp)
def box_mesh(size):
    vertices = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]) * np.asarray(size) / 2
    return vertices, [4] * 6, [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7]


def author_scene(pkg, path, robot, collision='hull', wallet=False):
    """Candidate geometry plus the public robot. Rigid dynamic bodies collide by convex hull (or decomposition);
    static entities by their triangle mesh. In wallet mode an elastic target gets no rigid body or collider."""
    from pxr import Usd, UsdGeom, UsdPhysics, Gf, UsdShade
    pkg = Path(pkg)
    scene = json.loads((pkg / 'scene.json').read_text())
    if not Path(robot).is_file():
        raise FileNotFoundError(robot)
    st = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.)
    if not wallet:
        st.SetTimeCodesPerSecond(60.)   # the native clock; the USD default (24 Hz) loses robot render updates
    world = UsdGeom.Xform.Define(st, '/World')
    st.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(st, '/World/Robot').GetPrim().GetReferences().AddReference(str(robot))
    UsdPhysics.Scene.Define(st, '/World/PhysicsScene').CreateGravityMagnitudeAttr(9.81)
    UsdPhysics.Scene.Get(st, '/World/PhysicsScene').CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    entities = [(e, True) for e in scene['objects']] + [(e, False) for e in scene.get('props', [])] + ([(scene['table'], False)] if scene.get('table') else [])
    for e, dynamic in entities:
        prim = UsdGeom.Xform.Define(st, '/World/Entities/' + e['name'])
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, e['pos'])))
        q = e['quat_wxyz']
        xf.AddOrientOp().Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*map(float, q[1:]))))
        if e.get('geometry_npz'):
            z = np.load(pkg / e['geometry_npz'], allow_pickle=False)
            vertices, counts, indices = z['vertices'], z['face_vertex_counts'], z['face_vertex_indices']
        else:
            vertices, counts, indices = box_mesh(e['size'])
        mesh = UsdGeom.Mesh.Define(st, str(prim.GetPath()) + '/mesh')
        mesh.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in vertices])
        mesh.CreateFaceVertexCountsAttr(list(map(int, counts)))
        mesh.CreateFaceVertexIndicesAttr(list(map(int, indices)))
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*map(float, e['color']))])
        rigid = dynamic and not (wallet and e.get('deformable'))
        if wallet and dynamic and e['name'] != 'target':
            raise ValueError('The wallet profile supports one dynamic target')
        if rigid or not dynamic:
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
            approximation = ('convexDecomposition' if collision == 'decomposition' else 'convexHull') if rigid else 'none'
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(approximation)
        if rigid:
            UsdPhysics.RigidBodyAPI.Apply(prim.GetPrim()).CreateKinematicEnabledAttr(False)
            UsdPhysics.MassAPI.Apply(prim.GetPrim()).CreateMassAttr(float(e.get('mass', .1)))
        mat = UsdShade.Material.Define(st, str(prim.GetPath()) + '/material')
        api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        api.CreateStaticFrictionAttr(float(e.get('static_friction', .6)))
        api.CreateDynamicFrictionAttr(float(e.get('dynamic_friction', .6)))
        api.CreateRestitutionAttr(float(e.get('restitution', 0)))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat, materialPurpose='physics')
    st.GetRootLayer().Save()


# ---------------------------------------------------------------- physical audit
class PhysicsAudit:
    """Read-only PhysX audit: contact separations (including zero-force overlaps), joint limits and critical USD edits."""

    def __init__(self, stage, target, out):
        from pxr import UsdPhysics, PhysxSchema, Tf, Usd
        from omni.physx import get_physx_simulation_interface
        self.stage, self.target, self.out = stage, target, Path(out)
        self.started, self.step = False, -1
        self.events, self.errors, self.min_sep, self.deepest = 0, [], 0., None
        self.joint_violations, self.max_joint_excess, self.changed = {}, {}, []
        self.UsdPhysics = UsdPhysics
        for path in (target, '/World/Robot'):
            PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr().Set(0.)
        self.contact_log = (self.out / 'contact_separations.jsonl').open('w', buffering=1)
        self.sub = get_physx_simulation_interface().subscribe_contact_report_events(self.on_contact)
        self.notice = Tf.Notice.RegisterGlobally(Usd.Notice.ObjectsChanged, self.on_notice)

    def on_notice(self, notice, sender):
        if not self.started or sender != self.stage:
            return
        for p in list(notice.GetResyncedPaths()) + list(notice.GetChangedInfoOnlyPaths()):
            p = str(p)
            # Drives are actions; collision, constraint and kinematic edits never are.
            if any(t in p for t in ['physics:collisionEnabled', 'physics:kinematicEnabled', 'physics:body0', 'physics:body1', 'physics:jointEnabled', 'physxCollision:']):
                self.changed.append(dict(step=self.step, path=p))

    def on_contact(self, headers, data):
        if not self.started:
            return
        from pxr import PhysicsSchemaTools
        try:
            for h in headers:
                b = [str(PhysicsSchemaTools.intToSdfPath(getattr(h, k))) for k in ('actor0', 'actor1')]
                if not any(x.startswith('/World/Robot') or x == self.target for x in b):
                    continue
                for i in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
                    c = data[i]
                    sep = float(c.separation)
                    if not math.isfinite(sep):
                        raise ValueError('nonfinite separation')
                    r = dict(step=self.step, body0=b[0], body1=b[1], separation_m=sep, position=list(c.position), normal=list(c.normal), impulse=list(c.impulse))
                    self.events += 1
                    self.contact_log.write(json.dumps(r) + '\n')
                    if sep < self.min_sep:
                        self.min_sep, self.deepest = sep, r
        except Exception as e:
            self.errors.append(repr(e))

    def colliders(self):
        return {str(p.GetPath()): bool(self.UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()) for p in self.stage.Traverse() if p.HasAPI(self.UsdPhysics.CollisionAPI)}

    def constraints(self):
        J = self.UsdPhysics.Joint
        return {str(p.GetPath()): dict(type=p.GetTypeName(), body0=[str(x) for x in J(p).GetBody0Rel().GetTargets()], body1=[str(x) for x in J(p).GetBody1Rel().GetTargets()])
                for p in self.stage.Traverse() if p.IsA(J)}

    def start(self, robot):
        self.names, self.limits = robot.dof_names, {}
        joint_prims = {p.GetName(): p for p in self.stage.Traverse() if p.IsA(self.UsdPhysics.Joint)}
        for name in self.names:
            p = joint_prims.get(name)
            if p is None:
                raise ValueError('missing USD joint ' + name)
            lo, hi = p.GetAttribute('physics:lowerLimit').Get(), p.GetAttribute('physics:upperLimit').Get()
            if lo is None or hi is None:
                continue
            angular = p.GetTypeName() == 'PhysicsRevoluteJoint'
            scale = np.pi / 180 if angular else 1
            self.limits[name] = dict(lower=float(lo) * scale, upper=float(hi) * scale, unit='rad' if angular else 'm', numeric_tolerance=1e-4 if angular else 1e-5)
        self.before_colliders, self.before_constraints = self.colliders(), self.constraints()
        self.started = True

    def sample(self, state, step):
        self.step = step
        for name, q in zip(self.names, state):
            lim = self.limits.get(name)
            if not lim:
                continue
            excess = max(lim['lower'] - q, q - lim['upper'], 0.)
            self.max_joint_excess[name] = max(float(excess), self.max_joint_excess.get(name, 0.))
            if excess > lim['numeric_tolerance'] and name not in self.joint_violations:
                self.joint_violations[name] = dict(joint=name, step=step, position=float(q), excess=float(excess), **lim)

    def finish(self):
        self.started = False
        self.contact_log.close()
        after, cons = self.colliders(), self.constraints()
        disabled = [p for p, v in self.before_colliders.items() if v and not after.get(p, False)]
        illegal = [dict(path=p, **v) for p, v in self.before_constraints.items() if self.target in v['body0'] + v['body1']]
        illegal.extend(dict(path=p, reason='constraint changed during execution') for p in set(cons) | set(self.before_constraints) if cons.get(p) != self.before_constraints.get(p))
        complete = not self.errors and self.events > 0 and bool(self.limits)
        penetration_limit = .002
        valid = bool(complete and not self.joint_violations and not disabled and not illegal and not self.changed and -self.min_sep <= penetration_limit)
        return dict(complete=complete, valid=valid, status='pass' if valid else ('fail' if complete else 'unknown'),
                    contact_report_points=self.events, contact_report_errors=self.errors,
                    max_critical_penetration_m=-self.min_sep, max_penetration_allowed_m=penetration_limit, deepest_contact=self.deepest,
                    joint_limit_violations=list(self.joint_violations.values()), joint_limits=self.limits, max_joint_excess=self.max_joint_excess,
                    disabled_critical_collisions=disabled, illegal_constraints=illegal, critical_property_mutations=self.changed)


# ---------------------------------------------------------------- rigid target
def run_rigid(a):
    pkg, out = Path(a.package), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    pr = json.loads((pkg / 'protocol.json').read_text())
    actions = np.load(pkg / 'actions.npy')
    post = np.load(pkg / 'posture.npz')
    n = len(actions) if a.limit is None else min(a.limit, len(actions))
    tstart = time.time()
    status = dict(status='starting', sample=pr['sample'], profile=pr['physics_profile'], smoke_test=a.limit is not None,
                  frames_requested=n, video=a.video_dir is not None)
    write(out / 'execution.json', status)
    config, kit = kit_config(out)
    if 'V2W_ISAAC_GPU' in os.environ:
        config.update(active_gpu=int(os.environ['V2W_ISAAC_GPU']), physics_gpu=int(os.environ['V2W_ISAAC_GPU']))
    from isaacsim import SimulationApp
    app = None
    try:
        app = SimulationApp(config)
        import carb.settings
        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.prims import SingleArticulation, RigidPrim
        from isaacsim.sensors.physics import ContactSensor
        from scipy.spatial.transform import Rotation
        from PIL import Image
        settings = carb.settings.get_settings()
        kit_verify(kit, settings)
        if a.video_dir:
            settings.set_bool('/isaaclab/render/offscreen', True)
            settings.set_bool('/isaaclab/render/active_viewport', False)
            settings.set_bool('/isaaclab/render/rtx_sensors', False)
            settings.set_bool('/isaaclab/cameras_enabled', True)
            for key in ['/app/asyncRendering', '/app/asyncRenderingLowLatency', '/omni/replicator/asyncRendering', '/rtx-transient/dlssg/enabled']:
                settings.set_bool(key, False)
        settings.set_bool('/physics/updateToUsd', True)
        author_scene(pkg, out / 'input_scene.usda', a.robot, collision=a.collision)
        omni.usd.get_context().open_stage(str(out / 'input_scene.usda'))
        app.update()
        stage = omni.usd.get_context().get_stage()
        for _ in range(10):
            app.update()
        target = stage.GetPrimAtPath(pr['target'])
        assert target.IsValid(), pr['target']
        target.SetActive(True)
        if not target.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError('target has no rigid body')

        def pose(prim, p, q):
            xf = UsdGeom.Xformable(prim)
            scale = np.ones(3)
            for op in xf.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    scale = np.asarray(op.Get(), float)
            xf.ClearXformOpOrder()
            m = np.eye(4)
            m[:3, :3] = Rotation.from_quat(np.roll(q, -1)).as_matrix() @ np.diag(scale)
            m[:3, 3] = p
            xf.AddTransformOp(opSuffix='benchmarkPose').Set(Gf.Matrix4d(m.T.tolist()))

        base = np.array(pr['T_world_base'])
        pose(stage.GetPrimAtPath('/World/Robot'), base[:3, 3], np.roll(Rotation.from_matrix(base[:3, :3]).as_quat(), 1))
        pose(target, pr['initial_object_xyz'], pr['initial_object_wxyz'])
        UsdPhysics.RigidBodyAPI(target).CreateKinematicEnabledAttr(False)
        joints = {p.GetName(): p for p in stage.Traverse() if 'Joint' in p.GetTypeName()}
        names = ARM_JOINTS + OTHER_JOINTS
        drives = []
        for name in names:
            p = joints[name]
            kind = 'linear' if p.GetTypeName() == 'PhysicsPrismaticJoint' else 'angular'
            d = UsdPhysics.DriveAPI.Get(p, kind)
            if not d:
                raise ValueError('missing native drive ' + name)
            drives.append((d, kind))
        for d, _ in drives[-4:-2]:
            (d.GetStiffnessAttr() or d.CreateStiffnessAttr()).Set(80000.)
            (d.GetDampingAttr() or d.CreateDampingAttr()).Set(2500.)
        for d, _ in drives[-2:]:
            (d.GetStiffnessAttr() or d.CreateStiffnessAttr()).Set(500.)
            (d.GetDampingAttr() or d.CreateDampingAttr()).Set(100.)
        opens = [min(.9, np.deg2rad(float(joints[name].GetAttribute('physics:upperLimit').Get())) - .001) for name in OTHER_JOINTS[-2:]]

        def command(i):
            q = np.r_[actions[i, :14], post['waist'][i, 1], post['waist'][i, 0], post['head'][i], (1 - actions[i, 14:]) * np.array(opens)]
            for v, (d, kind) in zip(q, drives):
                (d.GetTargetPositionAttr() or d.CreateTargetPositionAttr()).Set(float(v if kind == 'linear' else np.rad2deg(v)))
            return q

        q0 = command(0)
        # The contact report is observational and active for every physics step.
        sensor = ContactSensor(prim_path=pr['target'] + '/BenchmarkContacts', name='benchmark_contacts', dt=1 / 60, translation=np.zeros(3), min_threshold=0, max_threshold=1e9, radius=-1)
        sensor.add_raw_contact_data_to_frame()
        audit = PhysicsAudit(stage, pr['target'], out)
        sim = SimulationContext(physics_dt=1 / 60, rendering_dt=1 / 60, physics_prim_path='/World/PhysicsScene', set_defaults=True)
        PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath('/World/PhysicsScene')).CreateEnableEnhancedDeterminismAttr(True)
        sim.initialize_physics()
        sim.reset()
        robot = SingleArticulation('/World/Robot')
        robot.initialize()
        obj = RigidPrim(pr['target'])
        obj.initialize()
        sensor.initialize()
        idx = np.array([robot.get_dof_index(name) for name in names])
        initial_q = q0.copy()
        initial_q[:14] = post['initial_arm']
        robot.set_joint_positions(initial_q.astype(np.float32), joint_indices=idx)
        robot.set_joint_velocities(np.zeros(len(idx), dtype=np.float32), joint_indices=idx)
        for k, name in enumerate(MIMIC_JOINTS):   # mimic followers are made consistent only at initialization
            robot.set_joint_positions(np.array([-q0[-2 + k]], np.float32), joint_indices=np.array([robot.get_dof_index(name)]))
        obj.set_world_poses(positions=np.array([pr['initial_object_xyz']], np.float32), orientations=np.array([pr['initial_object_wxyz']], np.float32))
        obj.set_velocities(np.zeros((1, 6), np.float32))
        # One initialization step propagates articulation FK before frame zero; it is outside the evaluation clock.
        sim.step(render=False)
        stage.GetRootLayer().Export(str(out / 'initialized_scene.usda'))
        annotators = {}
        video_dir = Path(a.video_dir).resolve() if a.video_dir else None
        if video_dir:
            import omni.replicator.core as rep
            from pxr import UsdLux
            UsdLux.DomeLight.Define(stage, '/World/BenchmarkReviewLight').CreateIntensityAttr(1000.)
            for key, name, res, fl, ha, va in [('head', 'Head_Camera', (640, 400), 1.93, 3.896, 2.453), ('left', 'Left_Camera', (424, 240), 1.93, 3.760, 2.131), ('right', 'Right_Camera', (424, 240), 1.93, 3.760, 2.131)]:
                cam = next(p for p in stage.Traverse() if p.GetName() == name and p.IsA(UsdGeom.Camera))
                c = UsdGeom.Camera(cam)
                c.GetFocalLengthAttr().Set(fl)
                c.GetHorizontalApertureAttr().Set(ha)
                c.GetVerticalApertureAttr().Set(va)
                product = rep.create.render_product(str(cam.GetPath()), res)
                an = rep.AnnotatorRegistry.get_annotator('rgb', device='cpu', do_array_copy=True)
                an.attach([product])
                annotators[key] = an
                (video_dir / key).mkdir(parents=True, exist_ok=False)

        def capture(frame):
            if not annotators:
                return
            before = (float(sim.current_time), np.array(robot.get_joint_positions()), np.array(obj.get_world_poses()[0]))
            for _ in range(5):
                for _ in range(12 if frame == 0 else 2):
                    sim.render()
                images = {key: an.get_data() for key, an in annotators.items()}
                if all(im is not None and im.size for im in images.values()):
                    break
            for key, im in images.items():
                if im is None or not im.size:
                    raise RuntimeError('camera missing ' + key)
                Image.fromarray(im[..., :3]).save(video_dir / key / f'f_{frame:04d}.jpg', quality=85)
            if sim.current_time != before[0] or not np.array_equal(robot.get_joint_positions(), before[1]) or not np.array_equal(obj.get_world_poses()[0], before[2]):
                raise RuntimeError('Rendering advanced physics or changed recorded state')

        collisions = [str(p.GetPath()) for p in Usd.PrimRange(target) if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
        if not collisions:
            raise ValueError('target has no enabled collider')
        rows, decode_errors = [], 0

        def record(i, substep):
            nonlocal decode_errors
            p, q = obj.get_world_poses()
            vel = obj.get_velocities()
            sensor_frame = sensor.get_current_frame()
            raw = sensor._contact_sensor_interface.get_rigid_body_raw_data(pr['target'])
            contacts = []
            for c in raw:
                try:
                    b0 = str(sensor._contact_sensor_interface.decode_body_name(int(c['body0'])))
                    b1 = str(sensor._contact_sensor_interface.decode_body_name(int(c['body1'])))
                    contacts.append(dict(body0=b0, body1=b1, impulse=np.asarray(c['impulse']).tolist()))
                except Exception:
                    decode_errors += 1
            state = np.array(robot.get_joint_positions())
            bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'], useExtentsHint=False).ComputeWorldBound(target).ComputeAlignedRange()
            r = dict(bounds=[list(bbox.GetMin()), list(bbox.GetMax())], frame=i, substep=substep, physics_time_s=float(sim.current_time),
                     xyz=p[0].tolist(), wxyz=q[0].tolist(), velocity=vel[0].tolist(), joints=state.tolist(), contacts=contacts,
                     sensor_time_s=float(sensor_frame.get('time', 0)), sensor_in_contact=sensor_frame.get('in_contact'), raw_contact_count=len(raw))
            if not np.isfinite(np.r_[p.ravel(), q.ravel(), vel.ravel(), state]).all():
                raise ValueError('nonfinite physics state')
            return r

        audit.start(robot)
        rows.append(record(0, 0))
        audit.sample(rows[-1]['joints'], 0)
        capture(0)
        steps = 0
        with (out / 'physics_steps.jsonl').open('w', buffering=1) as log:
            for i in range(n + pr['settle_frames'] - 1):
                command(min(i, n - 1))
                for j in range(3):
                    audit.step = i * 3 + j + 1
                    sim.step(render=False)
                    rr = record(i + 1, j + 1)
                    audit.sample(rr['joints'], i * 3 + j + 1)
                    log.write(json.dumps(rr) + '\n')
                    steps += 1
                rows.append(rr)
                capture(i + 1)
                if i % 50 == 0:
                    print(pr['kind'], i, n, flush=True)
        np.savez_compressed(out / 'trajectory.npz', time_s=np.arange(len(rows)) * .05, xyz=[r['xyz'] for r in rows], wxyz=[r['wxyz'] for r in rows],
                            velocity=[r['velocity'] for r in rows], joints=[r['joints'] for r in rows],
                            command_timestamp_ns=post['timestamp_ns'][:n], joint_names=robot.dof_names)
        write(out / 'observations.json', rows)
        status.update(status='executed', frames=len(rows), control_frames=n, settle_frames=pr['settle_frames'], seconds=time.time() - tstart, physics_steps=steps,
                      audit=dict(no_post_start_rigid_body_state_writes=True, no_kinematic_target=True, target_enabled_colliders=collisions,
                                 contact_decode_errors=decode_errors, **audit.finish()))
        if annotators:
            status['video_capture'] = dict(directory=str(video_dir), views=list(annotators), frames=len(rows), fps=20, includes_settle=True)
        write(out / 'execution.json', status)
    except Exception as exc:
        status.update(status='failed', error=repr(exc), trace=traceback.format_exc(), seconds=time.time() - tstart)
        write(out / 'execution.json', status)
        raise
    finally:
        if app is not None:
            app.close()


# ---------------------------------------------------------------- wallet target
def surface_from_state(local_vertices, pose_xyzw, velocity):
    """Rigid-body state to surface vertices and vertex velocities."""
    from scipy.spatial.transform import Rotation
    v, pose, velocity = (np.asarray(x, float) for x in (local_vertices, pose_xyzw, velocity))
    if v.ndim != 2 or v.shape[1] != 3 or pose.shape != (7,) or velocity.shape != (6,) or not all(np.isfinite(x).all() for x in [v, pose, velocity]):
        raise ValueError('Invalid native rigid surface state')
    if not np.isclose(np.linalg.norm(pose[3:]), 1., atol=2e-3):
        raise ValueError('Invalid native rigid quaternion')
    offsets = v @ Rotation.from_quat(pose[3:]).as_matrix().T
    return offsets + pose[:3], velocity[:3] + np.cross(velocity[3:], offsets)


def run_wallet(a):
    """PhysX rigid mesh or particle-cloth closed shell. Cloth tensors (not USD display points) are authoritative;
    fingertip evidence is the pad collision geometry, never a force."""
    pkg, out = Path(a.package).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    pr = json.loads((pkg / 'protocol.json').read_text())
    scene = json.loads((pkg / 'scene.json').read_text())
    q = np.load(pkg / 'actions.npy')
    elastic = bool(scene['objects'][0].get('deformable'))
    n = len(q) if a.limit is None else min(len(q), a.limit)
    settle = 40 if a.limit is None else 0
    start = time.time()
    status = dict(status='starting', sample=pr['sample'], physics_dt=1 / 120, control_dt=.05, smoke_test=a.limit is not None, control_frames=n, settle_frames=settle,
                  physics_model='PhysX particle spring shell' if elastic else 'PhysX dynamic rigid body, convex-hull collision', target_dynamics='elastic' if elastic else 'rigid')
    write(out / 'execution.json', status)
    from isaacsim import SimulationApp
    gpu = int(os.environ.get('V2W_ISAAC_GPU', '0'))
    device = 'cuda:' + str(gpu)
    app = SimulationApp({'headless': True, 'active_gpu': gpu, 'physics_gpu': gpu, 'multi_gpu': False, 'width': 320, 'height': 200})
    failed = False
    try:
        import torch
        import omni.usd
        import omni.physics.tensors
        from pxr import Usd, UsdGeom, UsdPhysics, Tf
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleParticleSystem
        from isaacsim.core.api.materials.particle_material import ParticleMaterial
        from omni.physx.scripts import particleUtils
        from scipy.spatial.transform import Rotation  # noqa: F401  (Isaac's scipy must be importable before stepping)
        array = lambda x: x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)
        author_scene(pkg, out / 'scene.usda', a.robot, wallet=True)
        omni.usd.get_context().open_stage(str(out / 'scene.usda'))
        app.update()
        stage = omni.usd.get_context().get_stage()
        world = World(physics_dt=1 / 120, rendering_dt=1 / 120, physics_prim_path='/World/PhysicsScene', backend='torch', device=device)
        world.get_physics_context().enable_gpu_dynamics(True)
        world.get_physics_context().set_broadphase_type('GPU')
        target = next(e for e in scene['objects'] if e['name'] == 'target')
        model = target.get('deformable')
        meshpath = '/World/Entities/target/mesh'
        if elastic:
            radius = float(model['particle_radius_m'])
            ps = SingleParticleSystem('/World/Particles', simulation_owner='/World/PhysicsScene', rest_offset=radius, contact_offset=2 * radius, solid_rest_offset=radius,
                                      fluid_rest_offset=radius, particle_contact_offset=2 * radius, solver_position_iteration_count=16)
            ps.apply_particle_material(ParticleMaterial('/World/ParticleMaterial', friction=float(target.get('dynamic_friction', .6))))
            particleUtils.add_physx_particle_cloth(stage, meshpath, dynamic_mesh_path=None, particle_system_path='/World/Particles', spring_stretch_stiffness=model['stretch_stiffness'],
                                                   spring_bend_stiffness=model['bend_stiffness'], spring_shear_stiffness=model['shear_stiffness'], spring_damping=model['damping'],
                                                   pressure=model['pressure'], self_collision=True, self_collision_filter=True, particle_group=0)
            UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(meshpath)).CreateMassAttr(float(target['mass']))
        mesh = UsdGeom.Mesh.Get(stage, meshpath)
        local_vertices = np.asarray(mesh.GetPointsAttr().Get(), float)
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), np.int32).reshape(-1, 3)
        V = len(mesh.GetPointsAttr().Get())
        joints = {x.GetName(): x for x in stage.Traverse() if x.IsA(UsdPhysics.Joint)}
        names = ARM_JOINTS + OTHER_JOINTS
        drives = []
        for name in names:
            joint = joints[name]
            linear = joint.GetTypeName() == 'PhysicsPrismaticJoint'
            d = UsdPhysics.DriveAPI.Get(joint, 'linear' if linear else 'angular')
            assert d, 'missing native drive'
            drives.append((d, linear))
        for d, _ in drives[-4:-2]:
            d.GetStiffnessAttr().Set(80000.)
            d.GetDampingAttr().Set(2500.)
        for d, _ in drives[-2:]:
            d.GetStiffnessAttr().Set(500.)
            d.GetDampingAttr().Set(100.)
        opens = [min(.9, np.deg2rad(float(joints[name].GetAttribute('physics:upperLimit').Get())) - .001) for name in names[-2:]]
        ready = False

        def command(i):
            cmd = q[i].copy()
            cmd[-2:] = (1 - cmd[-2:]) * opens
            if ready:
                from isaacsim.core.utils.types import ArticulationAction
                robot.apply_action(ArticulationAction(joint_positions=torch.tensor(cmd, dtype=torch.float32, device=device), joint_indices=idx))
            else:
                for value, (d, linear) in zip(cmd, drives):
                    d.GetTargetPositionAttr().Set(float(value if linear else np.rad2deg(value)))
            return cmd

        q0 = command(0)
        robot = SingleArticulation('/World/Robot', name='robot')
        world.scene.add(robot)
        world.reset()
        idx = torch.tensor([robot.get_dof_index(x) for x in names], device=device)
        robot.set_joint_positions(torch.tensor(q0, dtype=torch.float32, device=device), joint_indices=idx)
        robot.set_joint_velocities(torch.zeros(20, device=device), joint_indices=idx)
        for k, name in enumerate(MIMIC_JOINTS):
            robot.set_joint_positions(torch.tensor([-q0[-2 + k]], dtype=torch.float32, device=device), joint_indices=torch.tensor([robot.get_dof_index(name)], device=device))
        ready = True
        command(0)
        world.step(render=False)
        sv = omni.physics.tensors.create_simulation_view('torch')
        sv.set_subspace_roots('/')
        if elastic:
            cloth = sv.create_particle_cloth_view(meshpath)
            if cloth.count != 1 or cloth.max_particles_per_cloth != V:
                raise ValueError('Cloth tensor topology mismatch')
        else:
            body = sv.create_rigid_body_view('/World/Entities/target')
            if body.count != 1:
                raise ValueError('Rigid wallet tensor binding mismatch')

        def surface_state():
            if elastic:
                return array(cloth.get_positions()).reshape(V, 3).copy(), array(cloth.get_velocities()).reshape(V, 3).copy()
            return surface_from_state(local_vertices, array(body.get_transforms()).reshape(7), array(body.get_velocities()).reshape(6))

        pads, padlocal, cache = [], {}, UsdGeom.XformCache()
        for hand in ['l', 'r']:
            for side in ['inner', 'outer']:
                name = f'gripper_{hand}_{side}_link4'
                prim = next(x for x in stage.Traverse() if x.GetName() == name and x.HasAPI(UsdPhysics.RigidBodyAPI))
                path = str(prim.GetPath())
                pads.append((path, sv.create_rigid_body_view(path)))
                vs, fs, offset = [], [], 0
                inv = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(prim)).T)
                for x in Usd.PrimRange(prim):
                    if not x.IsA(UsdGeom.Mesh) or '/collisions/' not in str(x.GetPath()):
                        continue
                    m = UsdGeom.Mesh(x)
                    v = np.asarray(m.GetPointsAttr().Get())
                    t = inv @ np.asarray(cache.GetLocalToWorldTransform(x)).T
                    v = v @ t[:3, :3].T + t[:3, 3]
                    counts = m.GetFaceVertexCountsAttr().Get()
                    ii = np.asarray(m.GetFaceVertexIndicesAttr().Get())
                    j = 0
                    for count in counts:
                        poly = ii[j:j + count]
                        j += count
                        fs.extend([[offset + poly[0], offset + poly[k], offset + poly[k + 1]] for k in range(1, count - 1)])
                    vs.extend(v)
                    offset += len(v)
                if not vs:
                    raise ValueError('No pad collision mesh ' + path)
                padlocal[name + '_vertices'] = np.asarray(vs)
                padlocal[name + '_faces'] = np.asarray(fs, np.int32)
        np.savez_compressed(out / 'pad_geometry.npz', **padlocal)
        stage.GetRootLayer().Export(str(out / 'initialized_scene.usda'))
        mutated, started = [], True

        def on_notice(notice, sender):
            if sender != stage or not started:
                return
            for p in list(notice.GetResyncedPaths()) + list(notice.GetChangedInfoOnlyPaths()):
                s = str(p)
                if any(k in s for k in ['physics:collisionEnabled', 'physics:kinematicEnabled', 'physics:body0', 'physics:body1', 'physxParticle:particleEnabled', 'physxParticle:particleSystem', 'springStiffness', 'physics:mass']):
                    mutated.append(s)

        notice = Tf.Notice.RegisterGlobally(Usd.Notice.ObjectsChanged, on_notice)  # noqa: F841  (kept alive for the run)
        vertices, velocities, poses, jointstates, times, step = [], [], [], [], [], 0
        jointlimits = robot._articulation_view.get_dof_limits()
        jointnames = robot.dof_names
        maxspeed = maxjoint_excess = 0.

        def record():
            v, vel = surface_state()
            j = array(robot.get_joint_positions()).copy()
            pose = np.concatenate([array(view.get_transforms()) for _, view in pads])
            if not all(np.isfinite(x).all() for x in [v, vel, j, pose]):
                raise ValueError('Nonfinite native physics state')
            vertices.append(v)
            velocities.append(vel)
            poses.append(pose)
            jointstates.append(j)
            times.append(step / 120)

        record()
        with (out / 'physics_audit.jsonl').open('w', buffering=1) as log:
            for frame in range(n + settle - 1):
                command(min(frame, n - 1))
                for _ in range(6):
                    world.step(render=False)
                    step += 1
                    v, vel = surface_state()
                    j = array(robot.get_joint_positions())
                    lim = array(jointlimits).reshape(-1, 2)
                    if not np.isfinite(v).all() or not np.isfinite(vel).all() or np.max(np.abs(v)) > 100:
                        raise ValueError('Particle simulation diverged')
                    speed = float(np.linalg.norm(vel, axis=1).max())
                    excess = float(np.maximum(np.maximum(lim[:, 0] - j, j - lim[:, 1]), 0).max())
                    maxspeed, maxjoint_excess = max(maxspeed, speed), max(maxjoint_excess, excess)
                    log.write(json.dumps(dict(step=step, time_s=step / 120, center=v.mean(0).tolist(), bounds=[v.min(0).tolist(), v.max(0).tolist()], max_speed_m_s=speed, max_joint_limit_excess=excess)) + '\n')
                record()
                if frame % 50 == 0:
                    print('WALLET', pr['sample'], frame, n, flush=True)
        started = False
        np.savez_compressed(out / 'trajectory.npz', vertices_base=vertices, velocities_base=velocities, faces=faces, video_time_s=times,
                            pad_pose_xyzw=poses, joints=jointstates, joint_names=jointnames, pad_paths=[x for x, _ in pads])
        status.update(status='executed', frames=len(times), physics_steps=step, seconds=time.time() - start,
                      audit=dict(complete=True, finite_state=True, no_post_start_object_state_writes=True, no_object_attachment=True,
                                 critical_property_mutations=mutated, max_particle_speed_m_s=maxspeed, max_joint_limit_excess=maxjoint_excess,
                                 contact_force_available=False, particle_self_collision=elastic))
    except Exception as exc:
        failed = True
        traceback.print_exc()
        status.update(status='failed', error=repr(exc), trace=traceback.format_exc(), seconds=time.time() - start)
    finally:
        write(out / 'execution.json', status)
        app.close()
    if failed:
        raise RuntimeError(status['error'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['rigid', 'wallet'])
    p.add_argument('--package', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--robot', required=True, help='robot.usda of the public G1 + OmniPicker model')
    p.add_argument('--collision', choices=['hull', 'decomposition'], default='hull')
    p.add_argument('--video-dir')
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    (run_rigid if a.mode == 'rigid' else run_wallet)(a)


if __name__ == '__main__':
    main()
