"""Execute an agent-owned MuJoCo scene and actuator controls, without reference access."""
from pathlib import Path
import json, time, xml.etree.ElementTree as ET
import numpy as np
import mujoco
import trimesh
from ._twin_geometry import sample_mesh

PROFILE = 'twin_mujoco_v2'


def package_file(pkg, value):
    p = (Path(pkg)/value).resolve()
    if not p.is_relative_to(Path(pkg).resolve()) or not p.is_file():
        raise ValueError('Asset must exist inside candidate package: '+str(value))
    return p


def check_xml(pkg, path, visited=None):
    visited = set() if visited is None else visited
    if path in visited: return
    visited.add(path)
    if len(visited)>128: raise ValueError('Too many MJCF includes')
    root = ET.parse(path).getroot()
    for e in root.iter():
        if e.tag in ('plugin', 'extension'): raise ValueError('External plugins unsupported')
        for k in ('meshdir','texturedir','assetdir'):
            if e.get(k) and (Path(e.get(k)).is_absolute() or '..' in Path(e.get(k)).parts):
                raise ValueError('Asset directories must be package relative')
        if e.get('file'):
            v = Path(e.get('file'))
            if v.is_absolute() or '..' in v.parts: raise ValueError('MJCF file paths must be package relative')
            if e.tag == 'include': check_xml(pkg, package_file(pkg, str(path.parent.relative_to(pkg)/v)), visited)
    # Symlinks anywhere in assets may otherwise escape through compiler directories.
    for p in Path(pkg).rglob('*'):
        if p.is_symlink() and not p.resolve().is_relative_to(Path(pkg).resolve()):
            raise ValueError('External asset symlink unsupported')


def descendants(model, names):
    ids = set()
    for name in names:
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if idx <= 0: raise ValueError('Unknown/nonphysical body: '+name)
        ids.add(idx)
    for i in range(1, model.nbody):
        if int(model.body_parentid[i]) in ids: ids.add(i)
    return ids


def load_package(pkg):
    pkg = Path(pkg).resolve(); manifest = json.loads((pkg/'protocol.json').read_text())
    if manifest.get('physics_profile') != PROFILE: raise ValueError('Expected '+PROFILE+' native scene package')
    if manifest.get('status') != 'success': raise ValueError('Candidate did not deliver a successful build')
    if manifest.get('coordinate_frame') != 'first_camera_opencv':
        raise ValueError('Scene must use first_camera_opencv, metres')
    xml = package_file(pkg, manifest['model_path']); check_xml(pkg, xml)
    model = mujoco.MjModel.from_xml_path(str(xml))
    if not 5e-5 <= model.opt.timestep <= .0002: raise ValueError('Source gripper requires physics timestep 0.00005..0.0002 s')
    if model.nbody>10000 or model.nv>30000 or model.ngeom>20000 or model.nu>128:
        raise ValueError('Candidate exceeds profile resource limits')
    if model.nmocap or model.nplugin:
        raise ValueError('Use contact dynamics: mocap and plugins unsupported')
    roles = manifest['roles']; obj = roles['object']
    bodies = descendants(model, obj.get('bodies', []))
    robot = descendants(model, roles.get('robot', {}).get('bodies', []))
    target = descendants(model, roles.get('target', {}).get('bodies', []))
    # Only the source robot's five mimic joints may be constrained. No welds,
    # object couplings, constant-coordinate equalities, tendons or free joints.
    if not np.all(model.eq_active0):raise ValueError('source mimic constraints must be active')
    if model.neq != 5: raise ValueError('expected five URDF gripper mimic constraints')
    drive=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'drive_joint')
    expected={'left_finger_joint','left_inner_knuckle_joint','right_outer_knuckle_joint','right_finger_joint','right_inner_knuckle_joint'}
    found=set()
    for k in range(model.neq):
        if model.eq_type[k] != mujoco.mjtEq.mjEQ_JOINT:raise ValueError('only robot mimic joint equality allowed')
        a,b=int(model.eq_obj1id[k]),int(model.eq_obj2id[k])
        if a<0 or b!=drive or model.jnt_bodyid[a] not in robot or model.jnt_bodyid[b] not in robot:raise ValueError('equality must stay within robot')
        name=model.joint(a).name
        if name not in expected or name in found or not np.allclose(model.eq_data[k,:5],[0,1,0,0,0]):raise ValueError('invalid source mimic relation')
        found.add(name)
    if found!=expected:raise ValueError('missing source mimic relation')
    if int(model.opt.disableflags) or not np.isclose(np.linalg.norm(model.opt.gravity),9.81,atol=.02):raise ValueError('Earth gravity and enabled physics required')
    flexids = []
    for name in obj.get('flex', []):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, name)
        if i < 0: raise ValueError('Unknown object flex: '+name)
        if not model.flex_rigid[i] and model.flex_edgestiffness[i] <= 0:
            sa = int(model.flex_stiffnessadr[i])
            following = [int(x) for x in model.flex_stiffnessadr[i+1:] if int(x) > sa]
            end = min(following) if following else len(model.flex_stiffness)
            if sa < 0 or not np.any(model.flex_stiffness[sa:end]):
                raise ValueError('Object flex must have physical elasticity/edge stiffness')
        if not (model.flex_contype[i] or model.flex_conaffinity[i]):
            raise ValueError('Object flex must collide')
        flexids.append(i)
        adr, num = model.flex_vertadr[i], model.flex_vertnum[i]
        bodies.update(int(x) for x in model.flex_vertbodyid[adr:adr+num] if x != 0)
    if not bodies: raise ValueError('At least one dynamic manipulated object is required')
    if bodies & robot or bodies & target or target & robot: raise ValueError('Object/robot/target roles must be disjoint')
    if not any(model.body_dofnum[i] for i in bodies): raise ValueError('Manipulated object must be dynamic')
    if not any(model.geom_contype[i] or model.geom_conaffinity[i] for i in range(model.ngeom) if model.geom_bodyid[i] in bodies) and not flexids:
        raise ValueError('Object must have collision geometry')
    for i in range(model.nu):
        if model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT:
            raise ValueError('Only joint actuators on robot bodies supported')
        joint = int(model.actuator_trnid[i,0]); body = int(model.jnt_bodyid[joint])
        if body not in robot: raise ValueError('Actions may actuate only robot/manipulator joints, never object/target')
    act = manifest['actions']
    if act.get('format') != 'actuator_ctrl': raise ValueError('actions.format must be actuator_ctrl')
    commands = np.load(package_file(pkg, act['path']), allow_pickle=False)
    dt = float(act['dt'])
    if commands.ndim != 2 or commands.shape[1] != model.nu or not 1 <= len(commands) <= 20000 or not np.isfinite(commands).all():
        raise ValueError('actions must be a finite (T, model.nu) array')
    if not .001 <= dt <= 1 or len(commands)*dt > 180: raise ValueError('Action duration must be <=180 s; dt .001..1 s')
    if model.nu and (not robot): raise ValueError('Actuated manipulator role missing')
    limited = model.actuator_ctrllimited.astype(bool)
    if np.any(commands[:,limited] < model.actuator_ctrlrange[limited,0]-1e-8) or np.any(commands[:,limited] > model.actuator_ctrlrange[limited,1]+1e-8):
        raise ValueError('Controls exceed declared actuator limits')
    verify_robot(model, robot)
    return manifest, model, commands, dt, bodies, robot, target, flexids


def geom_mesh(model, i):
    kind = model.geom_type[i]; size = model.geom_size[i]
    if kind == mujoco.mjtGeom.mjGEOM_BOX: return trimesh.creation.box(extents=2*size)
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE: return trimesh.creation.icosphere(subdivisions=2, radius=size[0])
    if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        mesh = trimesh.creation.icosphere(subdivisions=2); mesh.apply_scale(size); return mesh
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER: return trimesh.creation.cylinder(radius=size[0], height=2*size[1], sections=32)
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE: return trimesh.creation.capsule(radius=size[0], height=2*size[1], count=[16,16])
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        j = model.geom_dataid[i]; va, vn = model.mesh_vertadr[j], model.mesh_vertnum[j]; fa, fn = model.mesh_faceadr[j], model.mesh_facenum[j]
        return trimesh.Trimesh(vertices=model.mesh_vert[va:va+vn], faces=model.mesh_face[fa:fa+fn], process=False)
    raise ValueError('Object/target geometry type unsupported: '+str(kind))


class Geometry:
    def __init__(self, model, bodies, flexids=(), n=4000):
        self.model=model; self.bodies=sorted(bodies); self.flexids=list(flexids); self.n=n
        self.local=[]; self.flex=[]
        for i in range(model.ngeom):
            if model.geom_bodyid[i] in bodies:
                mesh=geom_mesh(model,i); self.local.append((i, sample_mesh(mesh,n,seed=i),float(mesh.area)))
        for f in flexids:
            dim=int(model.flex_dim[f]); adr=int(model.flex_elemadr[f]); num=int(model.flex_elemnum[f]); va=int(model.flex_vertadr[f]); vn=int(model.flex_vertnum[f])
            elements=model.flex_elem[adr:adr+num*(dim+1)].reshape(-1,dim+1)
            faces=[]
            if dim==2: faces=elements
            elif dim==3:
                from collections import Counter
                counts=Counter(tuple(sorted(face)) for tet in elements for face in [tet[[0,1,2]],tet[[0,1,3]],tet[[0,2,3]],tet[[1,2,3]]])
                faces=[face for face,count in counts.items() if count==1]
            self.flex.append((f,va,vn,np.asarray(faces,int).reshape(-1,3)))
        if not self.local and not self.flex: raise ValueError('Role contains no geometry')
    def points(self, data):
        clouds=[]; weights=[]
        for i,p,area in self.local:
            clouds.append(p @ data.geom_xmat[i].reshape(3,3).T+data.geom_xpos[i]); weights.append(area)
        for f,va,vn,faces in self.flex:
            vertices=data.flexvert_xpos[va:va+vn]
            if len(faces):
                mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
                clouds.append(sample_mesh(mesh,self.n,seed=f)); weights.append(float(mesh.area))
            else:
                # 1D flex: actual collision radius, sample tube around physical edges.
                ea,en=self.model.flex_edgeadr[f],self.model.flex_edgenum[f]
                edges=self.model.flex_edge[ea:ea+en]; rng=np.random.default_rng(f)
                lengths=np.linalg.norm(vertices[edges[:,1]]-vertices[edges[:,0]],axis=1)
                selected=rng.choice(len(edges),self.n,p=lengths/lengths.sum()); e=edges[selected]
                directions=vertices[e[:,1]]-vertices[e[:,0]]; directions/=np.linalg.norm(directions,axis=1)[:,None]
                radial=rng.normal(size=(self.n,3)); radial-=np.sum(radial*directions,axis=1)[:,None]*directions; radial/=np.linalg.norm(radial,axis=1)[:,None]
                clouds.append(vertices[e[:,0]]+rng.random((self.n,1))*(vertices[e[:,1]]-vertices[e[:,0]])+radial*self.model.flex_radius[f]); weights.append(float(lengths.sum()*2*np.pi*self.model.flex_radius[f]))
        if len(clouds)==1: return clouds[0]
        rng=np.random.default_rng(123); choice=rng.choice(len(clouds),self.n,p=np.asarray(weights)/sum(weights))
        return np.asarray([clouds[c][i] for i,c in enumerate(choice)])
    def graph(self,data):
        nodes=[data.xpos[i].copy() for i in self.bodies]; lookup={b:i for i,b in enumerate(self.bodies)}
        edges=[(lookup[b],lookup[int(self.model.body_parentid[b])]) for b in self.bodies if int(self.model.body_parentid[b]) in lookup]
        for f,va,vn,_ in self.flex:
            offset=len(nodes); nodes.extend(data.flexvert_xpos[va:va+vn].copy()); ea,en=self.model.flex_edgeadr[f],self.model.flex_edgenum[f]
            edges.extend((self.model.flex_edge[ea:ea+en]+offset).tolist())
        return np.asarray(nodes).reshape(-1,3),np.asarray(edges,int).reshape(-1,2)


def rollout(pkg, out):
    """No hidden sample argument: only candidate model and candidate controls are read."""
    manifest, model, commands, dt, bodies, robot, target, flexids = load_package(pkg)
    data=mujoco.MjData(model); mujoco.mj_forward(model,data)
    obj=Geometry(model,bodies,flexids); tgt=Geometry(model,target) if target else None
    initial=obj.points(data); initial_target=tgt.points(data) if tgt else np.empty((0,3))
    points=[]; nodes=[]; graph=None; times=[]; qpos=[]; started=time.monotonic()
    def capture():
        nonlocal graph
        points.append(obj.points(data)); v,graph=obj.graph(data); nodes.append(v); times.append(float(data.time));qpos.append(data.qpos.copy())
    grip_ids=[int(model.jnt_qposadr[model.joint(n).id]) for n in ['drive_joint','left_finger_joint','left_inner_knuckle_joint','right_outer_knuckle_joint','right_finger_joint','right_inner_knuckle_joint']]
    mimic_max=0.
    capture(); next_capture=.05; step=0; total=float(len(commands)*dt)
    while data.time < total-1e-9:
        target_ctrl=commands[min(int((data.time+1e-10)/dt),len(commands)-1)]
        data.ctrl[:7]=target_ctrl[:7]
        data.ctrl[7]+=np.clip(target_ctrl[7]-data.ctrl[7],-2*model.opt.timestep,2*model.opt.timestep)
        mujoco.mj_step(model,data); step+=1
        err=float(np.max(np.abs(data.qpos[grip_ids[1:]]-data.qpos[grip_ids[0]])));mimic_max=max(mimic_max,err)
        if err>.001:raise ValueError('Gripper mimic drift exceeds 0.001 rad under contact; reduce physics timestep to 0.00005 s and mimic solref to 0.0001 1')
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or np.any(data.warning.number):
            raise ValueError('Simulation unstable or MuJoCo emitted a warning')
        if data.time>=next_capture-1e-9 or data.time>=total-1e-9:
            mujoco.mj_forward(model,data);capture();next_capture+=.05
        if step%1000==0 and time.monotonic()-started>1200: raise TimeoutError('Candidate simulation exceeded 1200 s wall-clock budget')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'executed_states.npz',time_s=times,qpos=qpos,object_points=np.asarray(points,dtype=np.float32),nodes=np.asarray(nodes,dtype=np.float32),edges=graph,initial_target=initial_target)
    return {'points':points,'nodes':nodes,'edges':graph,'time_s':times,'initial_target':initial_target,
        'simulator':'mujoco','simulator_version':mujoco.__version__,'physics_steps':step,'seconds':time.monotonic()-started,
        'mimic_error_max_rad':mimic_max,'mimic_checked_every_physics_step':True,'executed_states':str(out/'executed_states.npz')}


def validate(pkg):
    try:
        manifest, model, *_ = load_package(pkg)
        return []
    except (ValueError, KeyError, OSError) as exc:
        return [str(exc)]

def replay(pkg, render=False):
    """Native execution receipt; task scoring remains benchmark-side."""
    import hashlib
    from .execution_binding import snapshot, verify
    binding=snapshot(pkg)
    pkg=Path(pkg);out=pkg/'verification/native_twin'
    inputs={str(p.relative_to(pkg)):hashlib.sha256(p.read_bytes()).hexdigest() for p in pkg.rglob('*') if p.is_file() and 'verification' not in p.relative_to(pkg).parts}
    result=rollout(pkg,out)
    for relative,h in inputs.items():
        if hashlib.sha256((pkg/relative).read_bytes()).hexdigest()!=h:raise RuntimeError('input changed during execution')
    verify(pkg,binding)
    report={k:v for k,v in result.items() if k not in ('points','nodes','edges','time_s','initial_target')}
    report.update(profile=PROFILE,ok=True,execution_completed=True,task_success=None,robot='xarm7_gripper',robot_revision='xarm7-gripper/urdf-v2',input_sha256=inputs,states_sha256=hashlib.sha256(Path(result['executed_states']).read_bytes()).hexdigest(),note='Execution verified; task success is not inferred from successful stepping.')
    report.update(binding)
    report['render_requested']=bool(render)
    report['render_generated']=False
    if render:
        from .native_twin_render import render_states
        video=render_states(pkg,result['executed_states'],out/'render.mp4')
        report.update(render_generated=True,render_path=str(video),render_sha256=hashlib.sha256(video.read_bytes()).hexdigest())
    (out/'execution_receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def verify_robot(model, robot):
    """Compare robot mechanics to the published source-derived template.
    Only root placement can differ; task objects/receivers are independent.
    """
    ref=mujoco.MjModel.from_xml_path(str(Path(__file__).parent/'data/xarm7_gripper_v2/robot.xml'))
    if model.nu!=ref.nu:raise ValueError('source robot actuator count changed')
    expected={ref.body(i).name:i for i in range(1,ref.nbody)}
    actual={model.body(i).name:i for i in robot}
    if set(actual)!=set(expected):raise ValueError('xArm source body topology mismatch')
    for name,r in expected.items():
        i=actual[name]
        if name!='xarm_base' and model.body(int(model.body_parentid[i])).name!=ref.body(int(ref.body_parentid[r])).name:raise ValueError('source robot parent changed')
        fields=['body_mass','body_inertia','body_ipos','body_iquat','body_gravcomp']
        if name!='xarm_base':fields+=['body_pos','body_quat']
        for field in fields:
            if not np.allclose(getattr(model,field)[i],getattr(ref,field)[r],atol=1e-7,rtol=1e-6):raise ValueError('source robot '+field+' changed: '+name)
        gs=[j for j in range(model.ngeom) if model.geom_bodyid[j]==i];rs=[j for j in range(ref.ngeom) if ref.geom_bodyid[j]==r]
        if len(gs)!=len(rs):raise ValueError('source robot geometry count changed')
        for g,h in zip(gs,rs):
            for field in ['geom_type','geom_size','geom_pos','geom_quat','geom_contype','geom_conaffinity','geom_friction']:
                if not np.allclose(getattr(model,field)[g],getattr(ref,field)[h],atol=1e-7,rtol=1e-6):raise ValueError('source robot '+field+' changed')
            if ref.geom_type[h]==mujoco.mjtGeom.mjGEOM_MESH:
                a,b=int(model.geom_dataid[g]),int(ref.geom_dataid[h]);x=model.mesh_vert[model.mesh_vertadr[a]:model.mesh_vertadr[a]+model.mesh_vertnum[a]];y=ref.mesh_vert[ref.mesh_vertadr[b]:ref.mesh_vertadr[b]+ref.mesh_vertnum[b]]
                if x.shape!=y.shape or not np.allclose(x,y,atol=1e-7,rtol=1e-6):raise ValueError('source robot collision mesh changed')
    for r in range(ref.njnt):
        i=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,ref.joint(r).name)
        if i<0:raise ValueError('source robot joint missing')
        for field in ['jnt_type','jnt_pos','jnt_axis','jnt_stiffness','jnt_range','jnt_limited','jnt_actfrclimited','jnt_actfrcrange']:
            if not np.allclose(getattr(model,field)[i],getattr(ref,field)[r],atol=1e-7,rtol=1e-6):raise ValueError('source robot '+field+' changed')
    for r in range(ref.njnt):
        i=model.joint(ref.joint(r).name).id
        a,b=int(model.jnt_dofadr[i]),int(ref.jnt_dofadr[r])
        for field in ['dof_damping','dof_armature','dof_frictionloss']:
            if not np.isclose(getattr(model,field)[a],getattr(ref,field)[b]):raise ValueError('source robot '+field+' changed')
    for k in range(ref.nu):
        if model.joint(int(model.actuator_trnid[k,0])).name!=ref.joint(int(ref.actuator_trnid[k,0])).name:raise ValueError('source robot actuator destination changed')
    if model.nu!=ref.nu:raise ValueError('source robot actuator count changed')
    for field in ['actuator_dyntype','actuator_gaintype','actuator_biastype','actuator_trntype','actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange','actuator_forcelimited','actuator_ctrllimited','actuator_gear']:
        if not np.allclose(getattr(model,field),getattr(ref,field),atol=1e-7,rtol=1e-6):raise ValueError('source robot '+field+' changed')
    def exclusions(m):
        return {tuple(sorted((m.body(int(sig)>>16).name,m.body(int(sig)&65535).name))) for sig in m.exclude_signature}
    if exclusions(model)!=exclusions(ref):raise ValueError('contact exclusions must match the four source linkage adjacency pairs')
