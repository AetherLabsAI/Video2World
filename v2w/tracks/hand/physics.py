"""Robot execution contract of the hand track: joint limits, collision filters, motion audit and physical validity.

Only robot joint targets are controlled; objects are never attached or repositioned. An execution is physically valid
when the audited motion stays within the limits below; a task success with an invalid execution is not a success.
"""
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from v2w import paths

VERSION = 'hand-contact-physics/1.3'
VALIDITY_VERSION = 'hand-physical-validity/1.1'
LIMITS = dict(max_robot_penetration_m=.006, max_initial_object_shift_m=.015,
              max_self_overlap_diameter_m=.004,
              max_joint_position_excess=.001, max_ik_position_cm=2., max_ik_rotation_deg=20.)


def joint_limits(path):
    result = {}
    for j in ET.parse(path).findall('joint'):
        if j.get('type') == 'fixed': continue
        limit = j.find('limit')
        if limit is None: raise ValueError('missing joint limit: '+j.get('name'))
        values = {key: float(limit.get(key)) for key in ('effort', 'velocity', 'lower', 'upper')}
        if not all(math.isfinite(v) for v in values.values()) or min(values['effort'], values['velocity']) <= 0:
            raise ValueError('invalid joint limits: '+j.get('name'))
        result[j.get('name')] = values
    return result


def collision_exclusions(path):
    """One ignore bit per closed graph neighborhood, exactly excluding distance <=2."""
    tree=ET.parse(path);graph={}
    for j in tree.findall('joint'):
        a=j.find('parent').get('link');b=j.find('child').get('link')
        graph.setdefault(a,set()).add(b);graph.setdefault(b,set()).add(a)
    geometry={l.get('name') for l in tree.findall('link') if l.find('collision') is not None}
    cliques=sorted({tuple(sorted(({k}|v)&geometry)) for k,v in graph.items() if len(({k}|v)&geometry)>1})
    if len(cliques)>32: raise ValueError('adjacency exclusions exceed collision mask capacity')
    masks={k:0 for k in graph}
    for bit,names in enumerate(cliques):
        for name in names:masks[name]|=1<<bit
    return masks


def configure_robot(agent):
    from mani_skill import format_path
    path=Path(format_path(str(agent.urdf_path)));limits=joint_limits(path);records=[]
    for j in agent.robot.active_joints:
        lim=limits[j.name]
        for native in j._objs:
            effort=min(float(native.get_force_limit()),lim['effort'])
            native.set_drive_properties(native.get_stiffness(),native.get_damping(),force_limit=effort,mode='force')
            native.set_max_velocity(lim['velocity'])
            actual_effort=float(native.get_force_limit());actual_velocity=float(native.get_max_velocity())
            if not (0<actual_effort<=lim['effort']*(1+1e-6)) or not np.isclose(actual_velocity,lim['velocity'],rtol=1e-6):
                raise RuntimeError('robot limit readback mismatch: '+j.name)
        records.append(dict(name=j.name,declared=lim,force_limit=actual_effort,max_velocity=actual_velocity,drive_mode='force'))
    return dict(version=VERSION,verified=True,urdf_path=str(path),urdf_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),joints=records,
                self_collision=getattr(agent,'self_collision_contract',None))


def terminal_motion(env):
    """Both free and articulated objects; unknown motion is never quiescent."""
    records={};errors=[]
    for name,obj in env.objs.items():
        try:
            linear=float(np.linalg.norm(obj.linear_velocity[0].detach().cpu().numpy()))
            angular=float(np.linalg.norm(obj.angular_velocity[0].detach().cpu().numpy()))
            joint=float(np.max(np.abs(env.arts[name].get_qvel()[0].cpu().numpy()))) if name in env.arts else 0.
            if not all(math.isfinite(v) for v in (linear,angular,joint)):raise ValueError('nonfinite motion')
            joint_limit=.02 if getattr(env,'joints',{}).get(name,{}).get('type')=='prismatic' else .2
            records[name]=dict(linear_m_s=linear,angular_rad_s=angular,joint_speed=joint,joint_speed_limit=joint_limit)
        except Exception as e:errors.append(dict(object=name,error=str(e)))
    quiet=not errors and all(r['linear_m_s']+.1*r['angular_rad_s']<.02 and r['joint_speed']<r['joint_speed_limit'] for r in records.values())
    return dict(complete=not errors,quiescent=quiet,objects=records,errors=errors,limits=dict(weighted_speed_m_s=.02,angular_weight_m=.1,revolute_rad_s=.2,prismatic_m_s=.02))


def assess(rep):
    reasons=[];a=rep.get('motion_audit') or {};contract=rep.get('robot_contract') or {}
    if rep.get('physics_protocol')!=VERSION:reasons.append('incompatible physics protocol')
    if rep.get('object_attachment') is not False or rep.get('object_state_override') is not False:reasons.append('object transport override')
    if not contract.get('verified'):reasons.append('unverified robot limits')
    if not a.get('finite_state',False):reasons.append('missing or nonfinite motion audit')
    checks=[('max_robot_penetration_m',LIMITS['max_robot_penetration_m'],'robot penetration'),
            ('max_initial_object_shift_m',LIMITS['max_initial_object_shift_m'],'initial object moved during prelude'),
            ('max_joint_position_excess',LIMITS['max_joint_position_excess'],'joint position limit exceeded')]
    for key,limit,reason in checks:
        value=a.get(key)
        if not isinstance(value,(float,int)) or not math.isfinite(value):reasons.append('missing '+key)
        elif value>limit:reasons.append(reason)
    # Speed exceedance is diagnostic only, per the benchmark protocol.
    velocity_ratio=a.get('max_joint_velocity_ratio')
    if not isinstance(velocity_ratio,(float,int)) or not math.isfinite(velocity_ratio):
        reasons.append('missing max_joint_velocity_ratio')
    if not (rep.get('terminal_motion') or {}).get('complete'):reasons.append('missing terminal motion')
    expected=(rep.get('T',rep.get('N',0))-1)*rep.get('dt',0)+rep.get('prelude_s',0)+rep.get('settle_s',0)+rep.get('adaptive_wait_s',0)
    if not math.isfinite(a.get('duration_s',float('nan'))) or abs(a['duration_s']-expected)>1e-5:reasons.append('incomplete physical clock')
    if rep.get('embodiment') in ('dexhand','dexhand_bimanual'):
        if not (contract.get('self_collision') or {}).get('enabled'):reasons.append('self collision disabled')
        if rep.get('embodiment')=='dexhand_bimanual':
            if rep.get('hand_count') != 2 or len(contract.get('joints',[])) != 52: reasons.append('incomplete bimanual robot audit')
            if not (contract.get('self_collision') or {}).get('inter_hand_enabled'): reasons.append('inter-hand collision disabled')
            if not (rep.get('self_collision') or {}).get('includes_inter_hand'): reasons.append('missing inter-hand overlap audit')
        s=rep.get('self_collision');value=(s or {}).get('max_overlap_diameter_m')
        if not isinstance(value,(float,int)) or not math.isfinite(value):reasons.append('missing self collision audit')
        elif value>LIMITS['max_self_overlap_diameter_m']:reasons.append('nonadjacent hand self-intersection')
    else:
        for key,limit in [('ik_pos_err_max_cm',LIMITS['max_ik_position_cm']),('ik_rot_err_max_deg',LIMITS['max_ik_rotation_deg'])]:
            if not math.isfinite(rep.get(key,float('nan'))) or rep[key]>limit:reasons.append('IK infeasible');break
    return dict(version=VALIDITY_VERSION,valid=not reasons,violations=reasons,limits=LIMITS.copy(),
                diagnostics=dict(max_joint_velocity_ratio=velocity_ratio,velocity_gates_success=False))


def gate_success(raw_success,rep):
    validity=assess(rep)
    return bool(raw_success and rep.get('ok') and validity['valid']),validity


class MotionAudit:
    """Read-only robot/scene diagnostics over every physics step, including prelude and settle."""

    def __init__(self, env, sample_every=10, hz=300):
        self.env, self.hz, self.sample_every = env, hz, sample_every
        self.steps=0; self.phase='initial'; self.rows=[]; self.max_penetration=0.0
        self.max_contact_impulse=0.0; self.contact_records=[]; self.max_robot_penetration=0.0; self.worst_robot_contact=None
        self.links=list(env.agent.robot.links)
        self.names=[link.name for link in self.links]
        self.objects=list(env.objs)
        self.velocity_limits=np.array([j._objs[0].get_max_velocity() for j in env.agent.robot.active_joints])
        self.max_velocity_ratio=0.;self.worst_velocity=None
        self.finite_state=True;self.max_position_excess=0.;self.max_initial_shift=0.;self.primary_target=None
        self.position_limits=env.agent.robot.get_qlimits()[0].cpu().numpy()
        self.initial_objects={n:env.obj_pose_np(n).copy() for n in self.objects}
        self.record()

    def record(self):
        u=self.env
        poses=[]
        for link in self.links:
            p=link.pose
            poses.append(np.r_[p.p[0].cpu().numpy(),p.q[0].cpu().numpy()])
        self.finite_state &= all(np.isfinite(p).all() for p in poses) and all(np.isfinite(u.obj_pose_np(n)).all() for n in self.objects)
        self.rows.append(dict(time=self.steps/self.hz,phase=self.phase,
            qpos=u.robot_qpos_np(),qvel=u.agent.robot.get_qvel()[0].cpu().numpy(),
            links=np.array(poses),objects=np.array([u.obj_pose_np(n) for n in self.objects])))

    def step(self):
        self.steps+=1
        qv=self.env.agent.robot.get_qvel()[0].cpu().numpy();qp=self.env.robot_qpos_np()
        self.finite_state &= bool(np.isfinite(qv).all() and np.isfinite(qp).all())
        self.max_position_excess=max(self.max_position_excess,float(np.maximum(self.position_limits[:,0]-qp,qp-self.position_limits[:,1]).max()),0.)
        if self.phase=='prelude' and self.primary_target is not None:
            self.max_initial_shift=max(self.max_initial_shift,float(np.linalg.norm(self.env.obj_pose_np(self.primary_target)[:3]-self.initial_objects[self.primary_target][:3])))
        ratio=np.abs(qv)/self.velocity_limits;j=int(ratio.argmax())
        if ratio[j]>self.max_velocity_ratio:
            self.max_velocity_ratio=float(ratio[j]);self.worst_velocity=dict(time_s=self.steps/self.hz,phase=self.phase,joint=self.env.agent.robot.active_joints[j].name,velocity=float(qv[j]),limit=float(self.velocity_limits[j]))
        # Contact penetration is checked at every physics step, not only video rows.
        contacts=self.env.scene.sub_scenes[0].get_contacts()
        for c in contacts:
            for p in c.points:
                self.max_penetration=max(self.max_penetration,-float(p.separation))
                bodies=[body.entity.name for body in c.bodies]
                if any(name in self.names for name in bodies) and -float(p.separation)>self.max_robot_penetration:
                    self.max_robot_penetration=-float(p.separation)
                    self.worst_robot_contact=dict(time=self.steps/self.hz,phase=self.phase,bodies=bodies,depth_m=self.max_robot_penetration)
                self.max_contact_impulse=max(self.max_contact_impulse,float(np.linalg.norm(p.impulse)))
                if self.steps % self.sample_every == 0:
                    self.contact_records.append(dict(time=self.steps/self.hz,phase=self.phase,
                        body0=c.bodies[0].entity.name,body1=c.bodies[1].entity.name,
                        position=np.asarray(p.position).tolist(),normal=np.asarray(p.normal).tolist(),
                        separation=float(p.separation),impulse=np.asarray(p.impulse).tolist()))
        if self.steps % self.sample_every == 0: self.record()

    def mark(self, phase):
        self.phase=phase
        self.record()

    def save(self, directory):
        path=Path(directory);path.mkdir(parents=True,exist_ok=True)
        rows=self.rows
        np.savez_compressed(path/'motion.npz',time_s=[r['time'] for r in rows],phase=[r['phase'] for r in rows],
            qpos=[r['qpos'] for r in rows],qvel=[r['qvel'] for r in rows],
            link_poses=[r['links'] for r in rows],object_poses=[r['objects'] for r in rows],
            link_names=self.names,object_names=self.objects,joint_names=[j.name for j in self.env.agent.robot.active_joints],joint_velocity_limits=[j._objs[0].get_max_velocity() for j in self.env.agent.robot.active_joints])
        d=dict(finite_state=bool(self.finite_state),max_joint_position_excess=self.max_position_excess,max_initial_object_shift_m=self.max_initial_shift,physics_steps=self.steps,duration_s=self.steps/self.hz,sample_every=self.sample_every,
               max_joint_velocity_ratio=self.max_velocity_ratio,worst_joint_velocity=self.worst_velocity,
               max_contact_penetration_m=self.max_penetration,max_robot_penetration_m=self.max_robot_penetration,worst_robot_contact=self.worst_robot_contact,max_contact_impulse_Ns=self.max_contact_impulse,
               contacts=self.contact_records,includes_prelude=True,includes_settle=True)
        (path/'contacts.json').write_text(json.dumps(d,indent=1)+'\n')
        return {k:v for k,v in d.items() if k!='contacts'}


def self_collision(motion_path, side, stride=1):
    """Sampled convex-link intersection audit (no simulator state changes): the overlap diameter is twice the largest
    inscribed-ball radius of the overlap; nonadjacent links only (pairs within two URDF joints are excluded)."""
    import trimesh
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation
    z=np.load(motion_path);root=paths.asset('hand',f'wuji_{side}_floating')
    urdf=ET.parse(root/f'wuji_{side}_floating.urdf'); names=list(z['link_names']); meshes={};graph={}
    for j in urdf.findall('joint'):
        a=j.find('parent').get('link');b=j.find('child').get('link');graph.setdefault(a,set()).add(b);graph.setdefault(b,set()).add(a)
    for link in urdf.findall('link'):
        name=link.get('name');geometry=link.find('collision/geometry/mesh')
        if geometry is None or name not in names:continue
        mesh=trimesh.load(root/geometry.get('filename'),force='mesh');v=np.asarray(mesh.vertices)
        hull=ConvexHull(v);v=v[hull.vertices];planes=hull.equations
        meshes[name]=(names.index(name),v,planes)
    pairs=[]
    for i,a in enumerate(meshes):
        near={a}|graph.get(a,set())
        for n in list(near):near|=graph.get(n,set())
        pairs.extend((a,b) for b in list(meshes)[i+1:] if b not in near)
    max_d=0.;worst=None;checks=0;cross_max=0.;per_hand={'left':0.,'right':0.}
    if side=='bimanual' and (not any(n.startswith('left_') for n in meshes) or not any(n.startswith('right_') for n in meshes)): raise ValueError('incomplete bimanual link audit')
    for k in range(0,len(z['time_s']),stride):
        transformed={}
        for name,(index,v,planes) in meshes.items():
            pose=z['link_poses'][k,index];r=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix();p=pose[:3]
            w=v@r.T+p;n=planes[:,:3]@r.T;d=planes[:,3]-n@p
            transformed[name]=(w.min(0),w.max(0),np.c_[n,d])
        for a,b in pairs:
            alo,ahi,A=transformed[a];blo,bhi,B=transformed[b]
            lower=np.maximum(alo,blo);upper=np.minimum(ahi,bhi)
            if np.any(upper<=lower):continue
            P=np.vstack([A,B]);result=linprog([0,0,0,-1],A_ub=np.c_[P[:,:3],np.ones(len(P))],b_ub=-P[:,3],bounds=[*zip(lower,upper),(0,None)],method='highs');checks+=1
            if result.success:
                diameter=2*float(result.x[3])
                sa=a.split('_')[0];sb=b.split('_')[0]
                if sa!=sb: cross_max=max(cross_max,diameter)
                elif sa in per_hand: per_hand[sa]=max(per_hand[sa],diameter)
            if result.success and 2*result.x[3]>max_d:
                max_d=2*float(result.x[3]);worst=dict(time_s=float(z['time_s'][k]),phase=str(z['phase'][k]),links=[a,b],overlap_diameter_m=max_d)
    return dict(includes_inter_hand=side=='bimanual',inter_hand_max_overlap_diameter_m=cross_max,per_hand_max_overlap_diameter_m=per_hand,method='convex overlap inscribed-ball diameter',max_overlap_diameter_m=max_d,worst=worst,checked_pairs=checks,sample_stride=stride,sampled_states=len(z['time_s'][::stride]),exclusion='URDF graph distance <= 2',limitation='sampled link poses, convex hulls; not continuous mesh collision certification')
