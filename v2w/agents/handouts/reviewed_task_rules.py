"""Approved 2026-09-23: same task stages for original and native observations.
No GT endpoint is read. Inputs must already be in one same-execution frame.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from inhouse.rgbd_provider_semantics import score as existing_score, first_run

DEFAULT = dict(bilateral_force_min_n=.05, grasp_hold_s=.15, terminal_hold_s=.25,
 min_lift_m=.05, progress_lift_m=.03, release_open_threshold=.2, release_hold_s=.25,
 settle_s=1., settle_position_range_m=.005, settle_angle_range_rad=.0872665,
 support_height_tolerance_m=.02, min_carry_m=.05)


def ordinary(trace, kind, hand, zone=None, contract=None):
    """Retain existing force-based rigid rules; no rotation required for picking."""
    c={**DEFAULT, **(contract or {}), 'kind': 'initially_held_place' if kind=='place_surface' else 'pick_and_hold', 'hand':hand}
    if zone is not None:c['destination_zone']=np.asarray(zone).tolist()
    z=dict(trace)
    p=z.get('obj_xyz',z.get('fabric_obj_xyz'))
    if p is None:raise ValueError('Missing observed object position')
    p=np.asarray(p);n=len(p)
    if n<3 or p.shape!=(n,3) or not np.isfinite(p).all():raise ValueError('Invalid observed positions')
    z.update(obj_xyz=p,fabric_obj_xyz=p)
    # None is consumed only by np.asarray in existing pick implementation,
    # never used for a rotation calculation. No orientation is fabricated.
    q=z.get('obj_quat',z.get('usd_obj_wxyz',z.get('target_quaternion_wxyz')))
    z.update(obj_quat=q,usd_obj_wxyz=q)
    if kind=='place_surface':
        for k in ['obj_quat','grip','bbox_center_xyz','support_z']:
            if z.get(k) is None or not np.isfinite(z[k]).all():raise ValueError('Missing/invalid '+k)
        if np.any(np.linalg.norm(q,axis=1)<1e-8):raise ValueError('Invalid zero quaternion')
    result=existing_score(z,c)
    result.update(rule_version='inhouse-owner-task/2',evidence_kind='target-specific bilateral physical force')
    return result


def bag_insert(center, forces, grip, mouth, half_xy, bottom, *, yaw_rad=0., dt=.05, held_witness=None, released_witness=None):
    """Body-center through mouth, release inside, terminal retention.
    This is explicitly a region proxy, NOT proof of full mesh containment.
    Original adapter uses the explicitly recorded world-axis zone; no PCA yaw inference.
    Agent adapter uses the authored receiver orientation and validates cavity.
    """
    no_forces=forces is None
    if no_forces and (held_witness is None or released_witness is None):raise ValueError('Missing force or geometric holding/release witnesses')
    # Empty force input is omitted from the finite-value validation below.
    center,forces,grip,mouth,half_xy,bottom=map(lambda x:np.asarray(x,float),(center,forces,grip,mouth,half_xy,bottom))
    n=len(center)
    if n<3 or center.shape!=(n,3) or (not no_forces and forces.shape!=(n,2)) or grip.shape!=(n,) or mouth.shape!=(n,3) or half_xy.shape!=(n,2) or bottom.shape!=(n,):raise ValueError('Incomplete bag observations')
    if not all(np.isfinite(x).all() for x in [center,grip,mouth,half_xy,bottom]+([] if no_forces else [forces])) or (half_xy<=0).any():raise ValueError('Nonfinite/invalid bag observations')
    if (grip<0).any() or (grip>1).any() or (not no_forces and (forces<0).any()) or dt<=0:raise ValueError('Invalid grip, forces or clock')
    rel=center-mouth;co,si=np.cos(yaw_rad),np.sin(yaw_rad)
    localxy=rel[:,:2]@np.array([[co,-si],[si,co]])
    inxy=(np.abs(localxy)<=half_xy).all(1)
    inside=inxy&(rel[:,2]<0)&(center[:,2]>=bottom)
    held=(forces>.05).all(1) if held_witness is None else np.asarray(held_witness,bool)
    opened=(grip<=.2)&(forces<=.05).all(1) if released_witness is None else np.asarray(released_witness,bool)
    if held.shape!=(n,) or opened.shape!=(n,):raise ValueError('Invalid holding/release witnesses')
    steps=lambda s:max(1,int(np.ceil(s/dt-1e-9)))
    carry=first_run(held&(np.linalg.norm(center-center[0],axis=1)>=.05),steps(.15))
    crosses=[]
    for i in range(1,n):
        if rel[i-1,2]>0 and rel[i,2]<=0:
            alpha=rel[i-1,2]/(rel[i-1,2]-rel[i,2]);xy=localxy[i-1]*(1-alpha)+localxy[i]*alpha;half=half_xy[i-1]*(1-alpha)+half_xy[i]*alpha
            if (np.abs(xy)<=half).all() and carry is not None and i>=carry:crosses.append(i)
    entry=crosses[0] if crosses else None
    release=first_run(opened&inside,steps(.25),start=entry if entry is not None else n)
    count=steps(1.);terminal_range=float(np.linalg.norm(np.ptp(rel[-count:],axis=0)))
    retained=bool(n>=count and inside[-count:].all() and opened[-count:].all() and terminal_range<=.005)
    success=bool(entry is not None and release is not None and retained)
    stages=[dict(name=k,done=v is not None,frame=v) for k,v in [('carry',carry),('through_mouth',entry),('release_inside',release),('retained',n-1 if success else None)]]
    achieved=0
    for s in stages:
        if not s['done']:break
        achieved+=1
    return dict(task_success=success,progress=dict(progress=achieved/4,achieved=achieved,total=4,stages=stages),task_observations=dict(initial_inside=bool(inside[0]),terminal_inside=bool(inside[-1]),terminal_released=bool(opened[-count:].all()),terminal_relative_range_m=terminal_range,terminal_retained=retained,mouth_crossing_frames=crosses,final_center_relative_m=rel[-1].tolist(),final_half_spans_m=half_xy[-1].tolist(),yaw_assumption_deg=float(np.degrees(yaw_rad))),rule_version='inhouse-owner-task/2',limitation='Object bbox center proxy; not full mesh containment; aperture axes/shape require validated receiver geometry or sensitivity review.')


def pick_with_witness(center, held, *, dt=.05):
    """Same 3/5 cm lift and timing, with an explicitly supplied holding witness."""
    center=np.asarray(center,float);held=np.asarray(held,bool);n=len(center)
    if center.shape!=(n,3) or held.shape!=(n,) or not np.isfinite(center).all():raise ValueError('Invalid hold observations')
    steps=lambda s:max(1,int(np.ceil(s/dt-1e-9)))
    g=first_run(held,steps(.15));dz=center[:,2]-center[0,2]
    lift=first_run(held&(dz>=.03),steps(.15),start=g if g is not None else n)
    done=bool(g is not None and lift is not None and n>=steps(.25) and (held&(dz>=.05))[-steps(.25):].all())
    stages=[dict(name=k,done=v is not None,frame=v) for k,v in [('grasp',g),('lift',lift),('held_high',n-1 if done else None)]]
    achieved=0
    for s in stages:
        if not s['done']:break
        achieved+=1
    return dict(task_success=done,progress=dict(progress=achieved/3,stages=stages),task_observations=dict(final_lift_m=float(dz[-1]),holding_witness='supplied separately; no inferred force'))


def fryer_draft(observations,trajectory,grip,contract,vertices):
    """Existing extraction geometry, with actual contact loss required to release."""
    from inhouse.fryer_task import score,sustained
    from inhouse.episode_contract import contact_forces
    r=score(observations,trajectory,grip,contract,vertices);obs=r['task_observations'];e=obs['first_extracted_frame'];n=len(grip)
    forces=np.linalg.norm(contact_forces(observations,contract['hand']),axis=-1)
    opened=(np.asarray(grip)<=contract['release_open_threshold'])&(forces<=contract['bilateral_force_min_n']).all(1)
    rel=sustained(opened&(np.arange(n)>=(e if e is not None else n)),max(1,int(np.ceil(contract['release_hold_s']/contract['dt']))))
    released=bool(e is not None and len(rel));tail=max(1,int(np.ceil(contract['settle_s']/contract['dt'])))
    success=bool(r['task_success'] and released and opened[-tail:].all());r['task_success']=success;r['progress']['stages']['released']=released;r['progress']['stages']['supported_settled']=success;r['progress']['progress']=sum(r['progress']['stages'].values())/4
    obs.update(first_release_frame=int(rel[0]) if released else None,release_evidence='grip opened and both measured target contact forces <= threshold',terminal_contact_released=bool(opened[-tail:].all()))
    return r


def source_fryer(z,B,vertices,c):
    from inhouse.provided_observation import transform_to_base
    xyz,q=transform_to_base(z['fabric_obj_xyz'],z['target_quaternion_wxyz'],B);rot=Rotation.from_quat(np.roll(q,-1,axis=1)).as_matrix();world=np.einsum('nij,vj->nvi',rot,vertices)+xyz[:,None,:];force=np.asarray(z['pad_force_vectors'])@B[:3,:3];paths=[str(p) for p in z['pad_body_paths']]
    if not all('gripper_r_' in p for p in paths):raise ValueError('Wrong original fryer hand')
    observations=[dict(bounds=[v.min(0).tolist(),v.max(0).tolist()],contacts=[dict(body0='/World/Entities/target',body1=p,impulse=(f/60.).tolist()) for p,f in zip(paths,ff)]) for v,ff in zip(world,force)]
    grip=1-np.clip(z['grip_joint_position']/(np.pi/4),0,1);result=fryer_draft(observations,dict(xyz=xyz,wxyz=q),grip,c,vertices);result['task_observations'].update(final_bottom_z=float(world[-1,:,2].min()),required_support_z=c['support_z'],contact_source='Original measured forces, converted to existing native interface units; no new simulation')
    return result


def bag_source_region(trace):
    """Use the original declared world-XY task region, not unlabeled PCA spans.

    This is a task region, not reconstructed instantaneous soft-bag geometry.
    The original observed bottom remains the lower retention boundary.
    """
    center=np.asarray(trace['bbox_center_xyz']);n=len(center);zone=np.asarray(trace['zone'],float)
    if zone.shape!=(5,) or not np.isfinite(zone).all() or (zone[2:4]<=0).any():raise ValueError('Invalid original bag task region')
    mouth=np.tile([*zone[:2],zone[4]],(n,1));half=np.tile(zone[2:4],(n,1))
    result=bag_insert(center,np.linalg.norm(trace['pad_force_vectors'],axis=-1),trace['grip'],mouth,half,trace['task73_tote_min_z'])
    result['task_observations'].pop('yaw_assumption_deg',None)
    result['task_observations'].update(region_source='Original trace zone: world XY center, half widths and opening height',region_world=zone.tolist(),region_is_deformed_surface=False)
    result['limitation']='Center-through-declared-task-region proxy, not whole-surface containment or per-frame bag deformation reconstruction.'
    return result


def bag_candidate_region(center,receiver,pkg,grip,forces=None,held_witness=None,released_witness=None):
    """Same center-through-mouth predicate in an authored, validated local frame."""
    from inhouse.wallet_score import inside_cavity
    center=np.asarray(center,float);n=len(center)
    inside_cavity(center[:,None,:],receiver,pkg,0.)  # Validate actual walls/open mouth, not a metadata-only box.
    R=Rotation.from_quat(np.roll(receiver['quat_wxyz'],-1)).as_matrix();local=(center-receiver['pos'])@R
    inter=receiver['interior'];mouth=np.tile([*inter['center'][:2],inter['rim_z']],(n,1));half=np.tile(inter['half_extents_xy'],(n,1));bottom=np.full(n,inter['bottom_z'])
    result=bag_insert(local,forces,grip,mouth,half,bottom,held_witness=held_witness,released_witness=released_witness)
    result['task_observations'].pop('yaw_assumption_deg',None)
    result['task_observations']['region_source']='Candidate receiver entity-local frame with physical cavity validation'
    return result
