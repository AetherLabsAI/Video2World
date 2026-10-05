"""Public rigid insertion predicate: bilateral native force, full cavity, release and settle."""
import numpy as np
from scipy.spatial.transform import Rotation
def consecutive(mask,seconds,dt):
 n=max(1,int(np.ceil(seconds/dt-1e-9)));m=np.asarray(mask,bool);out=np.zeros(len(m),bool);run=0
 for i,value in enumerate(m):
  run=run+1 if value else 0;out[i]=run>=n
 return out

def stable(points,seconds,dt,limit):
 n=max(2,int(np.ceil(seconds/dt-1e-9)));x=np.asarray(points);out=np.zeros(len(x),bool)
 for i in range(n-1,len(x)):
  out[i]=np.linalg.norm(np.ptp(x[i-n+1:i+1],axis=0))<=limit
 return out

def score(xyz,quaternions,forces,grip,inside,c):
 # Same bilateral force thresholds as the existing rigid pick/place contract.
 forces=np.asarray(forces,float);xyz=np.asarray(xyz,float);grip=np.asarray(grip,float);inside=np.asarray(inside,bool);dt=.05;held=consecutive((forces>=c['bilateral_force_min_n']).all(1),c['grasp_hold_s'],dt)
 initial_window=max(1,int(np.ceil(c['grasp_hold_s']/dt)));initial=bool(held[:initial_window+1].any());released=np.asarray(grip)<=c['release_open_threshold'];stationary=stable(xyz,c['settle_s'],dt,c['settle_position_range_m'])
 # Also retain the inherited rotational stability requirement.
 rot=Rotation.from_quat(np.roll(quaternions,-1,axis=1));width=max(2,int(np.ceil(c['settle_s']/dt)));angular=np.zeros(len(xyz),bool)
 for i in range(width-1,len(xyz)):angular[i]=np.max((rot[i-width+1:i+1]*rot[i-width+1].inv()).magnitude())<=c['settle_angle_range_rad']
 # A fully contained start has already completed insertion; motion within the
 # cavity is not evidence of inserting it. An observed exit followed by re-entry
 # is valid, and partially inserted starts remain valid.
 initial_inside=bool(inside[0])
 settled=consecutive(inside&released&stationary&angular,c['terminal_hold_s'],dt);stage=1 if initial else 0;best=stage;events=[];outside_seen=not initial_inside
 for i in range(len(xyz)):
  outside_seen=outside_seen or not bool(inside[i])
  if stage==1 and held[i] and np.linalg.norm(xyz[i]-xyz[0])>=c['min_carry_m'] and outside_seen:stage=2
  if stage==2 and inside[i]:stage=3
  if stage==3 and inside[i] and released[i]:stage=4
  if stage==4 and settled[i]:stage=5
  if stage>best:best=stage;events.append(dict(frame=i,stage=c['progress_stages'][stage-1]))
 success=bool(stage==5 and settled[-1]);progress=1. if success else min(best/5,.8)
 return dict(task_success=success,progress=dict(progress=progress,stages=c['progress_stages'],completed_prefix=best,total_stages=5),task_observations=dict(stage_events=events,initial_hold=initial,initial_inside=initial_inside,degenerate_start=bool(initial_inside and not outside_seen),outside_observed=outside_seen,completion_protocol='inhouse-task-transition/2',held_frames=np.flatnonzero(held).tolist(),inside_frames=np.flatnonzero(inside).tolist(),terminal_settled=bool(settled[-1]),contact_evidence='Audited native forces on both opposing pads; full candidate mesh inside physically open cavity, including floor and rim'))
