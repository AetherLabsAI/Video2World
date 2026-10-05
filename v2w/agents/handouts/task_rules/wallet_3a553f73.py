"""Public task semantics for physically executed closed wallet surfaces.

No rigid orientation or material correspondence is assumed. Fingertip evidence
is geometric, not a force measurement. Shape GT is never consulted by this file.
"""
import numpy as np

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

def score(features,contract):
 c=contract;dt=c['dt'];center=np.asarray(features['center']);relative=np.asarray(features['center_in_gripper']);grip=np.asarray(features['grip']);dist=np.asarray(features['pad_distances_m']);n=len(center)
 if n<2 or dist.shape!=(n,2) or relative.shape!=(n,3) or grip.shape!=(n,) or not all(np.isfinite(x).all() for x in [center,relative,grip,dist]):raise ValueError('Invalid task observations')
 near=(dist<=c['proximity_m']).all(1)&(grip>=c['grip_closed_min'])
 held=consecutive(near,c['grasp_hold_s'],dt)&stable(relative,c['grasp_hold_s'],dt,c['hold_center_range_m'])
 events=[];best=0;stages=len(c['stages']);success=False
 if c['kind']=='pick_and_hold':
  bottom=np.asarray(features['bottom_z']);lift=bottom>=bottom[0]+c['min_lift_m'];terminal=consecutive(held&lift,c['terminal_hold_s'],dt)&stable(relative,c['terminal_hold_s'],dt,c['hold_center_range_m'])
  stage=0
  for i in range(n):
   if not near[i]:stage=0
   if stage==0 and held[i]:stage=1
   if stage==1 and held[i] and lift[i]:stage=2
   if stage==2 and terminal[i]:stage=3
   if stage>best:best=stage;events.append(dict(frame=i,stage=c['stages'][stage-1]))
  success=bool(terminal[-1]);progress=best/stages
 elif c['kind']=='initially_held_insert':
  inside=np.asarray(features['inside'],bool);speed=np.asarray(features['surface_speed_m_s']);released=grip<=c['release_open_max'];settled=consecutive(inside&released&(speed<=c['settle_speed_m_s']),c['terminal_hold_s'],dt)&stable(center,c['terminal_hold_s'],dt,c['settle_center_range_m'])
  # Initial held condition must be observed at the beginning, not after picking
  # a dropped object up later; delivery may then proceed through one attempt.
  initial_window=max(1,int(np.ceil(c['grasp_hold_s']/dt)));initial=bool(held[:initial_window+1].any());stage=1 if initial else 0
  if initial:events.append(dict(frame=initial_window-1,stage='initial_hold'));best=1
  ever_inside=False
  for i in range(n):
   carry=np.linalg.norm(center[i]-center[0])>=.03
   if stage==1 and held[i] and carry:stage=2
   if stage==2 and inside[i]:stage=3;ever_inside=True
   if stage==3 and released[i] and inside[i]:stage=4
   if stage==4 and settled[i]:stage=5
   if stage>best:best=stage;events.append(dict(frame=i,stage=c['stages'][stage-1]))
  success=bool(stage>=4 and settled[-1] and ever_inside);progress=best/stages
 else:raise ValueError('Unsupported wallet task')
 if success:progress=1.
 else:progress=min(progress,(stages-1)/stages)
 return dict(task_success=success,progress=dict(progress=float(progress),stages=c['stages'],completed_prefix=best,total_stages=stages),task_observations=dict(stage_events=events,held_frames=np.flatnonzero(held).tolist(),proximity_threshold_m=c['proximity_m'],contact_evidence=c['contact_evidence'],terminal_success_required=True))
