"""Explicit rigid robot scene execution from public package data only.

No sample/hidden argument, recorded reference trajectory, attachment or task oracle.
"""
from pathlib import Path
import json,hashlib
import numpy as np

ROBOTS=('xarm7_pusher','panda','panda_robotiq')

def rollout(pkg,render=False):
 from .bench.protocol import load_manifest,scene_in_base_frame,validate_manifest
 from .robot_ik import make_ik,robot_table
 from .robot_rigs import ms3_panda_env
 import gymnasium as gym, torch
 from scipy.spatial.transform import Rotation
 from .execution_binding import snapshot,verify,digest
 binding=snapshot(pkg)
 pkg=Path(pkg);m=load_manifest(pkg);errors=validate_manifest(pkg,m)
 if errors:raise ValueError('; '.join(errors))
 uid=m.get('robot',{}).get('uid')
 if uid not in ROBOTS:raise ValueError('Unregistered explicit robot: '+str(uid))
 a=np.load(pkg/m['actions']['path']).astype(float)
 if a.ndim!=2 or a.shape[1]!=7 or not np.isfinite(a).all():raise ValueError('Expected finite (T,7) absolute TCP actions')
 dt=float(m['actions']['dt'])
 if dt<=0 or len(a)*dt>180:raise ValueError('Invalid action clock')
 # scene_in_base_frame transforms the entire authored scene into robot-base
 # coordinates. Apply the same transform to its TCP actions.
 bp=np.array(m['robot'].get('base_pose',[0,0,0,1,0,0,0]),float);rb=Rotation.from_quat(bp[[4,5,6,3]]).as_matrix()
 pos=(a[:,:3]-bp[:3])@rb;rot=Rotation.from_euler('xyz',a[:,3:6]).as_matrix();rot=np.einsum('ij,tjk->tik',rb.T,rot);quat=Rotation.from_matrix(rot).as_quat()[:,[3,0,1,2]]
 solve=make_ik([0,0,0,1,0,0,0],uid);seed=None;targets=[];errors=[]
 for p,q in zip(pos,quat):
  seed,pe,re=solve(p,q,seed);targets.append(seed);errors.append((pe,re))
 targets=np.array(targets);scene=scene_in_base_frame(m,pkg)
 if max(x[0] for x in errors)>.05:raise ValueError('TCP IK error exceeds 5 cm; inspect authored robot frame/actions')
 spec=robot_table(uid);g=spec['gripper'];qinit=np.r_[targets[0],.04,.04] if g else targets[0]
 if uid=='panda_robotiq':
  from .robot_rigs.panda_robotiq import gripper_qpos
  qinit=np.r_[targets[0],gripper_qpos(0.,['left_outer_knuckle_joint','right_outer_knuckle_joint','left_inner_knuckle_joint','right_inner_knuckle_joint','left_inner_finger_joint','right_inner_finger_joint'])]
 env=gym.make('V2SRobotScene-v1',obs_mode='rgb' if render else 'state',num_envs=1,sim_backend='physx_cpu',render_backend='gpu' if render else 'none',scene_spec=scene,camera=(m.get('cameras') or [None])[0] if render else None,enable_cameras=render,robot_base_pose=[0,0,0,1,0,0,0],robot_uid=uid)
 ver=pkg/'verification/robot_scene';ver.mkdir(parents=True,exist_ok=True);poses=[];qs=[];times=[];frames=[]
 try:
  env.reset(seed=0,options=dict(qpos=qinit.astype(np.float32),settle=0.));u=env.unwrapped
  actual_uid=u.agent.uid
  if actual_uid!=uid:raise RuntimeError(f'Robot dispatch mismatch: requested {uid}, loaded {actual_uid}')
  def capture(t):
   poses.append({n:u.obj_pose_np(n).tolist() for n in u.objs});qs.append(u.agent.robot.get_qpos().cpu().numpy()[0].copy());times.append(t)
   if render:frames.append(u.get_obs()['sensor_data']['eval_camera']['rgb'][0].cpu().numpy())
  capture(0.)
  hz=float(ms3_panda_env.CONTROL_FREQ);steps=max(1,int(round(dt*hz)))
  if abs(steps/hz-dt)>1e-6:raise ValueError('Action dt must be representable by native robot control clock')
  for k in range(len(a)):
   for j in range(1,steps+1):
    q=targets[k] if k==0 else targets[k-1]+(targets[k]-targets[k-1])*(j/steps)
    cmd=np.r_[q,u.gripper_cmd((.085 if uid=='panda_robotiq' else .08) if a[k,6]>.5 else 0.)] if g else q
    env.step(torch.tensor(cmd[None],dtype=torch.float32))
   capture((k+1)*dt)
  final_velocity={n:float(np.linalg.norm(u.objs[n].linear_velocity.cpu().numpy())) for n in u.objs}
 finally:env.close()
 names=list(poses[0]);trajectory=np.array([[p[n] for n in names] for p in poses]);np.savez_compressed(ver/'executed_states.npz',time_s=times,qpos=qs,object_poses=trajectory,object_names=names)
 if render:
  import imageio.v2 as imageio
  imageio.mimwrite(ver/'render.mp4',frames,fps=1/dt,macro_block_size=1)
 report=dict(profile='bridge_widowx',execution_completed=True,ok=True,requested_robot=uid,robot=actual_uid,robot_urdf=str(spec['urdf']),robot_urdf_sha256=hashlib.sha256(Path(spec['urdf']).read_bytes()).hexdigest(),T=len(a),physics_steps=len(a)*steps*int(ms3_panda_env.SIM_FREQ/hz),ik_max_position_error_m=max(x[0] for x in errors),quiescent=max(final_velocity.values(),default=0)<.005,final_velocity=final_velocity,initial_poses=poses[0],final_poses=poses[-1],task_success=None,attachment=False,note='Native own-action contact execution. Task success is evaluated separately.')
 verify(pkg,binding);report.update(binding);report['states_sha256']=digest(ver/'executed_states.npz')
 (ver/'execution_receipt.json').write_text(json.dumps(report,indent=2)+'\n');return report
