"""Native task state during execution: success/score history, completion-transition guard and non-rigid state capture.

Runs inside the recorder. ``CompletionTransition`` requires a new object-goal transition when a candidate starts
already complete: native predicates (AND/OR grouping and thresholds) without robot-home/gripper conditions.
"""
import json
import types
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

VERSION='robodojo-completion-transition/1'
ROBOT_CONDITIONS={'all_robot_back_to_origin','is_robot_back_to_origin','is_robot_not_back_to_origin','is_all_gripper_open'}

def object_only(check):
    if isinstance(check,tuple):
        return None if check[0] in ROBOT_CONDITIONS else check
    if isinstance(check,list):
        kept=[v for x in check if (v:=object_only(x)) is not None]
        return kept or None
    raise ValueError('Unsupported native predicate structure')

class CompletionTransition:
    def __init__(self,reward_manager):
        self.rm=reward_manager
        self.groups=[g for group in reward_manager.check_list[0] if (g:=object_only(group)) is not None]
        if not self.groups:
            raise ValueError('Native task has no inspectable object-goal conditions')
        self.initial_goal=self.goal();self.left_goal=not self.initial_goal
        self.reentered=False;self.frames_observed=1
    def goal(self):
        # A check-list group is AND; nested lists use native alternating OR/AND.
        return all(self.rm.check_once(check,0) for group in self.groups for check in group)
    def observe(self):
        current=self.goal();self.frames_observed+=1
        if not current:self.left_goal=True
        elif self.initial_goal and self.left_goal:self.reentered=True
    @property
    def allowed(self):return not self.initial_goal or self.reentered
    def result(self):
        return dict(protocol=VERSION,initial_object_goal=self.initial_goal,left_object_goal=self.left_goal,reentered_object_goal=self.reentered,completion_allowed=self.allowed,frames_observed=self.frames_observed,reason='ok' if self.allowed else 'initial_object_goal_without_exit_reentry',scope='native object predicate conjunction, with robot-home/gripper-only checks removed; initially complete tasks require exit and re-entry')

def array(x):return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)

class TaskState:
    def __init__(self,env,out,embodiment,eval_seed=0):
        import h5py
        self.env=env;self.out=Path(out);self.frames=0;self.history=[];self.datasets={}
        self.file=h5py.File(self.out/'extended_state.h5','w');self.items={};self.coverage=[]
        env.eval_seed=eval_seed
        self.original_step_success=bool(env.success[0])
        # Original xArm wrapper reports closing angle, while native X5 reports
        # opening distance. Preserve the task threshold and normalize OEM endpoints.
        def normalized_open(parser,args):
            for robot,art in zip(env.robot_manager.robot_list,env.robot_manager.robot_key):
                if robot.type!='target':continue
                if not hasattr(robot,'gripper_joint_targets'):
                    val=env.robot_manager.get_end_effector_real_val(robot=robot,env_idx_list=[0])[0]
                    op=(float(np.mean(val))-robot.gripper_scale[0])/(robot.gripper_scale[1]-robot.gripper_scale[0])
                else:
                    opened=robot.gripper_joint_targets(1.);closed=robot.gripper_joint_targets(0.)
                    values=[(float(art.data.joint_pos[0,list(art.joint_names).index(n)])-closed[n])/(v-closed[n]) for n,v in opened.items() if abs(v-closed[n])>1e-8]
                    op=float(np.mean(values))
                if op<float(args['open_threshold']):return 0.
            return 1.
        env.reward_manager.func_parser.is_all_gripper_open=types.MethodType(normalized_open,env.reward_manager.func_parser)
        from v2w.tracks.robodojo.nesting_dolls import bind as bind_doll_sizes
        self.doll_sizes=bind_doll_sizes(env,self.out)
        if self.doll_sizes is not None:
            (self.out/'nesting_dolls_geometry.json').write_text(json.dumps(self.doll_sizes,indent=2)+'\n')
        env.run_reward()
        if hasattr(env,'get_score'):env.get_score()
        self.completion_transition=CompletionTransition(env.reward_manager)
        lm=env.scene_manager.layout_manager;initial={}
        for typ in ['Rigid','Geometry','Dynamic']:
            for item in lm.get_layout_records(0,typ):
                pos,quat=lm.get_instance_pose(0,inst_name=item['inst_name'],relative=False)
                initial[item.get('label') or item['inst_name']]=np.r_[array(pos).reshape(3),array(quat).reshape(4)].tolist()
        (self.out/'completion_initial_state.json').write_text(json.dumps(dict(epoch='pre_action',object_poses_wxyz=initial),indent=2)+'\n')
        self.registered={k:len(getattr(env.reward_manager,k)[0]) for k in ['check_list','query_list','trigger_check_list','trigger_query_list','final_check_list']}
        if not any(self.registered[k] for k in ['check_list','trigger_check_list','final_check_list']):raise ValueError('Task registered no success criterion')
        for typ in ['Rigid','Geometry','Dynamic','Articulation','Garment','Fluid']:
            for r in env.scene_manager.layout_manager.get_layout_records(0,typ):
                obj=env.scene_manager.layout_manager.get_scene_object(0,r['inst_name']);label=r.get('label') or r['inst_name']
                self.items[label]=(typ,obj,r)
                fields=['root_pose_wxyz']
                if typ=='Articulation':fields+=['joint_positions','joint_velocities','link_world_matrices']
                elif typ=='Garment':fields+=['vertices_world']
                elif typ=='Fluid':fields+=['particles_world','particles_local','world_from_instancer']
                self.coverage.append(dict(label=label,type=typ,fields=fields,root_pose_scope='rigid pose' if typ in ['Rigid','Geometry','Dynamic'] else 'root descriptor; detailed state is authoritative'))
        self._register_support()
    def _register_support(self):
        if getattr(self.env,'interact',False) and hasattr(self.env,'query_support_arm_traj'):self.env.query_support_arm_traj(env_idx=0)
    def before_physics(self):
        env=self.env
        if not hasattr(env,'support_arm_action') or not env.support_arm_action[0]:return
        # Native support queues are at the physics/control rate. Drive only the
        # unchanged auxiliary articulation, never task object poses.
        step=env.support_arm_action[0].pop(0)
        for robot,art in zip(env.robot_manager.robot_list[2:],env.robot_manager.robot_key[2:]):
            q=art.data.joint_pos.clone();names=list(art.joint_names)
            for key,state in step.items():
                vals=state['position'];joints=robot.arm_joints_name if '_joint_state' in key and '_ee' not in key else robot.gripper_joints_name
                if len(joints)!=len(vals):raise ValueError('Auxiliary joint contract mismatch: '+key)
                for n,v in zip(joints,vals):q[:,names.index(n)]=float(v)
            art.set_joint_position_target(q)
    def after_action(self):
        env=self.env;self.completion_transition.observe();env.reward_manager.step(env_idx_list=[0]);self._register_support()
        if getattr(env,'interact',False) and hasattr(env,'check_support_arm_stable'):env.check_support_arm_stable(env_idx=0)
        score=env.reward_manager.get_score()[0] if hasattr(env,'get_score') else None
        self.history.append(dict(frame=self.frames,task_success=bool(env.reward_manager.get_reward(final_check=False)[0]) and self.completion_transition.allowed,score=None if score is None else (float(score) if self.completion_transition.allowed else 0.),native_score=None if score is None else float(score),native_task_success=bool(env.reward_manager.get_reward(final_check=False)[0]),environment_valid=bool(env.success[0])))
        if self.doll_sizes is not None:
            self.history[-1]['robots_home']=bool(env.reward_manager.call_func_parser(env.reward_manager.all_robot_back_to_origin(),0))
    def _append(self,key,value):
        v=np.asarray(value)
        if not np.issubdtype(v.dtype,np.number) or not np.isfinite(v).all():raise ValueError('Invalid native state '+key)
        if key not in self.datasets:
            if not v.size:raise ValueError('Empty native state '+key)
            self.datasets[key]=self.file.create_dataset(key,shape=(0,*v.shape),maxshape=(None,*v.shape),chunks=(1,*v.shape),dtype=v.dtype,compression='gzip',compression_opts=1)
        ds=self.datasets[key]
        if ds.shape[1:]!=v.shape:raise ValueError('Topology changed; state stream must be explicitly versioned: '+key)
        ds.resize(len(ds)+1,axis=0);ds[-1]=v
    def capture(self):
        from pxr import UsdGeom
        for label,(typ,obj,r) in self.items.items():
            key=label.replace('/','_')
            if typ=='Garment':self._append(key+'/vertices_world',obj.sample_mesh_vertices()[0])
            elif typ=='Fluid':
                local=np.asarray(obj.get_particle_positions()[0]);T=np.array(UsdGeom.XformCache().GetLocalToWorldTransform(obj.point_instancer.GetPrim())).T
                self._append(key+'/particles_local',local);self._append(key+'/world_from_instancer',T);self._append(key+'/particles_world',local@T[:3,:3].T+T[:3,3])
            elif typ=='Articulation':
                self._append(key+'/joint_positions',array(obj.get_joint_positions()));self._append(key+'/joint_velocities',array(obj.get_joint_velocities()))
                physics=obj._articulation_view._physics_view
                poses=array(physics.get_link_transforms())[0];names=list(physics.link_paths[0])
                if poses.shape!=(len(names),7):raise ValueError('Articulation link path/state mismatch')
                if np.max(np.abs(np.linalg.norm(poses[:,3:],axis=-1)-1))>.002:raise ValueError('Invalid physics link quaternion')
                links=np.tile(np.eye(4),(len(poses),1,1));links[:,:3,3]=poses[:,:3];links[:,:3,:3]=Rotation.from_quat(poses[:,3:]).as_matrix()
                self._append(key+'/link_world_matrices',links);self.file[key].attrs['link_paths']=json.dumps(names)
                self.file[key].attrs['link_pose_source']='PhysX articulation view; xyz+xyzw converted to matrices, single env0 at origin'
        for i,art in enumerate(self.env.robot_manager.robot_key[2:]):
            self._append(f'auxiliary_{i}/joint_positions',array(art.data.joint_pos[0]));self._append(f'auxiliary_{i}/joint_velocities',array(art.data.joint_vel[0]))
            self.file[f'auxiliary_{i}'].attrs['joint_names']=json.dumps(list(art.joint_names))
        self.frames+=1
        if self.frames%25==0:self.file.flush()
    def finish(self):
        env=self.env
        result=dict(status='complete',frames=self.frames,registered=self.registered,task_success=bool(env.reward_manager.get_reward(final_check=True)[0]) and self.completion_transition.allowed,native_task_success=bool(env.reward_manager.get_reward(final_check=True)[0]),completion_transition=self.completion_transition.result(),environment_valid=bool(env.success[0]),history=self.history,coverage=self.coverage,
            gripper_rule='native thresholds with physical opening endpoints; xArm OEM closing-angle reversal corrected',auxiliary_robots=max(0,len(env.robot_manager.robot_key)-2),source='original task.run_reward/get_score and native RewardManager.step/get_reward')
        self.file.attrs['frames']=self.frames;self.file.close()
        (self.out/'task_result.json').write_text(json.dumps(result,indent=2)+'\n');return result

def install_pose_reader(env):
    """Extend missing upstream Dynamic/Fluid pose branches; preserve detailed state."""
    from pxr import UsdGeom
    from scipy.spatial.transform import Rotation
    lm=env.scene_manager.layout_manager;old=lm.get_instance_pose
    def pose(self,env_idx,label=None,inst_name=None,relative=True):
        name=inst_name or self.get_instance_name(env_idx,label);typ=self.instance_type_by_env[env_idx].get(name)
        if typ not in ['dynamic','fluid']:return old(env_idx,label,inst_name,relative)
        obj=self.get_scene_object(env_idx,name);prim=env.stage.GetPrimAtPath(obj.prim_path);T=np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
        R=T[:3,:3];U,_,V=np.linalg.svd(R);q=Rotation.from_matrix(U@V).as_quat();return T[:3,3],np.r_[q[3],q[:3]]
    lm.get_instance_pose=types.MethodType(pose,lm)
