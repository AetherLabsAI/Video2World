"""Public video-only dual-arm submission contracts. Never reads hidden samples."""
import argparse, copy, json, re
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

PROFILES = {'inhouse_provider_native_v2': 'inhouse_camera_native_v1',
            'robodojo_candidate_task_v1': 'robodojo_camera_task_v1'}

def local(pkg, name):
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError('Package member must be relative')
    p = (pkg/name).resolve()
    if not p.is_relative_to(pkg.resolve()) or not p.is_file():
        raise ValueError('Missing or external package member: '+name)
    return p

def transform(value):
    t = np.asarray(value, float)
    if t.shape != (4,4) or not np.isfinite(t).all() or not np.allclose(t[3], [0,0,0,1], atol=1e-6):
        raise ValueError('T_world_camera must be finite rigid 4x4')
    r = t[:3,:3]
    if not np.allclose(r.T@r, np.eye(3), atol=1e-5) or abs(np.linalg.det(r)-1)>1e-5:
        raise ValueError('Camera transform may not scale, shear or reflect')
    return t

def load(pkg, profile, public=None):
    pkg = Path(pkg).resolve(); pr = json.loads((pkg/'protocol.json').read_text())
    if not isinstance(pr, dict): raise ValueError('protocol.json must be an object')
    if pr.get('physics_profile') != PROFILES[profile] or pr.get('frame') != 'camera_opencv':
        raise ValueError('Expected '+PROFILES[profile]+' in camera_opencv metres')
    t = transform(pr.get('T_world_camera'))
    scene = json.loads(local(pkg,pr.get('scene_file')).read_text())
    if not isinstance(scene, dict): raise ValueError('scene.json must be an object')
    objects = scene.get('objects',[])
    props = scene.get('props', [])
    if not isinstance(objects, list) or not isinstance(props, list): raise ValueError('objects and props must be arrays')
    if scene.get('table') is not None and not isinstance(scene['table'], dict): raise ValueError('table must be an object')
    entities = objects+props+([scene['table']] if scene.get('table') else [])
    if any(not isinstance(e, dict) for e in entities): raise ValueError('Scene entities must be objects')
    if any(not isinstance(e.get('name'), str) for e in entities): raise ValueError('Entity names must be strings')
    names = [e['name'] for e in entities]
    if not objects or len(names)!=len(set(names)):
        raise ValueError('Dynamic objects required; entity names must be unique')
    for e in entities:
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',e['name']): raise ValueError('Entity ID must be a USD identifier')
        for key,n in [('size',3),('pos',3),('quat_wxyz',4),('color',3)]:
            x = np.asarray(e.get(key),float)
            if x.shape!=(n,) or not np.isfinite(x).all(): raise ValueError('Invalid '+key)
            if key=='size' and np.min(x)<=0: raise ValueError('Nonpositive size')
            if key=='quat_wxyz' and abs(np.linalg.norm(x)-1)>1e-4: raise ValueError('Quaternion must be normalized')
        mass = float(e.get('mass',.1))
        if not np.isfinite(mass) or mass<=0: raise ValueError('Mass must be finite and positive')
        for key,default in [('static_friction',.6),('dynamic_friction',.6),('restitution',0.)]:
            value=float(e.get(key,default))
            if not np.isfinite(value) or value<0 or (key=='restitution' and value>1):raise ValueError('Invalid material coefficient')
        if e.get('collision') is False: raise ValueError('Submitted entities must collide')
        if e.get('asset_path'): raise ValueError('Submit mesh geometry_npz, not a USD with external composition')
        if 'scale' in e and not np.array_equal(e['scale'],[1,1,1]): raise ValueError('Bake scale into mesh vertices and size')
        if e.get('geometry_npz'):
            with np.load(local(pkg,e['geometry_npz']),allow_pickle=False) as z:
                v=np.asarray(z['vertices']);c=np.asarray(z['face_vertex_counts']);i=np.asarray(z['face_vertex_indices'])
                if v.ndim!=2 or v.shape[1]!=3 or not len(v) or not np.isfinite(v).all(): raise ValueError('Invalid mesh vertices')
                if c.ndim!=1 or i.ndim!=1 or not np.issubdtype(c.dtype,np.integer) or not np.issubdtype(i.dtype,np.integer) or np.any(c<3) or c.sum()!=len(i) or np.any(i<0) or np.any(i>=len(v)): raise ValueError('Invalid mesh topology')
                if not np.allclose(np.ptp(v,axis=0),e['size'],atol=1e-5,rtol=1e-3): raise ValueError('size must describe submitted mesh bounds')
    if public:
        if pr.get('task') != public['task'] or pr.get('embodiment') != public['embodiment']:
            raise ValueError('Task/embodiment mismatch')
        if not set(public['dynamic_objects']).issubset({e['name'] for e in objects}): raise ValueError('Missing dynamic task object')
        if not set(public['required_entities']).issubset(names): raise ValueError('Missing required task receiver')
        if profile=='inhouse_provider_native_v2' and 'receiver' in public['required_entities'] and not any(e['name']=='receiver' for e in scene.get('props',[])):raise ValueError('In-house receiver must be a static prop')
    if profile=='robodojo_candidate_task_v1':
        kind=pr.get('embodiment')
        if kind not in ('x5','xarm7') or not scene.get('table'): raise ValueError('Expected x5/xarm7 and submitted table')
        if not isinstance(pr.get('camera'), dict): raise ValueError('camera must be an object')
        focal=float(pr.get('camera',{}).get('focal_px',0))
        if not np.isfinite(focal) or focal<=0:raise ValueError('Candidate camera focal_px must be positive')
        width=14 if kind=='x5' else 16;grip=[width//2-1,width-1];dt=.04;fmt='dual_joint_position_gripper01'
    else:
        if pr.get('embodiment')!='g1_omnipicker': raise ValueError('Expected g1_omnipicker')
        width=20;grip=[18,19];dt=.05;fmt='arm14_waist2_head2_gripper2'
    a=pr.get('actions',{})
    if not isinstance(a, dict): raise ValueError('actions must be an object')
    if a.get('dt')!=dt or a.get('format')!=fmt: raise ValueError('Incorrect action format or dt')
    q=np.load(local(pkg,a.get('path')),allow_pickle=False)
    if not isinstance(q, np.ndarray): raise ValueError('Actions must be an NPY array')
    if q.ndim!=2 or q.shape[1]!=width or len(q)<2 or not np.isfinite(q).all(): raise ValueError('Invalid action array')
    if np.any(q[:,grip]<0) or np.any(q[:,grip]>1): raise ValueError('Gripper commands must be in [0,1]')
    # Bounds come from the same public native model delivered to the agent.
    asset_name='g1_omnipicker' if profile=='inhouse_provider_native_v2' else pr['embodiment']
    asset=Path(public['robot_asset_directory']) if public and public.get('robot_asset_directory') else Path(__file__).parent/'robot'
    spec=json.loads((asset/'robot_spec.json').read_text())['joints']
    if asset_name=='g1_omnipicker':
        names=[f'idx2{i}_arm_l_joint{i}' for i in range(1,8)]+[f'idx6{i}_arm_r_joint{i}' for i in range(1,8)]+['idx01_body_joint1','idx02_body_joint2','idx11_head_joint1','idx12_head_joint2']
        bounds=[(i,spec[name]) for i,name in enumerate(names)]
    else:
        stride=width//2
        bounds=[(side*stride+i,spec['joint'+str(i+1)]['limits']) for side in (0,1) for i in range(stride-1)]
    for i,limit in bounds:
        if ('lower' in limit and np.any(q[:,i]<float(limit['lower'])-1e-6)) or ('upper' in limit and np.any(q[:,i]>float(limit['upper'])+1e-6)):
            raise ValueError('Joint command outside public native limit at column '+str(i))
    return pr,scene,q,t

def world_scene(scene,t):
    result=copy.deepcopy(scene)
    for e in result['objects']+result.get('props',[])+([result['table']] if result.get('table') else []):
        e['pos']=(t[:3,:3]@np.asarray(e['pos'])+t[:3,3]).tolist()
        r=t[:3,:3]@Rotation.from_quat(np.roll(e['quat_wxyz'],-1)).as_matrix()
        e['quat_wxyz']=np.roll(Rotation.from_matrix(r).as_quat(),1).tolist()
    return result

def errors(profile,pkg,public=None):
    try:load(pkg,profile,public);return []
    except (ValueError,KeyError,TypeError,OSError) as e:return [str(e)]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('package',type=Path);p.add_argument('--profile',choices=PROFILES,required=True);p.add_argument('--public',type=Path);a=p.parse_args()
    issues=errors(a.profile,a.package,json.loads(a.public.read_text()) if a.public else None)
    print(json.dumps({'issues':issues}));raise SystemExit(bool(issues))
