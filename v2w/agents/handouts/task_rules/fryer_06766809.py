"""Task66: grasp, extract the complete basket, release, and settle on support."""
import numpy as np
from scipy.spatial.transform import Rotation
def contact_forces(observations, hand, target='/World/Entities/target', physics_hz=60.):
    """Extract forces on opposing pads of the specified hand and target only."""
    if hand not in ('l', 'r'):
        raise ValueError('Unknown hand')
    forces = np.zeros((len(observations), 2, 3))
    def within(path, root):
        return path == root or path.startswith(root + '/')
    for i, row in enumerate(observations):
        for c in row['contacts']:
            a, b = c['body0'], c['body1']
            if within(a, target): other = b
            elif within(b, target): other = a
            else: continue
            impulse = np.asarray(c['impulse'], float)
            if impulse.shape != (3,) or not np.isfinite(impulse).all():
                raise ValueError('Invalid contact impulse')
            for side, part in enumerate(('inner', 'outer')):
                if any(segment.startswith('gripper_' + hand + '_' + part)
                       for segment in other.split('/')):
                    forces[i, side] += impulse * physics_hz
    return forces


def validate(c):
    if c.get('schema')!='inhouse-fryer-task/1' or c.get('kind')!='pull_out_and_release' or c.get('hand')!='r':
        raise ValueError('Expected right-hand fryer extraction task')
    for k in ['dt','bilateral_force_min_n','grasp_hold_s','release_hold_s','settle_s','settle_position_range_m','settle_angle_range_rad','support_height_tolerance_m','min_contact_pull_m']:
        if not isinstance(c.get(k),(int,float)) or not np.isfinite(c[k]) or c[k]<=0:raise ValueError('Invalid '+k)
    a=np.asarray(c['pull_axis_base'],float);p=np.asarray(c['aperture_point_base'],float)
    if a.shape!=(3,) or p.shape!=(3,) or not np.isfinite(np.r_[a,p]).all() or not np.isclose(np.linalg.norm(a),1):raise ValueError('Invalid aperture frame')
    if not np.isfinite(c['support_z']) or not 0<=c['release_open_threshold']<=1:raise ValueError('Invalid support/release')
    return c

def sustained(mask,n):
    mask=np.asarray(mask,bool)
    return np.flatnonzero(np.convolve(mask.astype(int),np.ones(n,int),'valid')==n)+n-1 if len(mask)>=n else np.array([],int)

def score(observations,trajectory,grip,c,vertices):
    validate(c);xyz=np.asarray(trajectory['xyz'],float);quat=np.asarray(trajectory['wxyz'],float);grip=np.asarray(grip,float);v=np.asarray(vertices,float);N=len(xyz)
    if xyz.shape!=(N,3) or quat.shape!=(N,4) or grip.shape!=(N,) or len(observations)!=N or v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(np.r_[xyz.ravel(),quat.ravel(),grip,v.ravel()]).all():raise ValueError('Invalid physical observations')
    rr=Rotation.from_quat(np.roll(quat,-1,axis=1));world=np.einsum('nij,vj->nvi',rr.as_matrix(),v)+xyz[:,None,:];axis=np.asarray(c['pull_axis_base']);plane=np.asarray(c['aperture_point_base']);clearance=((world-plane)@axis).min(1);bounds=np.asarray([o['bounds'] for o in observations]);bottom=bounds[:,0,2];force=np.linalg.norm(contact_forces(observations,c['hand']),axis=2);held=(force>=c['bilateral_force_min_n']).all(1);released=grip<=c['release_open_threshold'];steps=lambda k:max(1,int(np.ceil(c[k]/c['dt'])));h=sustained(held,steps('grasp_hold_s'));initial_inside=bool(clearance[0]<0);grasp=bool(len(h) and initial_inside);g=int(h[0]) if grasp else N
    cleared=np.flatnonzero((np.arange(N)>=g)&(clearance>=c.get('clearance_tolerance_m',.003)));engaged_pull=bool(grasp and np.any(held&(np.arange(N)>=g)&(((xyz-xyz[min(g,N-1)])@axis)>=c['min_contact_pull_m'])));extract=bool(len(cleared) and engaged_pull);e=int(cleared[0]) if extract else N
    rel=sustained(released&(np.arange(N)>=e),steps('release_hold_s'));release=bool(len(rel) and extract);n=steps('settle_s');tail=xyz[-n:];angle=(rr[-n:]*rr[-1].inv()).magnitude();settled=bool(N>=n and np.linalg.norm(np.ptp(tail,axis=0))<=c['settle_position_range_m'] and angle.max()<=c['settle_angle_range_rad']);supported=bool(np.max(np.abs(bottom[-n:]-c['support_z']))<=c['support_height_tolerance_m']);terminal_clear=bool((clearance[-n:]>=c.get('clearance_tolerance_m',.003)).all());done=bool(release and settled and supported and terminal_clear and released[-n:].all());stages=dict(grasp=grasp,extracted=extract,released=release,supported_settled=done)
    return dict(task_success=done,progress=dict(progress=sum(stages.values())/4,stages=stages),task_observations=dict(initial_inside=initial_inside,contact_driven_pull=engaged_pull,first_grasp_frame=None if not grasp else g,first_extracted_frame=None if not extract else e,first_release_frame=None if not release else int(rel[0]),maximum_clearance_m=float(clearance.max()),terminal_clearance_m=float(clearance[-1]),bilateral_contact_frames=int(held.sum()),terminal_supported=supported,terminal_settled=settled,task='Pull complete tray clear of original housing aperture; release on support. No lift or exact endpoint required.'))
