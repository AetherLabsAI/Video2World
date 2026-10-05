"""Package contract of the cloth profile (droid-cloth-camera/1): numpy-only, reads no sample or GT data.

The package gives the table plane, the initial sheet (four corners, grid, material) and fingertip poses with a binary
gripper command, all in the first video camera's OpenCV frame. ``python -m v2w.tracks.cloth.contract PKG`` validates.
"""
import argparse
import json
from zipfile import BadZipFile
from pathlib import Path
import numpy as np

PROFILE = 'droid_cloth'
VERSION = 'droid-cloth-camera/1'

def file_in(pkg, name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('package paths must be relative and stay inside the package')
    root = Path(pkg).resolve(); f = root / p
    if not f.is_file() or not f.resolve().is_relative_to(root):
        raise ValueError('missing or escaping package file: ' + str(name))
    return f

def array(value, shape, name):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(name + ' must be finite with shape ' + str(shape))
    return a

def load(pkg):
    pkg = Path(pkg); m = json.loads(file_in(pkg, 'protocol.json').read_text())
    for k, v in dict(protocol_version='3.0', physics_profile=VERSION,
                     coordinate_frame='first_camera_opencv', status='success').items():
        if m.get(k) != v: raise ValueError(k + ' must be ' + v)
    scene = json.loads(file_in(pkg, m['scene']['path']).read_text())
    if set(scene) != {'table', 'cloth'}: raise ValueError('scene must contain only table and cloth')
    table = scene['table']; cloth = scene['cloth']
    if set(table) != {'point', 'normal'}: raise ValueError('table requires point and upward unit normal')
    if set(cloth) - {'corners','grid','mass','friction','young','poisson','thickness'}:
        raise ValueError('unsupported cloth fields; constraints, XML, scripts and state trajectories are not scene inputs')
    pt = array(table['point'], (3,), 'table.point'); normal = array(table['normal'], (3,), 'table.normal')
    if not np.isclose(np.linalg.norm(normal), 1, atol=1e-4): raise ValueError('table.normal must be a unit vector')
    corners = array(cloth['corners'], (4,3), 'cloth.corners')
    if np.max(np.abs(corners)) > 10 or np.max(np.abs(pt)) > 10: raise ValueError('positions must be within 10 metres of the camera')
    # Four perimeter vertices, either winding, convex and non-degenerate in the table plane.
    edges = np.roll(corners, -1, axis=0) - corners
    turn = np.cross(edges, np.roll(edges, -1, axis=0)) @ normal
    if not (np.all(turn > 1e-6) or np.all(turn < -1e-6)): raise ValueError('corners must form a convex quadrilateral in perimeter order')
    lengths = np.linalg.norm(edges, axis=1)
    if lengths.min() < .02 or lengths.max() > 2: raise ValueError('cloth edges must be between 2 cm and 2 m')
    if np.ptp((corners - pt) @ normal) > .05: raise ValueError('initial sheet must be approximately planar and parallel to the table (5 cm)')
    grid = cloth['grid']
    if not isinstance(grid,list) or len(grid)!=2 or any(type(n) is not int or not 3<=n<=41 for n in grid) or np.prod(grid)>1024:
        raise ValueError('grid: two integers 3..41, at most 1024 vertices')
    for k, lo, hi in [('mass',.001,5),('friction',0,5),('young',1e3,1e7),('poisson',0,.49),('thickness',1e-4,.01)]:
        v = cloth.get(k)
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or not lo<=v<=hi:
            raise ValueError(f'cloth.{k} must be finite in [{lo}, {hi}]')
    actions = m['actions']
    if actions.get('format') != 'tcp_matrix_grip': raise ValueError('actions.format must be tcp_matrix_grip')
    dt = actions.get('dt')
    if isinstance(dt,bool) or not isinstance(dt,(int,float)) or not np.isfinite(dt) or not 1/15<=dt<=1 or abs(dt*15-round(dt*15))>1e-6:
        raise ValueError('actions.dt must be a positive multiple of 1/15 second, at most 1 second')
    with file_in(pkg, actions['path']).open('rb') as stream:
        try:
            with np.load(stream, allow_pickle=False) as z:
                if set(z.files) != {'tcp_pose','gripper_closure'}: raise ValueError('actions.npz requires exactly tcp_pose and gripper_closure')
                tcp = np.array(z['tcp_pose'],dtype=float); grip = np.array(z['gripper_closure'],dtype=float)
        except (BadZipFile, EOFError) as e:
            raise ValueError('actions.npz is corrupt') from e
    if tcp.ndim!=3 or tcp.shape[1:]!=(4,4) or not 2<=len(tcp)<=7200 or grip.shape!=(len(tcp),):
        raise ValueError('tcp_pose must be (T,4,4), gripper_closure (T,), with 2..7200 rows')
    if len(tcp)*dt>120: raise ValueError('action stream exceeds 120 seconds')
    if not np.isfinite(tcp).all() or not np.isfinite(grip).all(): raise ValueError('actions must be finite')
    if not np.all((grip==0)|(grip==1)): raise ValueError('gripper_closure must be 0=open or 1=closed')
    rot=tcp[:,:3,:3]
    if not np.allclose(tcp[:,3,:], [0,0,0,1], atol=1e-6) or not np.allclose(rot.transpose(0,2,1)@rot,np.eye(3),atol=1e-4) or not np.allclose(np.linalg.det(rot),1,atol=1e-4):
        raise ValueError('tcp_pose must contain proper rigid transforms')
    if np.max(np.abs(tcp[:,:3,3]))>10: raise ValueError('TCP positions outside 10 metre camera bound')
    file_in(pkg,'report.md'); file_in(pkg,'source/video.mp4')
    return m, scene, tcp, grip

def errors(pkg):
    try: load(pkg); return []
    except (ValueError, TypeError, KeyError, OSError, OverflowError) as e: return [str(e)]

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('pkg');a=p.parse_args()
    e=errors(a.pkg);print(json.dumps({'valid':not e,'errors':e},indent=2));raise SystemExit(bool(e))
