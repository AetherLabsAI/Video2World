"""Rigid or elastic wallet-target validation, usable without hidden episode data."""
from pathlib import Path
import numpy as np

def validate(scene,pkg):
 import trimesh
 pkg=Path(pkg)
 if len(scene.get('objects',[]))!=1 or scene['objects'][0]['name']!='target':raise ValueError('One dynamic target is required')
 e=scene['objects'][0];d=e.get('deformable',{})
 if d and d.get('model')!='closed_particle_shell':raise ValueError('Unsupported wallet deformable model')
 if e.get('dynamics','rigid' if not d else 'elastic') not in ['rigid','elastic']:raise ValueError('Unsupported wallet dynamics')
 if e.get('dynamics')=='rigid' and d:raise ValueError('Rigid wallet cannot declare a particle shell')
 ranges={'particle_radius_m':(.0003,.004),'stretch_stiffness':(1.,100000.),'bend_stiffness':(.001,10000.),'shear_stiffness':(1.,100000.),'damping':(0.,10.),'pressure':(0.,1.)}
 for k,(lo,hi) in (ranges.items() if d else []):
  value=float(d.get(k,float('nan')))
  if not np.isfinite(value) or not lo<=value<=hi:raise ValueError('Invalid shell '+k)
 if not e.get('geometry_npz'):raise ValueError('Full wallet triangle mesh is required')
 with np.load(pkg/e['geometry_npz'],allow_pickle=False) as z:
  v=z['vertices'];counts=z['face_vertex_counts'];ii=z['face_vertex_indices']
 if np.any(counts!=3) or len(v)>5000 or len(counts)>20000:raise ValueError('Triangular surface required, max 5000 vertices / 20000 faces')
 m=trimesh.Trimesh(v,ii.reshape(-1,3),process=False)
 if d and (not m.is_watertight or not m.is_winding_consistent or m.volume<=1e-10):raise ValueError('Elastic target must be a consistently oriented, closed positive-volume shell')
 if not d and (len(v)<4 or (np.ptp(v,axis=0)<=0).any()):raise ValueError('Rigid target must have finite nonzero 3D extents')
 receiver=next((p for p in scene.get('props',[]) if p['name']=='receiver'),None)
 if receiver is not None:
  r=receiver.get('interior',{});center=np.asarray(r.get('center'),float);half=np.asarray(r.get('half_extents_xy'),float)
  if r.get('frame')!='entity_local' or center.shape!=(3,) or half.shape!=(2,) or not np.isfinite(center).all() or not np.isfinite(half).all() or (half<=0).any():raise ValueError('Receiver needs finite local cavity center and half_extents_xy')
  if not np.isfinite([r.get('bottom_z'),r.get('rim_z')]).all() or not r['rim_z']>r['bottom_z']:raise ValueError('Invalid receiver opening/bottom')
  if not receiver.get('geometry_npz'):raise ValueError('Receiver needs actual open triangle mesh geometry')
 return True
