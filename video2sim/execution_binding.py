"""Bind public execution inputs and installed implementation before stepping."""
from pathlib import Path
import hashlib

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def snapshot(pkg):
 pkg=Path(pkg).resolve();base=Path(__file__).parent
 inputs={str(p.relative_to(pkg)):digest(p) for p in sorted(pkg.rglob('*')) if p.is_file() and 'verification' not in p.relative_to(pkg).parts}
 code={str(p.relative_to(base)):digest(p) for p in sorted(base.rglob('*.py')) if '__pycache__' not in p.parts}
 assets={str(p.relative_to(base)):digest(p) for directory in [base/'data/xarm7_gripper_v2',base/'robot_rigs/assets'] for p in sorted(directory.rglob('*')) if p.is_file()}
 return {'input_sha256':inputs,'backend_sha256':code,'backend_assets_sha256':assets}
def verify(pkg,binding):
 base=Path(__file__).parent
 for root,key in [(Path(pkg),'input_sha256'),(base,'backend_sha256')]:
  for name,h in binding[key].items():
   if digest(root/name)!=h:raise RuntimeError('Execution input or implementation changed: '+name)

 for name,h in binding.get('backend_assets_sha256',{}).items():
  if digest(base/name)!=h:raise RuntimeError('Backend robot asset changed: '+name)
