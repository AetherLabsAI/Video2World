"""Render the actual recorded qpos; never run/override object dynamics."""
from pathlib import Path
import json,os
import numpy as np

def render_states(pkg,states,out):
 os.environ.setdefault('MUJOCO_GL','egl')
 import mujoco,imageio.v2 as imageio
 pkg=Path(pkg);m=json.loads((pkg/'protocol.json').read_text());model=mujoco.MjModel.from_xml_path(str(pkg/m['model_path']));data=mujoco.MjData(model);z=np.load(states,allow_pickle=False)
 camera=mujoco.MjvCamera();mujoco.mjv_defaultFreeCamera(model,camera)
 out=Path(out);renderer=mujoco.Renderer(model,height=384,width=512)
 try:
  with imageio.get_writer(out,fps=20,macro_block_size=1) as writer:
   for q in z['qpos']:
    data.qpos[:]=q;mujoco.mj_forward(model,data);renderer.update_scene(data,camera=camera);writer.append_data(renderer.render())
 finally:renderer.close()
 return out
