"""Read Bridge V2 episodes from the local hdf5 conversion.

Layout (one file per episode):
    /action                 (T, 7)  [dx dy dz droll dpitch dyaw grip]  5 Hz,
                                    deltas of the end-effector target in the
                                    robot BASE frame; grip 1 = open, 0 = closed
    /proprio                (T, 7)  [x y z roll pitch yaw grip] end-effector
                                    pose in the base frame (evaluator-only)
    /observation/image_0    (T,)    JPEG bytes, 256x256
    /language_instruction   ()      bytes
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np

BRIDGE_ROOT = Path(os.environ.get("BRIDGE_ROOT", "bridge_train"))
FPS = 5


def episode_path(ep: int | str) -> Path:
    if isinstance(ep, int):
        return BRIDGE_ROOT / f"episode_{ep:06d}.hdf5"
    p = Path(ep)
    return p if p.suffix == ".hdf5" else BRIDGE_ROOT / f"{p.name}.hdf5"


def load(ep: int | str, frames: bool = True) -> dict:
    import h5py
    from PIL import Image

    p = episode_path(ep)
    with h5py.File(p) as h:
        s = h["language_instruction"][()]
        out = {
            "path": str(p),
            "instruction": s.decode() if isinstance(s, bytes) else str(s),
            "action": np.asarray(h["action"][:], dtype=np.float64),
            "proprio": np.asarray(h["proprio"][:], dtype=np.float64),
        }
        if frames:
            imgs = []
            for buf in h["observation/image_0"][:]:
                imgs.append(np.asarray(Image.open(io.BytesIO(np.asarray(buf).tobytes())).convert("RGB")))
            out["frames"] = np.stack(imgs)
    return out


def write_video(frames: np.ndarray, path: Path, fps: int = FPS) -> None:
    import imageio.v2 as imageio
    imageio.mimwrite(path, [f.astype(np.uint8) for f in frames], fps=fps,
                     codec="libx264", quality=8, macro_block_size=1)
