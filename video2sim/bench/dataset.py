"""Benchmark samples on disk.

    <root>/<sample_id>/
        video.mp4              V_i  — the ONLY thing an agent may read
        meta.json              sample id, instruction, bridge episode, scene, category
        hidden/
            trajectory.npz     a_i: action (T,7) deltas, proprio (T,7) EE states
            camera.json        real camera (base-frame OpenCV extrinsics + K), human-calibrated
            phi.json           success predicate over reference object names
        human/
            protocol.json      E_i^H — the human-built simulator (bridge_widowx package)
            report.md

`make_sample` writes video + hidden/trajectory.npz + meta from a Bridge
episode; camera.json, phi.json and human/ are produced by the annotation
tools (they need a person).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import bridge_data

BENCH_ROOT = Path(os.environ.get("V2S_BENCH_ROOT")
                  or Path(__file__).resolve().parents[2] / "bench_data" / "bridge")


def sample_dir(sample_id: str, root: Path = BENCH_ROOT) -> Path:
    return Path(root) / sample_id


def make_sample(bridge_ep: int, sample_id: str, scene: str, category: str,
                root: Path = BENCH_ROOT, instruction: str | None = None,
                fps_out: int = 5) -> Path:
    e = bridge_data.load(bridge_ep)
    d = sample_dir(sample_id, root)
    (d / "hidden").mkdir(parents=True, exist_ok=True)
    bridge_data.write_video(e["frames"], d / "video.mp4", fps=fps_out)
    np.savez(d / "hidden" / "trajectory.npz", action=e["action"], proprio=e["proprio"],
             fps=bridge_data.FPS)
    meta = {"sample_id": sample_id, "bridge_episode": int(bridge_ep),
            "bridge_path": e["path"], "scene": scene, "category": category,
            "instruction": instruction if instruction is not None else e["instruction"],
            "T": int(len(e["action"])), "fps": bridge_data.FPS,
            "video": "video.mp4", "resolution": [int(e["frames"].shape[2]), int(e["frames"].shape[1])]}
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    return d


def load_hidden(d: Path) -> dict:
    d = Path(d)
    z = np.load(d / "hidden" / "trajectory.npz")
    out = {"action": z["action"], "proprio": z["proprio"], "meta": json.loads((d / "meta.json").read_text())}
    cam = d / "hidden" / "camera.json"
    phi = d / "hidden" / "phi.json"
    out["camera"] = json.loads(cam.read_text()) if cam.exists() else None
    out["phi"] = json.loads(phi.read_text()) if phi.exists() else None
    return out


def load_frames(d: Path) -> np.ndarray:
    import imageio.v2 as imageio
    return np.stack([f for f in imageio.get_reader(str(Path(d) / "video.mp4"))])


def list_samples(root: Path = BENCH_ROOT) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if (p / "meta.json").exists())
