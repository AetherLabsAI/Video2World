"""Where things live: the repository, the data release and the host tools.

Data root: $V2W_DATA, else ``data/`` next to the package. Host tools (simulator interpreters, ffmpeg) come from the
environment or from ``v2w.local.json`` written by ``v2w setup``; nothing falls back to another machine's directory.
Sample files written by the original pipeline may name files by their pre-release location; ``resolve`` maps those
names into the release layout through ``data/paths.json``.
"""
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parent
CONFIG = PACKAGE / 'config'
LOCAL = REPO / 'v2w.local.json'

# tool -> environment variable; unset tools fall back as documented in tool()
TOOLS = {
    'python': 'V2W_PYTHON',           # evaluator interpreter (SAPIEN / ManiSkill / MuJoCo); default: this interpreter
    'isaac_python': 'V2W_ISAAC_PYTHON',   # Isaac Sim interpreter for the RoboDojo and in-house tracks
    'twin_python': 'V2W_TWIN_PYTHON',     # interpreter with the reconstructed-twin MuJoCo runtime; default: python
    'robodojo_source': 'V2W_ROBODOJO_SOURCE',   # RoboDojo checkout (upstream commit 25691aa) for the RoboDojo track
    'graphics_libs': 'V2W_GRAPHICS_LIBS',  # optional extra LD_LIBRARY_PATH for headless Isaac rendering
    'ffmpeg': 'V2W_FFMPEG',
}


def local_config():
    try:
        return json.loads(LOCAL.read_text())
    except FileNotFoundError:
        return {}


def data_root():
    value = os.environ.get('V2W_DATA') or local_config().get('data')
    return Path(value).expanduser().resolve() if value else REPO / 'data'


DATA = data_root()


def tool(name, required=True):
    value = os.environ.get(TOOLS[name]) or local_config().get(name)
    if not value and name in ('python', 'twin_python'):
        value = tool('python') if name == 'twin_python' else sys.executable
    if not value and name == 'ffmpeg':
        try:
            import imageio_ffmpeg
            value = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
    if not value and required:
        raise RuntimeError(f'{name} is not configured: set {TOOLS[name]} or run `v2w setup`')
    return value


def sample(name):
    return DATA / 'samples' / name


def asset(*parts):
    return DATA.joinpath('assets', *parts)


def env(**extra):
    """Environment for evaluator subprocesses: v2w and video2sim are importable from the repository root."""
    path = [str(REPO)]
    out = dict(os.environ, PYTHONPATH=os.pathsep.join(path + [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p]),
               PYTHONDONTWRITEBYTECODE='1', V2W_DATA=str(DATA))
    out.update({k: str(v) for k, v in extra.items()})
    return out


@lru_cache(maxsize=1)
def _legacy():
    try:
        table = json.loads((DATA / 'paths.json').read_text())
    except FileNotFoundError:
        return [], []
    return table['host_prefixes'], sorted(table['map'].items(), key=lambda kv: -len(kv[0]))


def resolve(path):
    """Release location of a file named by a sample record (absolute, release-relative, or pre-release)."""
    text = os.fspath(path)
    p = Path(text)
    if p.is_absolute() and p.exists():
        return p
    hosts, table = _legacy()
    for prefix in hosts:
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip('/')
            break
    if ':' in text and not text.startswith('/'):   # '<root>:<relative>' records
        text = text.split(':', 1)[1].lstrip('/')
    for old, new in table:
        if text == old or text.startswith(old + '/'):
            return DATA / (new + text[len(old):])
    return DATA / text if not Path(text).is_absolute() else Path(text)
