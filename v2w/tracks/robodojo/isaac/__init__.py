"""Code that runs inside the Isaac Sim interpreter, and how to launch it.

The RoboDojo checkout (upstream commit 25691aa) is a host tool: $V2W_ROBODOJO_SOURCE or ``robodojo_source`` in
v2w.local.json. Modules here import RoboDojo's ``env``/``task``/``utils`` packages from that checkout.
"""
from pathlib import Path

from v2w import paths


def source_root():
    return Path(paths.tool('robodojo_source')).expanduser().resolve()


def environment(runtime_dir=None, **extra):
    """Subprocess environment for the Isaac interpreter."""
    env = paths.env(PYTHONNOUSERSITE='1', OMNI_KIT_ACCEPT_EULA='YES', ACCEPT_EULA='Y', V2W_ROBODOJO_SOURCE=source_root(), **extra)
    libs = paths.tool('graphics_libs', required=False)
    if libs:
        env['LD_LIBRARY_PATH'] = libs
    if runtime_dir is not None:
        env.update({k: str(runtime_dir) for k in ('TMPDIR', 'TMP', 'TEMP')})
    return env


def command(module, *args):
    return [paths.tool('isaac_python'), '-u', '-m', 'v2w.tracks.robodojo.isaac.' + module, *map(str, args)]
