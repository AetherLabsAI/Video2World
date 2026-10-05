"""Package acceptance gates an agent runs on its own package.

    python -m video2sim.cli validate <package>               the package is well formed
    python -m video2sim.cli replay   <package> [--no-render]  the package reproduces in simulation

Exit code 0 means the gate passed.
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='video2sim', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('validate', help='the package is well formed')
    p.add_argument('package')
    p = sub.add_parser('replay', help='the package reproduces in simulation')
    p.add_argument('package')
    p.add_argument('--no-render', action='store_true')
    a = ap.parse_args(argv)
    if a.cmd == 'validate':
        from .protocol.validate import main as run
        return run(['validate', a.package])
    from .protocol.replay import main as run
    return run(['replay', a.package] + (['--no-render'] if a.no_render else []))


if __name__ == '__main__':
    sys.exit(main())
