"""video2sim — agent-driven reproduction of manipulation videos in simulation.

An agent watches ONE video and delivers a self-contained *protocol package*
that any machine can replay: actions, camera intrinsics/extrinsics, robot
URDF, scene, expected trajectories and the visual evidence behind every
number.  See PROTOCOL.md for the contract and .claude/skills for the agent
instructions.
"""
__version__ = "0.1.0"

from .protocol.validate import validate                      # noqa: F401
