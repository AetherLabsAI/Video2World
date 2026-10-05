"""Protocol package: the deliverable contract and its two acceptance gates.

    validate(pkg)  -> schema / files / shapes / self-consistency  (pure numpy)
    replay(pkg)    -> rebuild the scene from the package alone and re-simulate
"""
from .validate import validate                                # noqa: F401
