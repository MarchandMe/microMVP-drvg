"""
MicroMVP Environment Module.

An environment owns sensing and actuation. It reports where the robots are
and applies wheel commands, without knowing anything about control logic.

- Environment: Abstract base class
- SimEnv: In-memory simulation, no hardware required
- RealEnv: Real hardware — ArUco tracking with an adaptive workspace,
  plus motor commands over the Xiao AP
"""

from .base import Environment
from .sim_env import SimConfig, SimEnv

__all__ = [
    "Environment",
    "SimConfig",
    "SimEnv",
    "RealEnv",
]


def __getattr__(name: str):
    if name == "RealEnv":
        from .real_env import RealEnv

        return RealEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
