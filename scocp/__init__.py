"""scocp: Sequential Convex Optimization Control Problem"""

# check for dependencies
_hard_dependencies = ("cvxpy", "heyoka", "numba", "numpy", "matplotlib", "scipy")
_missing_dependencies = []
for _dependency in _hard_dependencies:
    try:
        __import__(_dependency)
    except ImportError as _e:  # pragma: no cover
        _missing_dependencies.append(f"{_dependency}: {_e}")

if _missing_dependencies:  # pragma: no cover
    raise ImportError(
        "Unable to import required dependencies:\n" + "\n".join(_missing_dependencies)
    )
del _hard_dependencies, _dependency, _missing_dependencies

# miscellaneous functions
from ._misc import zoh_control, zoh_controls, MovingTarget, kep2rv, rv2kep
from ._keplerder import keplerder, keplerder_nostm

# functions for integrating dynamics
from .eoms import *
from ._integrator_scipy import ScipyIntegrator
from ._integrator_heyoka import HeyokaIntegrator

# sequentially convexified optimal control problems
from ._scocp_impulsive import (
    ImpulsiveControlSCOCP,
    FixedTimeImpulsiveRdv
)
from ._scocp_continuous import (
    ContinuousControlSCOCP,
    FixedTimeContinuousRdv,
    FixedTimeContinuousRdvLogMass,
    FreeTimeContinuousRdv,
    FreeTimeContinuousRdvLogMass,
    FreeTimeContinuousMovingTargetRdvLogMass,
)

# SCP algorithm
from ._scvxstar import SCvxStar

# pykep-related functions
try:
    from .scocp_pykep import *
except ImportError:
    print(f"WARNING: pykep not found, functions within scocp_pykep will not be available")
    pass