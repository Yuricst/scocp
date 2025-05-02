"""scocp: Sequential Convex Optimization Control Problem"""

# check for dependencies
_hard_dependencies = ("cvxpy", "numpy", "matplotlib")
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

# import functions and classes
from .eoms import *
from ._integrator import ScipyIntegrator
from ._scocp_impulsive import (
    ImpulsiveControlSCOCP,
    FixedTimeImpulsiveRendezvous
)
from ._scvxstar import SCvxStar