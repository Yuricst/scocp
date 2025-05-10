"""scocp: Sequential Convex Optimization Control Problem"""

__copyright__    = 'Copyright (C) 2025 Yuri Shimane'
__version__      = '0.1.3'
__license__      = 'GPL-3.0 License'
__author__       = 'Yuri Shimane'
__author_email__ = 'yuri.shimane@gatech.edu'
__url__          = 'https://github.com/Yuricst/scocp'


# check for dependencies
_hard_dependencies = ("cvxpy", "numba", "numpy", "matplotlib", "scipy")
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
from ._misc import zoh_control, zoh_controls, get_augmented_lagrangian_penalty, MovingTarget, kep2rv, rv2kep
from ._keplerder import keplerder, keplerder_nostm

# functions for integrating dynamics
from .eoms import *
from ._integrator_scipy import ScipyIntegrator

# sequentially convexified optimal control problems
from .scocp_impulsive import *
from .scocp_continuous import *

# SCP algorithm
from ._scvxstar import SCvxStar
