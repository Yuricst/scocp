"""pykep-oriented methods for scocp"""

# check for dependencies
_hard_dependencies = ("cvxpy", "heyoka", "numba", "numpy", "matplotlib", "scipy", "pykep")
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

# pykep-related functions
from .scocp_pl2pl import CanonicalScales, PlanetTarget, scocp_pl2pl
