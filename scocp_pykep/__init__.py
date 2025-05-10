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
from ._heyoka_cr3bp import get_heyoka_integrator_cr3bp
from ._heyoka_twobody import get_heyoka_integrator_twobody_logmass, get_heyoka_integrator_twobody_mass
from ._integrator_heyoka import HeyokaIntegrator
from ._scocp_pl2pl import PlanetTarget, scocp_pl2pl_logmass, scocp_pl2pl
