"""Equations of motion library"""

from ._cr3bp_scipy import (
    gravity_gradient_cr3bp,
    rhs_cr3bp,
    rhs_cr3bp_stm,
    control_rhs_cr3bp,
    control_rhs_cr3bp_stm
)
from ._cr3bp_heyoka import get_heyoka_integrator_cr3bp