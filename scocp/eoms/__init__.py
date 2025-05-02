"""Equations of motion library"""

from ._cr3bp_scipy import rhs_cr3bp, rhs_cr3bp_with_stm
from ._cr3bp_heyoka import get_heyoka_integrator_cr3bp