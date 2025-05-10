"""Equations of motion library"""

from ._twobody_scipy import (
    gravity_gradient_twobody,
    rhs_twobody,
    rhs_twobody_stm,
    control_rhs_twobody_logmass_freetf,
    control_rhs_twobody_logmass_freetf_stm,
    control_rhs_twobody_mass_freetf,
    control_rhs_twobody_mass_freetf_stm,
)

from ._cr3bp_scipy import (
    gravity_gradient_cr3bp,
    rhs_cr3bp,
    rhs_cr3bp_stm,
    control_rhs_cr3bp,
    control_rhs_cr3bp_stm,
    control_rhs_cr3bp_logmass,
    control_rhs_cr3bp_logmass_stm,
    control_rhs_cr3bp_freetf,
    control_rhs_cr3bp_freetf_stm,
    control_rhs_cr3bp_logmass_freetf,
    control_rhs_cr3bp_logmass_freetf_stm,
)