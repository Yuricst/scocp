"""Miscellaneous functions"""

import cvxpy as cp
import numpy as np


def zoh_control(times, us, t):
    """Zero-order hold control"""
    for i in range(len(times)-1):
        # Binary search to find interval containing t
        idx = np.searchsorted(times, t, side='right') - 1
        if idx >= 0 and idx < len(times)-1:
            return us[idx]
    return us[-1]  # Return last control if t > times[-1]


def zoh_controls(times, us, t_eval):
    """Zero-order hold control"""
    _,nu = us.shape
    us_zoh = np.zeros((len(t_eval),nu))
    for i,t in enumerate(t_eval):
        us_zoh[i,:] = zoh_control(times, us, t)
    return us_zoh


def get_augmented_lagrangian_penalty(weight, xi_dyn, lmb_dyn, xi=None, lmb_eq=None, zeta=None, lmb_ineq=None):
    """Evaluate augmented Lagrangian penalty function
    
    Args:
        weight (float): weight of the penalty function
        xi_dyn (cp.Variable): slack variable for dynamics
        lmb_dyn (cp.Parameter): multiplier for dynamics
        xi (cp.Variable, optional): slack variable for equality constraints
        lmb_eq (cp.Parameter, optional): multiplier for equality constraints
        zeta (cp.Variable, optional): slack variable for inequality constraints
        lmb_ineq (cp.Parameter, optional): multiplier for inequality constraints
    
    Returns:
        (cp.Expression): augmented Lagrangian penalty function
    """
    assert xi_dyn.shape == lmb_dyn.shape, f"xi_dyn.shape = {xi_dyn.shape} must match lmb_dyn.shape = {lmb_dyn.shape}"
    penalty = weight/2 * cp.sum_squares(xi_dyn)
    #+ cp.sum_squares(xi) + cp.sum_squares(zeta))
    for i in range(lmb_dyn.shape[0]):
        penalty += lmb_dyn[i,:] @ xi_dyn[i,:]
    if xi is not None:
        penalty += weight/2 * cp.sum_squares(xi)
        for i in range(len(lmb_eq)):
            penalty += lmb_eq[i] * xi[i]
    if zeta is not None:
        penalty += weight/2 * cp.sum_squares(zeta)
        for i in range(len(lmb_ineq)):
            penalty += lmb_ineq[i] * zeta[i]
    return penalty