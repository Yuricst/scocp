"""Miscellaneous functions"""

from collections.abc import Callable
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


class MovingTarget:
    """Define moving target for rendezvous problem
    
    We assume the target state equality constraint is of the form

    ```
    g(r_N,v_N,t_N) = [ r_N - r_ref(t_N)
                       v_N - v_ref(t_N) ]
    ```

    where r_ref(t_N) and v_ref(t_N) are the position and velocity of the target at time t_N.

    Args:
        eval_state (function): function to evaluate target state
        eval_state_derivative (function): function to evaluate derivative of target state
    """
    def __init__(self, eval_state: Callable, eval_state_derivative: Callable):
        self.eval_state = eval_state
        self.eval_state_derivative = eval_state_derivative
        return
    
    def target_state(self, t: float) -> np.ndarray:
        """Get target state at time t"""
        return self.eval_state(t)
    
    def target_state_jacobian(self, t: float) -> np.ndarray:
        """Get 6-by-7 Jacobian of target state constraint w.r.t. [r_N, v_N, t_N]"""
        # dg = np.zeros((6,7))
        # dg[0:6,0:6] = np.eye(6)
        # dg[:,6] = self.eval_state_derivative(t)
        return self.eval_state_derivative(t)
    