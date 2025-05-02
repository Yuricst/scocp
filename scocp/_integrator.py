"""Integrator class"""

import numpy as np
from scipy.integrate import solve_ivp


class ScipyIntegrator:
    """Integrator class using `scipy.integrate.solve_ivp`
    
    Args:
        nx (int): dimension of states
        rhs (function): right-hand side function
        rhs_stm (function): right-hand side function for state transition matrix
        method (str): integration method
        reltol (float): relative tolerance
        abstol (float): absolute tolerance
        args (tuple): additional arguments for the right-hand side function
    """
    def __init__(self, nx, rhs, rhs_stm, method='RK45', reltol=1e-12, abstol=1e-12, args=None):
        """Initialize the integrator"""
        self.nx = nx
        self.rhs = rhs
        self.rhs_stm = rhs_stm
        self.method = method
        self.reltol = reltol
        self.abstol = abstol
        self.args = args
        return

    def solve(self, tspan, x0, stm=False, t_eval=None, args=None):
        """Solve initial value problem"""
        assert len(x0) == self.nx, f"x0 must be of length {self.nx}, but got {len(x0)}"
        if args is None:
            args = self.args
        if stm is False:
            sol = solve_ivp(self.rhs, tspan, x0, t_eval=t_eval, method=self.method, rtol=self.reltol, atol=self.abstol, args=args)
        else:
            x0_stm = np.concatenate((x0, np.eye(self.nx).flatten()))
            sol = solve_ivp(self.rhs_stm, tspan, x0_stm, t_eval=t_eval, method=self.method, rtol=self.reltol, atol=self.abstol, args=args)
        return sol