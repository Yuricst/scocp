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

    def solve(self, tspan, x0, stm=False, t_eval=None, args=None, get_ODESolution=False):
        """Solve initial value problem

        Args:
            tspan (tuple): time span
            x0 (np.array): initial state
            stm (bool): whether to solve for state transition matrix
            t_eval (np.array): evaluation times
            args (tuple): additional arguments for the right-hand side function
            get_ODESolution (bool): whether to return an `ODESolution` object
        
        Returns:
            (tuple or ODESolution):
                if `get_ODESolution` is False, return a tuple of times and state with shape `N-by-nx`
                if `get_ODESolution` is True, return an `ODESolution` object
        """
        assert len(x0) == self.nx, f"x0 must be of length {self.nx}, but got {len(x0)}"
        if args is None:
            args = self.args
        if stm is False:
            sol = solve_ivp(self.rhs, tspan, x0, t_eval=t_eval, method=self.method, rtol=self.reltol, atol=self.abstol, args=args)
        else:
            x0_stm = np.concatenate((x0, np.eye(self.nx).flatten()))
            sol = solve_ivp(self.rhs_stm, tspan, x0_stm, t_eval=t_eval, method=self.method, rtol=self.reltol, atol=self.abstol, args=args)
        if get_ODESolution is True:
            return sol
        else:
            return sol.t, sol.y.T
    