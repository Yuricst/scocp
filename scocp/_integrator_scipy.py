"""Integrator class"""

from collections.abc import Callable
import numpy as np
from scipy.integrate import solve_ivp

class ScipyIntegrator:
    """Integrator class using `scipy.integrate.solve_ivp`
    
    Args:
        nx (int): dimension of states
        nu (int): dimension of controls
        rhs (function): right-hand side function
        rhs_stm (function): right-hand side function for state transition matrix
        impulsive (bool): whether the dynamics are impulsive
        nv (int): dimensions corresponding to control norms to be augmented 
        method (str): integration method
        reltol (float): relative tolerance
        abstol (float): absolute tolerance
        args (tuple): additional arguments for the right-hand side function
    """
    def __init__(self, nx, nu, rhs: Callable, rhs_stm: Callable, impulsive=True, nv=0, method='RK45', reltol=1e-12, abstol=1e-12, args=None):
        """Initialize the integrator"""
        self.nx = nx
        self.nu = nu
        self.nv = nv
        self.rhs = rhs
        self.rhs_stm = rhs_stm
        self.impulsive = impulsive
        self.method = method
        self.reltol = reltol
        self.abstol = abstol
        self.args = args
        if impulsive is False:
            assert len(self.args[-1]) == self.nu + self.nv,\
                f"last argument must be a place-holder for control with length {self.nu + self.nv}, but got {len(self.args[-1])}"
        return

    def solve(self, tspan, x0, u=None, stm=False, t_eval=None, args=None, get_ODESolution=False):
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
        if (u is not None) and (self.impulsive is False):
            args[-1][:] = u[:]

        if stm is False:
            sol = solve_ivp(self.rhs, tspan, x0, t_eval=t_eval, method=self.method, rtol=self.reltol, atol=self.abstol, args=args)
        else:
            if self.impulsive is True:
                x0_stm = np.concatenate((x0, np.eye(self.nx).flatten()))
            else:
                x0_stm = np.concatenate((x0, np.eye(self.nx).flatten(), np.zeros(self.nx*(self.nu+self.nv))))
            sol = solve_ivp(self.rhs_stm, tspan, x0_stm, t_eval=t_eval, method=self.method, rtol=self.reltol, atol=self.abstol, args=args)
        if get_ODESolution is True:
            return sol
        else:
            return sol.t, sol.y.T
    