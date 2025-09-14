"""Integrator class"""

import copy
import numpy as np

class HeyokaIntegrator:
    """Wrapper around heyoka's Taylor adaptive integrator for SCOCP

    If the taylor adaptive object for STM is generated using `hy.var_ode_sys`, make sure to set `symmetric_storage` to True.
    
    Ref:
    - https://bluescarni.github.io/heyoka.py/notebooks/var_ode_sys.html

    Args:
        nx (int): state dimension
        nu (int): control dimension
        ta (obj): heyoka taylor adaptive object for dynamics
        ta_stm (obj): heyoka taylor adaptive object for state transition matrix
        impulsive (bool): whether the dynamics are impulsive
        nv (int): dimensions corresponding to control norms to be augmented 
        symmetric_storage (bool): whether STM is stored in terms of mindex 
    """
    def __init__(self, nx, nu, ta, ta_stm, impulsive=True, nv=0, symmetric_storage=False):
        self.nx = nx
        self.nu = nu
        self.ta = ta
        self.ta_stm = ta_stm
        self.nv = nv
        self.impulsive = impulsive
        self.symmetric_storage = symmetric_storage
        return
    
    def solve(self, tspan, x0, u=None, stm=False, t_eval=None):
        """Solve initial value problem

        Args:
            tspan (tuple): time span
            x0 (np.array): initial state
            u (np.array): control input
            stm (bool): whether to solve for state transition matrix
            t_eval (np.array): evaluation times
        
        Returns:
            (tuple): times and states
        """
        assert len(x0) == self.nx, f"x0 must be of length {self.nx}, but got {len(x0)}"

        if t_eval is None:
            t_eval = [float(tspan[0]), float(tspan[1])]
        
        if stm is False:
            self.ta.time = tspan[0]
            self.ta.state[:] = copy.copy(x0)
            if (u is not None) and (self.impulsive is False):
                self.ta.pars[-self.nu:] = u[:]
            elif u is None:
                self.ta.pars[-self.nu:] = np.zeros(self.nu)
            out = self.ta.propagate_grid(grid=t_eval)
        else:
            self.ta_stm.time = tspan[0]
            if self.impulsive is True:
                self.ta_stm.state[:] = np.concatenate((x0, np.eye(self.nx).flatten()))
            else:
                if u is not None:
                    self.ta_stm.pars[-len(u):] = u[:]
                if self.symmetric_storage:
                    raise NotImplementedError("Symmetric storage of STM is not implemented yet")
                else:
                    self.ta_stm.state[:] = np.concatenate((x0, np.eye(self.nx).flatten(), np.zeros(self.nx*(self.nu+self.nv))))
            out = self.ta_stm.propagate_grid(grid=t_eval)
        return t_eval, out[5]