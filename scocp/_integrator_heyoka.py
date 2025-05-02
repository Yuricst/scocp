"""Integrator class"""

import copy
import numpy as np
from scipy.integrate import solve_ivp


class HeyokaIntegrator:
    def __init__(self, nx, nu, ta, ta_stm, impulsive=True):
        self.nx = nx
        self.nu = nu
        self.ta = ta
        self.ta_stm = ta_stm
        self.impulsive = impulsive
        return
    
    def solve(self, tspan, x0, stm=False, t_eval=None):
        if t_eval is None:
            t_eval = [float(tspan[0]), float(tspan[1])]
        if stm is False:
            self.ta.time = tspan[0]
            self.ta.state[:] = copy.copy(x0)
            out = self.ta.propagate_grid(grid=t_eval)
        else:
            self.ta_stm.time = tspan[0]
            if self.impulsive is True:
                self.ta_stm.state[:] = np.concatenate((x0, np.eye(self.nx).flatten()))
            else:
                self.ta_stm.state[:] = np.concatenate((x0, np.eye(self.nx).flatten(), np.zeros(self.nx*self.nu)))
            out = self.ta_stm.propagate_grid(grid=t_eval)
        return t_eval, out[5]