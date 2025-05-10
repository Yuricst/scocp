"""Impulsive rendezvous problems with fixed final time"""

from collections.abc import Callable
import cvxpy as cp
import numpy as np

from ._scocp_impulsive import ImpulsiveControlSCOCP
from .._misc import get_augmented_lagrangian_penalty


class FixedTimeImpulsiveRdv(ImpulsiveControlSCOCP):
    """Fixed-time impulsive rendezvous problem class"""
    def __init__(self, x0, xf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x0 = x0
        self.xf = xf
        return
        
    def evaluate_objective(self, xs, us, vs, ys=None):
        """Evaluate the objective function"""
        return np.sum(vs)
    
    def solve_convex_problem(self, xbar, ubar, vbar, ybar=None):
        """Solve the convex subproblem
        
        Args:
            xbar (np.array): `(N, self.integrator.nx)` array of reference state history
            ubar (np.array): `(N-1, self.integrator.nu)` array of reference control history
            vbar (np.array): `(N-1, self.integrator.nv)` array of reference constraint history
        
        Returns:
            (tuple): np.array values of xs, us, vs, ys, xi_dyn, xi_eq, zeta_ineq
        """
        N,nx = xbar.shape
        _,nu = ubar.shape
        Nseg = N - 1
        
        xs = cp.Variable((N, nx), name='state')
        us = cp.Variable((N, nu), name='control')
        vs = cp.Variable((N, 1), name='Gamma')
        xis = cp.Variable((Nseg,nx), name='xi')         # slack for dynamics
        
        penalty = get_augmented_lagrangian_penalty(self.weight, xis, self.lmb_dynamics)
        objective_func = cp.sum(vs) + penalty
        constraints_objsoc = [cp.SOC(vs[i,0], us[i,:]) for i in range(N)]

        constraints_dyn = [
            xs[i+1,:] == self.Phi_A[i,:,:] @ xs[i,:] + self.Phi_B[i,:,:] @ us[i,:] + self.Phi_c[i,:] + xis[i,:]
            for i in range(Nseg)
        ]

        constraints_trustregion = [
            xs[i,:] - xbar[i,:] <= self.trust_region_radius for i in range(N)
        ] + [
            xs[i,:] - xbar[i,:] >= -self.trust_region_radius for i in range(N)
        ]

        constraints_initial = [xs[0,:] == self.x0]
        constraints_final   = [xs[-1,0:3] == self.xf[0:3], 
                               xs[-1,3:6] + us[-1,:] == self.xf[3:6]]

        convex_problem = cp.Problem(
            cp.Minimize(objective_func),
            constraints_objsoc + constraints_dyn + constraints_trustregion + constraints_initial + constraints_final)
        convex_problem.solve(solver = self.solver, verbose = self.verbose_solver)
        self.cp_status = convex_problem.status
        return xs.value, us.value, vs.value, None, xis.value, None, None
    
