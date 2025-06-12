"""Ballistic trajectory design problem with fixed boundary conditions"""

from collections.abc import Callable
import cvxpy as cp
import numpy as np

from ._scocp_continuous import ContinuousControlSCOCP
from .._misc import get_augmented_lagrangian_penalty


class FixedTimeBallisticTrajectory(ContinuousControlSCOCP):
    """Fixed-time ballistic trajectory design problem class
    
    Initialize ubar and zbar with zeros of size (N-1,0)
    """
    def __init__(self, x0, xf, umax, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(x0) == self.integrator.nx
        assert len(xf) == self.integrator.nx
        assert self.integrator.nu == 0,\
            "Ballistic trajectory design problem must be initialized with an integrator for continuous dynamics"
        self.x0 = x0
        self.xf = xf
        self.umax = umax
        return
        
    def evaluate_objective(self, xs, us, vs, ys=None):
        """Evaluate the objective function"""
        return 1.0
    
    def solve_convex_problem(self, xbar, ubar, vbar, ybar=None):
        """Solve the convex subproblem
        
        Args:
            xbar (np.array): `(N, self.integrator.nx)` array of reference state history
            ubar (np.array): `(N-1, self.integrator.nu)` array of reference control history
            vbar (np.array): `(N-1, self.integrator.nv)` array of reference constraint history
        
        Returns:
            (tuple): np.array values of xs, us, vs, xi_dyn, xi_eq, zeta_ineq
        """
        N,nx = xbar.shape
        _,nu = ubar.shape
        Nseg = N - 1
        
        xs = cp.Variable((N, nx), name='state')
        xis_dyn = cp.Variable((Nseg,nx), name='xi_dyn')         # slack for dynamics
        
        penalty = get_augmented_lagrangian_penalty(self.weight, xis_dyn, self.lmb_dynamics)
        dts = np.diff(self.times)
        objective_func = 1.0 + penalty
        #constraints_objsoc = [cp.SOC(vs[i,0], us[i,:]) for i in range(N-1)]

        if self.augment_Gamma:
            constraints_dyn = [
                #xs[i+1,:] == self.Phi_A[i,:,:] @ xs[i,:] + self.Phi_B[i,:,:] @ np.concatenate([us[i,:], vs[i,:]]) + self.Phi_c[i,:] + xis_dyn[i,:]
                xs[i+1,:] == self.Phi_A[i,:,:] @ xs[i,:] + self.Phi_c[i,:] + xis_dyn[i,:]
                for i in range(Nseg)
            ]
        else:
            constraints_dyn = [
                xs[i+1,:] == self.Phi_A[i,:,:] @ xs[i,:] + self.Phi_c[i,:] + xis_dyn[i,:]
                for i in range(Nseg)
            ]

        constraints_trustregion = [
            xs[i,:] - xbar[i,:] <= self.trust_region_radius for i in range(N)
        ] + [
            xs[i,:] - xbar[i,:] >= -self.trust_region_radius for i in range(N)
        ]

        constraints_initial = [xs[0,:] == self.x0]
        constraints_final   = [xs[-1,0:3] == self.xf[0:3], 
                               xs[-1,3:6] == self.xf[3:6]]

        convex_problem = cp.Problem(
            cp.Minimize(objective_func),
            constraints_dyn + constraints_trustregion + constraints_initial + constraints_final)
        convex_problem.solve(solver = self.solver, verbose = self.verbose_solver)
        self.cp_status = convex_problem.status
        return (
            xs.value,
            np.zeros((self.N-1,self.integrator.nu)),
            np.zeros((self.N-1,self.integrator.nu)),
            None, xis_dyn.value, None, None
        )
    
