"""Ballistic trajectory design problem with fixed boundary conditions"""

from collections.abc import Callable
import cvxpy as cp
import numpy as np

from ._scocp_impulsive import ImpulsiveControlSCOCP
from .._misc import get_augmented_lagrangian_penalty


class IndirectOptimalControl(ImpulsiveControlSCOCP):
    """Fixed-time ballistic trajectory design problem class
    
    Initialize ubar and zbar with zeros of size (N-1,0)

    For a rendez-vous problem, `xf_dict` should look like:
    xf_dict = {
        0: xf[0], 1: xf[1], 2: xf[2], 3: xf[3], 4: xf[4], 5: xf[5],
    }

    Args:
        x0 (np.array): initial state, does not include costate
        xf_dict (dict): dictionary of fixed states, where indices are integers corresponding to fixed states.
    """
    def __init__(self, x0, xf_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(x0) * 2 == self.integrator.nx        # nx is the number of states + costates
        assert self.integrator.nu == 0,\
            "Ballistic trajectory design problem must be initialized with an integrator for continuous dynamics"
        self.B = np.zeros((self.integrator.nx, self.integrator.nu))
        self.x0 = x0
        self.xf_dict = xf_dict
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
        xis_dyn = cp.Variable((Nseg,nx), name='xi_dyn')                         # slack for non-convex dynamics

        constraints_l1_penalty = []
        if self.l1_penalty:
            slack_l1_xi_dyn = cp.Variable((Nseg,nx), name='slack_l1_xi_dyn')    # slack for L1 penalization of non-convex dynamics
            constraints_l1_penalty.append(slack_l1_xi_dyn >= 0.0)               # slack must be non-negative
            constraints_l1_penalty.append(xis_dyn <=  slack_l1_xi_dyn)
            constraints_l1_penalty.append(xis_dyn >= -slack_l1_xi_dyn)
        else:
            slack_l1_xi_dyn = None
        
        penalty = get_augmented_lagrangian_penalty(self.weight, xis_dyn, self.lmb_dynamics, slack_l1_xi_dyn=slack_l1_xi_dyn)
        objective_func = 1.0 + penalty

        if self.augment_Gamma:
            constraints_dyn = [
                xs[i+1,:] == self.Phi_A[i,:,:] @ xs[i,:] + self.Phi_c[i,:] + xis_dyn[i,:]
                for i in range(Nseg)
            ]
        else:
            constraints_dyn = [
                xs[i+1,:] == self.Phi_A[i,:,:] @ xs[i,:] + self.Phi_c[i,:] + xis_dyn[i,:]
                for i in range(Nseg)
            ]

        constraints_trustregion = [
            xs[i,:] - xbar[i,:] <= self.trust_region_radius_x for i in range(N)
        ] + [
            xs[i,:] - xbar[i,:] >= -self.trust_region_radius_x for i in range(N)
        ]

        constraints_initial = [xs[0,:self.integrator.nx//2] == self.x0]     # fix initial state

        constraints_final = []
        for i in range(self.integrator.nx//2):
            if i in self.xf_dict.keys():
                constraints_final.append(xs[-1,i] == self.xf_dict[i])             # constraint on final state
            else:
                constraints_final.append(xs[-1,i+self.integrator.nx//2] == 0.0)   # constraint on final costate

        convex_problem = cp.Problem(
            cp.Minimize(objective_func),
            constraints_dyn + constraints_trustregion + constraints_initial + constraints_final + constraints_l1_penalty)
        convex_problem.solve(solver = self.solver, verbose = self.verbose_solver)
        self.cp_status = convex_problem.status
        return (
            xs.value,
            np.zeros((self.N,self.integrator.nu)),
            np.zeros((self.N,self.integrator.nv)),
            None, xis_dyn.value, None, None
        )
    
