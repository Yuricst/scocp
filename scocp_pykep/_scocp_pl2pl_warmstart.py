"""Warm-started pl2pl SCP"""

import copy
import cvxpy as cp
import numpy as np
import pykep as pk

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scocp import (
    get_augmented_lagrangian_penalty,
    ContinuousControlSCOCP,
    rhs_twobody,
    mee2rv,
    rv2mee,
    keplerder_nostm,
)
from ._scocp_pl2pl import scocp_pl2pl


class scocp_pl2pl_warmstart(scocp_pl2pl):
    """Warm-start variant of `scocp_pl2pl`
    
    This class inherits from `scocp_pl2pl` and uses a warm-start strategy to speed up the solution process.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # construct convex program
        self.create_convex_program()
        return
    
    def create_convex_program(self):
        """Solve the convex subproblem"""
        N = self.N
        nx = self.integrator.nx
        nu = self.integrator.nu
        Nseg = N - 1
        
        # reference variables
        self.xbar = cp.Parameter((N, nx), name='reference state')
        self.ubar = cp.Parameter((Nseg, nu), name='reference control')
        self.vbar = cp.Parameter((Nseg, 1), name='reference constraint')
        self.ybar = cp.Parameter((self.ny,), name='reference v-infinity vectors')

        self.pl0_rv = cp.Parameter((6,), name='departure planet state')
        self.plf_rv = cp.Parameter((6,), name='arrival planet state')
        self.pl0_rv_dot = cp.Parameter((6,), name='departure planet state derivative')
        self.plf_rv_dot = cp.Parameter((6,), name='arrival planet state derivative')

        self.weight_param = cp.Parameter((), name='weight parameter')
        self.lmb_dynamics_param = cp.Parameter((Nseg,nx), name='lmb_dynamics parameter')
        self.lmb_eq_param = cp.Parameter((self.ng,), name='lmb_eq parameter')

        # variables
        self.xs      = cp.Variable((N, nx), name='state')
        self.us      = cp.Variable((Nseg, nu), name='control')
        self.vs      = cp.Variable((Nseg, 1), name='Gamma')
        self.ys      = cp.Variable((self.ny,), name='v-infinity vectors')
        self.xis_dyn = cp.Variable((Nseg,nx), name='xi_dyn')     # slack for dynamics
        self.xis     = cp.Variable((self.ng,), name='xi')        # slack for target state
        
        penalty = get_augmented_lagrangian_penalty(
            self.weight_param,
            self.xis_dyn,
            self.lmb_dynamics_param,
            xi=self.xis,
            lmb_eq=self.lmb_eq_param,
        )
        objective_func = self.evaluate_objective(self.xs, self.us, self.vs, self.ys) + penalty
        constraints_objsoc = [cp.SOC(self.vs[i,0], self.us[i,0:3]) for i in range(N-1)]
        constraints_control = [self.vs[i,0] <= 1.0 for i in range(Nseg)]
        
        # constraints on dynamics for state and control
        constraints_dyn = [
            self.xs[i+1,:] == self.Phi_A[i,:,:] @ self.xs[i,:] + self.Phi_B[i,:,0:4] @ self.us[i,:] + self.Phi_B[i,:,4] * self.vs[i,:] + self.Phi_c[i,:] + self.xis_dyn[i,:]
            for i in range(Nseg)
        ]

        if self.uniform_dilation:
            constraints_dilation = [self.us[i,3] == self.us[0,3] for i in range(1,Nseg)]
        else:
            constraints_dilation = []

        # trust region constraints 
        constraints_trustregion = [
            self.xs[i,:] - self.xbar[i,:] <= self.trust_region_radius_x for i in range(N)
        ] + [
            self.xs[i,:] - self.xbar[i,:] >= -self.trust_region_radius_x for i in range(N)
        ]
        if self.trust_region_radius_u is not None:
            constraints_trustregion += [
                self.us[i,:] - self.ubar[i,:] <= self.trust_region_radius_u for i in range(Nseg)
            ] + [
                self.us[i,:] - self.ubar[i,:] >= -self.trust_region_radius_u for i in range(Nseg)
            ]

        # boundary conditions
        constraints_boundary = [
            self.xs[0,0:6] - np.concatenate((np.zeros((3,3)), np.eye(3))) @ self.ys[0:3] \
                - self.pl0_rv - self.pl0_rv_dot * (self.xs[0,7] - self.xbar[0,7]) == self.xis[0:6],
            self.xs[0,6] == self.mass0,
            self.xs[-1,0:6] - np.concatenate((np.zeros((3,3)), np.eye(3))) @ self.ys[3:6] \
                - self.plf_rv - self.plf_rv_dot * (self.xs[-1,7] - self.xbar[-1,7]) == self.xis[6:12]
        ]
        constraints_vinf_mag = [
            cp.SOC(self.vinf_dep, self.ys[0:3]),
            cp.SOC(self.vinf_arr, self.ys[3:6]),
        ]
        
        # constraints on times
        if abs(self.t0_bounds[1] - self.t0_bounds[0]) < 1e-12:
            constraints_t0 = [self.xs[0,7] == self.t0_bounds[0]]
        else:
            constraints_t0 = [self.t0_bounds[0] <= self.xs[0,7],
                              self.xs[0,7] <= self.t0_bounds[1]]
        constraints_tf = [self.tf_bounds[0] <= self.xs[-1,7],
                          self.xs[-1,7] <= self.tf_bounds[1]]
        constraints_s = [self.s_bounds[0] <= self.us[i,3] for i in range(Nseg)] + [self.us[i,3] <= self.s_bounds[1] for i in range(Nseg)]

        self.convex_problem = cp.Problem(
            cp.Minimize(objective_func),
            constraints_objsoc + constraints_dyn + constraints_trustregion +\
            constraints_boundary + constraints_vinf_mag + constraints_control +\
            constraints_t0 + constraints_tf + constraints_s + constraints_dilation)
        return
    
    def solve_convex_problem(self, xbar, ubar, vbar, ybar=None):
        """Solve the convex subproblem
        
        Args:
            xbar (np.array): `(N, self.integrator.nx)` array of reference state history
            ubar (np.array): `(N-1, self.integrator.nu)` array of reference control history
            vbar (np.array): `(N-1, self.integrator.nv)` array of reference constraint history
            ybar (np.array): `(N, self.integrator.ny)` array of reference v-infinity vectors

        Returns:
            (tuple): np.array values of xs, us, gs, xi_dyn, xi_eq, zeta_ineq
        """
        # set parameters
        self.xbar.value = xbar[:,:]
        self.ubar.value = ubar[:,:]
        self.vbar.value = vbar[:,:]
        self.ybar.value = ybar[:]

        self.pl0_rv.value = self.target_initial.target_state(self.xbar[0,7].value)
        self.plf_rv.value = self.target_final.target_state(self.xbar[-1,7].value)
        self.pl0_rv_dot.value = self.target_initial.target_state_derivative(self.xbar[0,7].value)
        self.plf_rv_dot.value = self.target_final.target_state_derivative(self.xbar[-1,7].value)

        self.weight_param.value = self.weight
        self.lmb_dynamics_param.value = self.lmb_dynamics[:,:]
        self.lmb_eq_param.value = self.lmb_eq[:]
        
        # solve convex problem
        self.convex_problem.solve(solver = self.solver, verbose = self.verbose_solver)
        self.cp_status = self.convex_problem.status
        return self.xs.value, self.us.value, self.vs.value, self.ys.value, self.xis_dyn.value, self.xis.value, None
    
