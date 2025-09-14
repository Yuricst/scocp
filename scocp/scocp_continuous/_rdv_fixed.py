"""Rendezvous problem with fixed boundary conditions"""

from collections.abc import Callable
import cvxpy as cp
import numpy as np

from ._scocp_continuous import ContinuousControlSCOCP
from .._misc import get_augmented_lagrangian_penalty


class FixedTimeContinuousRdv(ContinuousControlSCOCP):
    """Fixed-time continuous rendezvous problem class"""
    def __init__(self, x0, xf, umax, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(x0) == self.integrator.nx
        assert len(xf) == self.integrator.nx
        self.x0 = x0
        self.xf = xf
        self.umax = umax
        return
        
    def evaluate_objective(self, xs, us, vs, ys=None):
        """Evaluate the objective function"""
        dts = np.diff(self.times)
        return np.sum(vs.T @ dts)
    
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
        us = cp.Variable((Nseg, nu), name='control')
        vs = cp.Variable((Nseg, 1), name='Gamma')
        xis_dyn = cp.Variable((Nseg,nx), name='xi_dyn')         # slack for dynamics
        
        penalty = get_augmented_lagrangian_penalty(self.weight, xis_dyn, self.lmb_dynamics)
        dts = np.diff(self.times)
        objective_func = cp.sum(vs.T @ dts) + penalty
        constraints_objsoc = [cp.SOC(vs[i,0], us[i,:]) for i in range(N-1)]

        if self.augment_Gamma:
            constraints_dyn = [
                xs[i+1,:] == self.Phi_A[i,:,:] @ xs[i,:] + self.Phi_B[i,:,:] @ np.concatenate([us[i,:], vs[i,:]]) + self.Phi_c[i,:] + xis_dyn[i,:]
                for i in range(Nseg)
            ]
        else:
            constraints_dyn = [
                xs[i+1,:] == self.Phi_A[i,:,:] @ xs[i,:] + self.Phi_B[i,:,:] @ us[i,:] + self.Phi_c[i,:] + xis_dyn[i,:]
                for i in range(Nseg)
            ]

        constraints_trustregion = [
            xs[i,:] - xbar[i,:] <=  self.trust_region_radius_x for i in range(N)
        ] + [
            xs[i,:] - xbar[i,:] >= -self.trust_region_radius_x for i in range(N)
        ]
        if self.trust_region_radius_u is not None:
            constraints_trustregion += [
                us[i,:] - ubar[i,:] <=  self.trust_region_radius_u for i in range(Nseg)
            ] + [
                us[i,:] - ubar[i,:] >= -self.trust_region_radius_u for i in range(Nseg)
            ]

        constraints_initial = [xs[0,:] == self.x0]
        constraints_final   = [xs[-1,0:3] == self.xf[0:3], 
                               xs[-1,3:6] == self.xf[3:6]]
        
        constraints_control = [
            vs[i,0] <= self.umax for i in range(Nseg)
        ]

        convex_problem = cp.Problem(
            cp.Minimize(objective_func),
            constraints_objsoc + constraints_dyn + constraints_trustregion + constraints_initial + constraints_final + constraints_control)
        convex_problem.solve(solver = self.solver, verbose = self.verbose_solver)
        self.cp_status = convex_problem.status
        return xs.value, us.value, vs.value, None, xis_dyn.value, None, None
    

class FixedTimeContinuousRdvLogMass(ContinuousControlSCOCP):
    """Fixed-time continuous rendezvous problem class with log-mass dynamics"""
    def __init__(self, x0, xf, Tmax, N, *args, **kwargs):
        assert len(x0) == 7
        assert len(xf) >= 6
        super().__init__(nh = N - 1, *args, **kwargs)
        self.x0 = x0
        self.xf = xf
        self.Tmax = Tmax
        return
        
    def evaluate_objective(self, xs, us, vs, ys=None):
        """Evaluate the objective function"""
        return -xs[-1,6]
    
    def solve_convex_problem(self, xbar, ubar, vbar, ybar=None):
        """Solve the convex subproblem
        
        Args:
            xbar (np.array): `(N, self.integrator.nx)` array of reference state history
            ubar (np.array): `(N-1, self.integrator.nu)` array of reference control history
            vbar (np.array): `(N-1, self.integrator.nv)` array of reference constraint history
        
        Returns:
            (tuple): np.array values of xs, us, gs, xi_dyn, xi_eq, zeta_ineq
        """
        N,nx = xbar.shape
        _,nu = ubar.shape
        Nseg = N - 1
        
        xs = cp.Variable((N, nx), name='state')
        us = cp.Variable((Nseg, nu), name='control')
        vs = cp.Variable((Nseg, 1), name='Gamma')
        xis_dyn = cp.Variable((Nseg,nx), name='xi_dyn')         # slack for dynamics
        zetas = cp.Variable((Nseg,), name='zeta')     # slack for non-convex inequality
        
        penalty = get_augmented_lagrangian_penalty(self.weight, xis_dyn, self.lmb_dynamics, zeta=zetas, lmb_ineq=self.lmb_ineq)
        objective_func = cp.sum(vs) + penalty
        constraints_objsoc = [cp.SOC(vs[i,0], us[i,:]) for i in range(N-1)]

        constraints_dyn = [
            xs[i+1,:] == self.Phi_A[i,:,:] @ xs[i,:] + self.Phi_B[i,:,0:3] @ us[i,:] + self.Phi_B[i,:,3] * vs[i,:] + self.Phi_c[i,:] + xis_dyn[i,:]
            for i in range(Nseg)
        ]

        constraints_trustregion = [
            xs[i,:] - xbar[i,:] <= self.trust_region_radius_x for i in range(N)
        ] + [
            xs[i,:] - xbar[i,:] >= -self.trust_region_radius_x for i in range(N)
        ]
        if self.trust_region_radius_u is not None:
            constraints_trustregion += [
                us[i,:] - ubar[i,:] <=  self.trust_region_radius_u for i in range(Nseg)
            ] + [
                us[i,:] - ubar[i,:] >= -self.trust_region_radius_u for i in range(Nseg)
            ]

        constraints_initial = [xs[0,:] == self.x0]
        constraints_final   = [xs[-1,0:3] == self.xf[0:3], 
                               xs[-1,3:6] == self.xf[3:6]]
        
        constraints_control = [
            vs[i,0] - self.Tmax * np.exp(-xbar[i,6]) * (1 - (xs[i,6] - xbar[i,6])) <= zetas[i]
            for i in range(Nseg)
        ]

        convex_problem = cp.Problem(
            cp.Minimize(objective_func),
            constraints_objsoc + constraints_dyn + constraints_trustregion + constraints_initial + constraints_final + constraints_control)
        convex_problem.solve(solver = self.solver, verbose = self.verbose_solver)
        self.cp_status = convex_problem.status
        return xs.value, us.value, vs.value, None, xis_dyn.value, None, zetas.value
    
    def evaluate_nonlinear_constraints(self, xs, us, vs, ys=None):
        """Evaluate nonlinear constraints
        
        Returns:
            (tuple): tuple of 1D arrays of nonlinear equality and inequality constraints
        """
        h_ineq = np.array([
            max(vs[i,0] - self.Tmax * np.exp(-xs[i,6]), 0.0) for i in range(self.N-1)
        ])
        return np.zeros(self.ng), h_ineq


class FixedTimeContinuousRdvMass(ContinuousControlSCOCP):
    """Fixed-time continuous rendezvous problem class with log-mass dynamics"""
    def __init__(self, x0, xf, c1, c2, *args, **kwargs):
        assert len(x0) == 7
        assert len(xf) >= 6
        super().__init__(*args, **kwargs)
        self.x0 = x0
        self.xf = xf
        self.c1 = c1
        self.c2 = c2
        return
        
    def evaluate_objective(self, xs, us, vs, ys=None):
        """Evaluate the objective function"""
        return -xs[-1,6]
    
    def solve_convex_problem(self, xbar, ubar, vbar, ybar=None):
        """Solve the convex subproblem
        
        Args:
            xbar (np.array): `(N, self.integrator.nx)` array of reference state history
            ubar (np.array): `(N-1, self.integrator.nu)` array of reference control history
            vbar (np.array): `(N-1, self.integrator.nv)` array of reference constraint history
        
        Returns:
            (tuple): np.array values of xs, us, gs, xi_dyn, xi_eq, zeta_ineq
        """
        N,nx = xbar.shape
        _,nu = ubar.shape
        Nseg = N - 1
        
        xs = cp.Variable((N, nx), name='state')
        us = cp.Variable((Nseg, nu), name='control')
        vs = cp.Variable((Nseg, 1), name='Gamma')
        xis_dyn = cp.Variable((Nseg,nx), name='xi_dyn')         # slack for dynamics
        
        constraints_l1_penalty = []
        if self.l1_penalty:
            slack_l1_xi_dyn = cp.Variable((Nseg,nx), name='slack_l1_xi_dyn')    # slack for L1 penalization of non-convex dynamics
            constraints_l1_penalty.append(slack_l1_xi_dyn >= 0.0)               # slack must be non-negative
            constraints_l1_penalty.append(xis_dyn <=  slack_l1_xi_dyn)
            constraints_l1_penalty.append(xis_dyn >= -slack_l1_xi_dyn)
        else:
            slack_l1_xi_dyn = None

        penalty = get_augmented_lagrangian_penalty(self.weight, xis_dyn, self.lmb_dynamics, slack_l1_xi_dyn=slack_l1_xi_dyn)
        objective_func = -xs[-1,6] + penalty

        constraints_control = [cp.SOC(vs[i,0], us[i,:]) for i in range(Nseg)] + [
            vs[i,0] <= 1.0 for i in range(Nseg)
        ] + [
            0.0 <= vs[i,0] for i in range(Nseg)
        ]

        constraints_dyn = [
            xs[i+1,:] == self.Phi_A[i,:,:] @ xs[i,:] + self.Phi_B[i,:,0:3] @ us[i,:] + self.Phi_B[i,:,3] * vs[i,:] + self.Phi_c[i,:] + xis_dyn[i,:]
            for i in range(Nseg)
        ]

        constraints_trustregion = [
            xs[i,:] - xbar[i,:] <= self.trust_region_radius_x for i in range(N)
        ] + [
            xs[i,:] - xbar[i,:] >= -self.trust_region_radius_x for i in range(N)
        ]
        if self.trust_region_radius_u is not None:
            constraints_trustregion += [
                us[i,:] - ubar[i,:] <=  self.trust_region_radius_u for i in range(Nseg)
            ] + [
                us[i,:] - ubar[i,:] >= -self.trust_region_radius_u for i in range(Nseg)
            ]

        constraints_initial = [xs[0,:] == self.x0]
        constraints_final   = [xs[-1,0:6] == self.xf[0:6]]

        convex_problem = cp.Problem(
            cp.Minimize(objective_func),
            constraints_dyn + constraints_trustregion + constraints_initial +\
                  constraints_final + constraints_control + constraints_l1_penalty)
        convex_problem.solve(solver = self.solver, verbose = self.verbose_solver)
        self.cp_status = convex_problem.status
        return xs.value, us.value, vs.value, None, xis_dyn.value, None, None


