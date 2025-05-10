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
        assert len(x0) == 6
        assert len(xf) == 6
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
            xs[i,:] - xbar[i,:] <= self.trust_region_radius for i in range(N)
        ] + [
            xs[i,:] - xbar[i,:] >= -self.trust_region_radius for i in range(N)
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
            xs[i,:] - xbar[i,:] <= self.trust_region_radius for i in range(N)
        ] + [
            xs[i,:] - xbar[i,:] >= -self.trust_region_radius for i in range(N)
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



class FreeTimeContinuousRdv(ContinuousControlSCOCP):
    """Free-time continuous rendezvous problem class
    A good initial guess for the dilation factor `s` is the guessed TOF, since `dt/dtau = s`.
    
    Args:
        x0 (np.array): initial state
        xf (np.array): final state
        umax (float): maximum control magnitude
        tf_bounds (list): bounds on the final time, given as [tf_min, tf_max]
        s_bounds (list): bounds on time dilation factor, given as [s_min, s_max]
    """
    def __init__(self, x0, xf, umax, tf_bounds, s_bounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(x0) == 6
        assert len(xf) == 6
        assert abs(self.times[0]  - 0.0) < 1e-14, f"self.times[0] must be 0.0, but given {self.times[0]}"
        assert abs(self.times[-1] - 1.0) < 1e-14, f"self.times[-1] must be 1.0, but given {self.times[-1]}"
        assert s_bounds[0] > 0.0, f"s_bounds[0] must be greater than 0.0, but given {s_bounds[0]}"
        assert s_bounds[0] < s_bounds[1], f"s_bounds[0] must be less than s_bounds[1], but given {s_bounds[0]} and {s_bounds[1]}"
        self.x0 = x0
        self.xf = xf
        self.umax = umax
        self.tf_bounds = tf_bounds
        self.s_bounds = s_bounds
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
            (tuple): np.array values of xs, us, gs, xi_dyn, xi_eq, zeta_ineq
        """
        N,nx = xbar.shape
        _,nu = ubar.shape
        Nseg = N - 1
        assert nx == self.integrator.nx, f"xbar.shape[1] = {xbar.shape[1]} must match integrator.nx = {self.integrator.nx}"
        assert nu == self.integrator.nu, f"ubar.shape[1] = {ubar.shape[1]} must match integrator.nu = {self.integrator.nu}"
        
        xs = cp.Variable((N, nx), name='state')
        us = cp.Variable((Nseg, nu), name='control')
        vs = cp.Variable((Nseg, 1), name='Gamma')
        xis_dyn = cp.Variable((Nseg,nx), name='xi_dyn')         # slack for dynamics
        
        penalty = get_augmented_lagrangian_penalty(self.weight, xis_dyn, self.lmb_dynamics)
        dts = np.diff(self.times)
        objective_func = cp.sum(vs.T @ dts) + penalty
        constraints_objsoc = [cp.SOC(vs[i,0], us[i,0:3]) for i in range(N-1)]

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
            xs[i,0:6] - xbar[i,0:6] <= self.trust_region_radius for i in range(N)
        ] + [
            xs[i,0:6] - xbar[i,0:6] >= -self.trust_region_radius for i in range(N)
        ]

        constraints_initial = [xs[0,0:6]  == self.x0[0:6]]
        constraints_final   = [xs[-1,0:3] == self.xf[0:3], 
                               xs[-1,3:6] == self.xf[3:6]]
        
        constraint_t0 = [xs[0,6] == 0.0]
        constraints_tf      = [self.tf_bounds[0] <= xs[-1,6],
                               xs[-1,6] <= self.tf_bounds[1]]
        constraints_s       = [self.s_bounds[0] <= us[i,3] for i in range(Nseg)] + [us[i,3] <= self.s_bounds[1] for i in range(Nseg)]

        constraints_control = [
            vs[i,0] <= self.umax for i in range(Nseg)
        ]

        convex_problem = cp.Problem(
            cp.Minimize(objective_func),
            constraints_objsoc + constraints_dyn + constraints_trustregion +\
            constraints_initial + constraints_final + constraints_control +\
            constraint_t0 + constraints_tf + constraints_s
        )
        convex_problem.solve(solver = self.solver, verbose = self.verbose_solver)
        self.cp_status = convex_problem.status
        return xs.value, us.value, vs.value, None, xis_dyn.value, None, None


class FreeTimeContinuousRdvLogMass(ContinuousControlSCOCP):
    """Free-time continuous rendezvous problem class with log-mass dynamics
    
    Note the ordering expected for the state and the control vectors: 

    state = [x,y,z,vx,vy,vz,log(mass),t]
    u = [ax, ay, az, s, Gamma] where s is the dilation factor, Gamma is the control magnitude (at convergence)
    
    """
    def __init__(self, x0, xf, Tmax, tf_bounds, s_bounds, N, *args, **kwargs):
        super().__init__(nh = N - 1, *args, **kwargs)
        assert len(x0) == 7
        assert len(xf) >= 6
        assert abs(self.times[0]  - 0.0) < 1e-14, f"self.times[0] must be 0.0, but given {self.times[0]}"
        assert abs(self.times[-1] - 1.0) < 1e-14, f"self.times[-1] must be 1.0, but given {self.times[-1]}"
        assert s_bounds[0] > 0.0, f"s_bounds[0] must be greater than 0.0, but given {s_bounds[0]}"
        assert s_bounds[0] < s_bounds[1], f"s_bounds[0] must be less than s_bounds[1], but given {s_bounds[0]} and {s_bounds[1]}"
        self.x0 = x0
        self.xf = xf
        self.Tmax = Tmax
        self.tf_bounds = tf_bounds
        self.s_bounds = s_bounds
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
        objective_func = -xs[-1,6] + penalty
        constraints_objsoc = [cp.SOC(vs[i,0], us[i,0:3]) for i in range(N-1)]
        
        constraints_dyn = [
            xs[i+1,:] == self.Phi_A[i,:,:] @ xs[i,:] + self.Phi_B[i,:,0:4] @ us[i,:] + self.Phi_B[i,:,4] * vs[i,:] + self.Phi_c[i,:] + xis_dyn[i,:]
            for i in range(Nseg)
        ]

        constraints_trustregion = [
            xs[i,:] - xbar[i,:] <= self.trust_region_radius for i in range(N)
        ] + [
            xs[i,:] - xbar[i,:] >= -self.trust_region_radius for i in range(N)
        ]

        constraints_initial = [xs[0,0:7] == self.x0[0:7]]
        constraints_final   = [xs[-1,0:3] == self.xf[0:3], 
                               xs[-1,3:6] == self.xf[3:6]]
        
        constraint_t0 = [xs[0,7] == 0.0]
        constraints_tf      = [self.tf_bounds[0] <= xs[-1,7],
                               xs[-1,7] <= self.tf_bounds[1]]
        constraints_s       = [self.s_bounds[0] <= us[i,3] for i in range(Nseg)] + [us[i,3] <= self.s_bounds[1] for i in range(Nseg)]

        
        constraints_control = [
            vs[i,0] - self.Tmax * np.exp(-xbar[i,6]) * (1 - (xs[i,6] - xbar[i,6])) <= zetas[i]
            for i in range(Nseg)
        ]

        convex_problem = cp.Problem(
            cp.Minimize(objective_func),
            constraints_objsoc + constraints_dyn + constraints_trustregion +\
            constraints_initial + constraints_final + constraints_control +\
            constraint_t0 + constraints_tf + constraints_s)
        convex_problem.solve(solver = self.solver, verbose = self.verbose_solver)
        self.cp_status = convex_problem.status
        return xs.value, us.value, vs.value, None, xis_dyn.value, None, zetas.value
    
    def evaluate_nonlinear_constraints(self, xs, us, gs, ys=None):
        """Evaluate nonlinear constraints
        
        Returns:
            (tuple): tuple of 1D arrays of nonlinear equality and inequality constraints
        """
        h_ineq = np.array([
            max(gs[i,0] - self.Tmax * np.exp(-xs[i,6]), 0.0) for i in range(self.N-1)
        ])
        return np.zeros(self.ng), h_ineq
