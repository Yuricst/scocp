"""Sequentially convexified optimal control problem (SCOCP) for continuous dynamics"""

import cvxpy as cp
import numpy as np


class ContinuousControlSCOCP:
    """Sequentially convexified optimal control problem (SCOCP) for continuous dynamics
    
    Args:
        integrator (obj): integrator object
        times (np.array): time grid
        ng (int): number of nonlinear equality constraints, excluding dynamics constraints
        nh (int): number of nonlinear inequality constraints
        augment_Gamma (bool): whether to augment the control with the constraint vector when integrating the dynamics
        weight (float): weight of the objective function
        trust_region_radius (float): trust region radius
        solver (str): solver to use
        verbose_solver (bool): whether to print verbose output
    """
    def __init__(
        self,
        integrator,
        times,
        ng: int = 0,
        nh: int = 0,
        augment_Gamma: bool = False,
        weight: float = 1e2,
        trust_region_radius: float = 0.1,
        solver = cp.CLARABEL,
        verbose_solver: bool = False,
    ):
        assert integrator.impulsive is False, "Continuous control problem must be initialized with an integrator for continuous dynamics"
        self.integrator = integrator
        self.times = times
        self.N = len(times)
        self.ng_dyn = self.integrator.nx * (self.N - 1)
        self.ng = ng
        self.nh = nh
        self.weight = weight
        self.trust_region_radius = trust_region_radius
        self.solver = solver
        self.verbose_solver = verbose_solver
        self.augment_Gamma = augment_Gamma
        # initialize storage
        self.cp_status = "not_solved"
        Nseg = self.N - 1
        if augment_Gamma:
            self.Phi_A = np.zeros((Nseg,self.integrator.nx,self.integrator.nx))
            self.Phi_B = np.zeros((Nseg,self.integrator.nx,self.integrator.nu+1))
            self.Phi_c = np.zeros((Nseg,self.integrator.nx))
            assert self.integrator.n_gamma >= 1,\
                "When formulating SCOCP with `augment_Gamma=True`, `integrator.n_gamma` must be set to at least 1"
        else:
            self.Phi_A = np.zeros((Nseg,self.integrator.nx,self.integrator.nx))
            self.Phi_B = np.zeros((Nseg,self.integrator.nx,self.integrator.nu))
            self.Phi_c = np.zeros((Nseg,self.integrator.nx))

        # initialize multipliers
        self.lmb_dynamics = np.zeros((Nseg,self.integrator.nx))
        self.lmb_eq       = np.zeros(self.ng)
        self.lmb_ineq     = np.zeros(self.nh)
        return
    
    def evaluate_objective(self, xs, us, gs):
        """Evaluate the objective function"""
        raise NotImplementedError("Subproblem must be implemented by inherited class!")
    
    def solve_convex_problem(self, xbar, ubar, gbar):
        """Solve the convex subproblem"""
        raise NotImplementedError("Subproblem must be implemented by inherited class!")
    
    def build_linear_model(self, xbar, ubar, gbar):
        i_PhiA_end = self.integrator.nx + self.integrator.nx * self.integrator.nx
        for i,ti in enumerate(self.times[:-1]):
            _tspan = (ti, self.times[i+1])
            if self.augment_Gamma:
                _ubar = np.concatenate((ubar[i,:], gbar[i,:]))
            else:
                _ubar = ubar[i,:]
            _, _ys = self.integrator.solve(_tspan, xbar[i,:], u=_ubar, stm=True)

            xf  = _ys[-1,0:self.integrator.nx]
            self.Phi_A[i,:,:] = _ys[-1,self.integrator.nx:i_PhiA_end].reshape(self.integrator.nx,self.integrator.nx)
            if self.augment_Gamma:
                self.Phi_B[i,:,:] = _ys[-1,i_PhiA_end:].reshape(self.integrator.nx,self.integrator.nu+1)
            else:
                self.Phi_B[i,:,:] = _ys[-1,i_PhiA_end:].reshape(self.integrator.nx,self.integrator.nu)
            self.Phi_c[i,:]   = xf - self.Phi_A[i,:,:] @ xbar[i,:] - self.Phi_B[i,:,:] @ _ubar
        return
        
    def evaluate_nonlinear_dynamics(self, xs, us, gs, stm = False, steps = None):
        """Evaluate nonlinear dynamics along given state and control history
        
        Args:
            integrator (obj): integrator object
            times (np.array): time grid
            xs (np.array): state history
            ubar (np.array): control history
            stm (bool): whether to propagate STMs, defaults to False
        """
        assert xs.shape == (self.N,self.integrator.nx)
        assert us.shape == (self.N-1,self.integrator.nu)

        sols = []
        geq_nl = np.zeros((self.N-1,self.integrator.nx))
        for i,ti in enumerate(self.times[:-1]):
            _tspan = (ti, self.times[i+1])
            if steps is None:
                t_eval = None
            else:
                t_eval = np.linspace(ti, self.times[i+1], steps)
            if self.augment_Gamma:
                _ts, _ys = self.integrator.solve(_tspan, xs[i,:], u=np.concatenate((us[i,:], gs[i,:])), stm=stm, t_eval=t_eval)
            else:
                _ts, _ys = self.integrator.solve(_tspan, xs[i,:], u=us[i,:], stm=stm, t_eval=t_eval)
            sols.append([_ts,_ys])
            geq_nl[i,:] = xs[i+1,:] - _ys[-1,0:self.integrator.nx]
        return geq_nl, sols
    
    def evaluate_nonlinear_constraints(self, xs, us, gs):
        """Evaluate nonlinear constraints
        
        Returns:
            (tuple): tuple of 1D arrays of nonlinear equality and inequality constraints
        """
        return np.zeros(self.ng), np.zeros(self.nh)

class FixedTimeContinuousRendezvous(ContinuousControlSCOCP):
    """Fixed-time continuous rendezvous subproblem"""
    def __init__(self, x0, xf, umax, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(x0) == 7
        assert len(xf) >= 6
        self.x0 = x0
        self.xf = xf
        self.umax = umax
        return
        
    def evaluate_objective(self, xs, us, gs):
        """Evaluate the objective function"""
        return np.sum(gs)
    
    def solve_convex_problem(self, xbar, ubar, gbar):
        N,nx = xbar.shape
        _,nu = ubar.shape
        Nseg = N - 1
        
        xs = cp.Variable((N, nx), name='state')
        us = cp.Variable((Nseg, nu), name='control')
        gs = cp.Variable((Nseg, 1), name='Gamma')
        xis = cp.Variable((Nseg,nx), name='xi')         # slack for dynamics
        
        penalty = self.weight/2 * cp.sum_squares(xis)
        for i in range(Nseg):
            penalty += self.lmb_dynamics[i,:] @ xis[i,:]
        objective_func = cp.sum(gs) + penalty
        constraints_objsoc = [cp.SOC(gs[i,0], us[i,:]) for i in range(N-1)]

        if self.augment_Gamma:
            constraints_dyn = [
                xs[i+1,:] == self.Phi_A[i,:,:] @ xs[i,:] + self.Phi_B[i,:,:] @ np.concatenate([us[i,:], gs[i,:]]) + self.Phi_c[i,:] + xis[i,:]
                for i in range(Nseg)
            ]
        else:
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
                               xs[-1,3:6] == self.xf[3:6]]
        
        constraints_control = [
            gs[i,0] <= self.umax for i in range(Nseg)
        ]

        convex_problem = cp.Problem(
            cp.Minimize(objective_func),
            constraints_objsoc + constraints_dyn + constraints_trustregion + constraints_initial + constraints_final + constraints_control)
        convex_problem.solve(solver = self.solver, verbose = self.verbose_solver)
        self.cp_status = convex_problem.status
        return xs.value, us.value, gs.value, xis.value, None, None
    

class FixedTimeContinuousRendezvousLogMass(ContinuousControlSCOCP):
    """Fixed-time continuous rendezvous subproblem with log-mass dynamics"""
    def __init__(self, x0, xf, Tmax, N, *args, **kwargs):
        super().__init__(nh = N - 1, *args, **kwargs)
        self.x0 = x0
        self.xf = xf
        self.Tmax = Tmax
        return
        
    def evaluate_objective(self, xs, us, gs):
        """Evaluate the objective function"""
        return -xs[-1,6]
    
    def solve_convex_problem(self, xbar, ubar, gbar):
        N,nx = xbar.shape
        _,nu = ubar.shape
        Nseg = N - 1
        
        xs = cp.Variable((N, nx), name='state')
        us = cp.Variable((Nseg, nu), name='control')
        gs = cp.Variable((Nseg, 1), name='Gamma')
        xis = cp.Variable((Nseg,nx), name='xi')         # slack for dynamics
        zetas = cp.Variable((Nseg,), name='zeta')     # slack for non-convex inequality
        
        penalty = self.weight/2 * cp.sum_squares(xis) + self.weight/2 * cp.sum_squares(zetas)
        for i in range(Nseg):
            penalty += self.lmb_dynamics[i,:] @ xis[i,:] + self.lmb_ineq[i] * zetas[i]
        objective_func = cp.sum(gs) + penalty
        constraints_objsoc = [cp.SOC(gs[i,0], us[i,:]) for i in range(N-1)]

        constraints_dyn = [
            xs[i+1,:] == self.Phi_A[i,:,:] @ xs[i,:] + self.Phi_B[i,:,0:3] @ us[i,:] + self.Phi_B[i,:,3] * gs[i,:] + self.Phi_c[i,:] + xis[i,:]
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
            gs[i,0] - self.Tmax * np.exp(-xbar[i,6]) * (1 - (xs[i,6] - xbar[i,6])) <= zetas[i]
            for i in range(Nseg)
        ]

        convex_problem = cp.Problem(
            cp.Minimize(objective_func),
            constraints_objsoc + constraints_dyn + constraints_trustregion + constraints_initial + constraints_final + constraints_control)
        convex_problem.solve(solver = self.solver, verbose = self.verbose_solver)
        self.cp_status = convex_problem.status
        return xs.value, us.value, gs.value, xis.value, None, zetas.value
    
    def evaluate_nonlinear_constraints(self, xs, us, gs):
        """Evaluate nonlinear constraints
        
        Returns:
            (tuple): tuple of 1D arrays of nonlinear equality and inequality constraints
        """
        h_ineq = np.array([
            max(gs[i,0] - self.Tmax * np.exp(-xs[i,6]), 0.0) for i in range(self.N-1)
        ])
        return np.zeros(self.ng), h_ineq

    
