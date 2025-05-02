"""Convex subproblem class"""

import cvxpy as cp
import numpy as np


class ImpulsiveControlSCOCP:
    """Convex subproblem class"""
    def __init__(
        self,
        integrator,
        times,
        weight = 1e2,
        trust_region_radius = 0.1,
        solver = cp.CLARABEL,
        verbose_solver = False,
    ):
        self.integrator = integrator
        self.times = times
        self.N = len(times)
        self.weight = weight
        self.trust_region_radius = trust_region_radius
        self.solver = solver
        self.verbose_solver = verbose_solver

        # initialize storage
        self.cp_status = "not_solved"
        Nseg = self.N - 1
        self.Phi_A = np.zeros((Nseg,6,6))
        self.Phi_B = np.zeros((Nseg,6,3))
        self.Phi_c = np.zeros((Nseg,6))
        self.lmb_dynamics = np.zeros((Nseg,6))
        return
    
    def evaluate_objective(self, xs, us, gs):
        """Evaluate the objective function"""
        raise NotImplementedError("Subproblem must be implemented by inherited class!")
    
    def solve_convex_problem(self, xbar, ubar, gbar):
        """Solve the convex subproblem"""
        raise NotImplementedError("Subproblem must be implemented by inherited class!")
    
    def build_linear_model(self, xbar, ubar):
        B = np.concatenate((np.zeros((3,3)), np.eye(3)))
        sols = []
        for i,ti in enumerate(self.times[:-1]):
            _tspan = (ti, self.times[i+1])
            _x0 = xbar[i,:] + B @ ubar[i,:]
            _sol = self.integrator.solve(_tspan, _x0, stm=True)
            sols.append(_sol)

            xf = _sol.y[0:6,-1]
            STM = _sol.y[6:,-1].reshape(6,6)
            self.Phi_A[i,:,:] = STM
            self.Phi_B[i,:,:] = STM @ B
            self.Phi_c[i,:]   = xf - self.Phi_A[i,:,:] @ xbar[i,:] - self.Phi_B[i,:,:] @ ubar[i,:]
        return
        
    def evaluate_nonlinear_dynamics(
        self,
        xbar,
        ubar,
        stm = False,
    ):
        """Evaluate nonlinear dynamics along given state and control history
        
        Args:
            integrator (obj): integrator object
            times (np.array): time grid
            xbar (np.array): state history
            ubar (np.array): control history
            stm (bool): whether to propagate STMs, defaults to False
        """
        assert xbar.shape == (self.N,6)
        assert ubar.shape == (self.N,3)

        B = np.concatenate((np.zeros((3,3)), np.eye(3)))
        sols = []
        geq_nl = np.zeros((self.N-1,6))
        for i,ti in enumerate(self.times[:-1]):
            _tspan = (ti, self.times[i+1])
            _x0 = xbar[i,:] + B @ ubar[i,:]
            _sol = self.integrator.solve(_tspan, _x0, stm=stm)
            sols.append(_sol)
            geq_nl[i,:] = xbar[i+1,:] - _sol.y[0:6,-1]
        return geq_nl, sols
    

class FixedTimeImpulsiveRendezvous(ImpulsiveControlSCOCP):
    """Fixed-time impulsive rendezvous subproblem"""
    def __init__(self, x0, xf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x0 = x0
        self.xf = xf
        return
        
    def evaluate_objective(self, xs, us, gs):
        """Evaluate the objective function"""
        return np.sum(gs)
    
    def solve_convex_problem(self, xbar, ubar, gbar):
        N,nx = xbar.shape
        _,nu = ubar.shape
        Nseg = N - 1
        
        xs = cp.Variable((N, nx), name='state')
        us = cp.Variable((N, nu), name='control')
        gs = cp.Variable((N, 1), name='Gamma')
        xis = cp.Variable((Nseg,nx), name='xi')         # slack for dynamics
        
        penalty = self.weight/2 * cp.sum_squares(xis)
        for i in range(Nseg):
            penalty += self.lmb_dynamics[i,:] @ xis[i,:]
        objective_func = cp.sum(gs) + penalty
        constraints_objsoc = [cp.SOC(gs[i,0], us[i,:]) for i in range(N)]

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
        return xs.value, us.value, gs.value, xis.value, None, None
    
