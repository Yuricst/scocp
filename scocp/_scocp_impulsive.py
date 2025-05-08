"""Sequentially convexified optimal control problem (SCOCP) for impulsive dynamics"""

import cvxpy as cp
import numpy as np

from ._misc import get_augmented_lagrangian_penalty

class ImpulsiveControlSCOCP:
    """Sequentially convexified optimal control problem (SCOCP) for impulsive dynamics
    
    Args:
        integrator (obj): integrator object
        times (np.array): time grid
        ng (int): number of nonlinear equality constraints, excluding dynamics constraints
        nh (int): number of nonlinear inequality constraints
        ny (int): number of other variables
        augment_Gamma (bool): whether to augment the control with the constraint vector when integrating the dynamics
        B (np.array): control matrix
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
        ny: int = 0,
        augment_Gamma: bool = False,
        B = None,
        weight: float = 1e2,
        trust_region_radius: float = 0.1,
        solver = cp.CLARABEL,
        verbose_solver: bool = False,
    ):
        assert integrator.impulsive is True, "Impulsive control problem must be initialized with an integrator for impulsive dynamics"
        self.integrator = integrator
        self.times = times
        self.N = len(times)
        self.ng_dyn = self.integrator.nx * (self.N - 1)
        self.ng = ng
        self.nh = nh
        self.ny = ny
        self.weight = weight
        self.trust_region_radius = trust_region_radius
        self.solver = solver
        self.verbose_solver = verbose_solver
        self.augment_Gamma = augment_Gamma

        if B is None:
            if augment_Gamma:
                self.B = np.array([
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ])
            else:
                self.B = np.concatenate((np.zeros((3,3)), np.eye(3)))
        else:
            self.B = B

        # initialize storage
        self.cp_status = "not_solved"
        Nseg = self.N - 1
        if augment_Gamma:
            self.Phi_A = np.zeros((Nseg,self.integrator.nx,self.integrator.nx))
            self.Phi_B = np.zeros((Nseg,self.integrator.nx,self.integrator.nu+1))
            self.Phi_c = np.zeros((Nseg,self.integrator.nx))
        else:
            self.Phi_A = np.zeros((Nseg,self.integrator.nx,self.integrator.nx))
            self.Phi_B = np.zeros((Nseg,self.integrator.nx,self.integrator.nu))
            self.Phi_c = np.zeros((Nseg,self.integrator.nx))

        # initialize multipliers
        self.lmb_dynamics = np.zeros((Nseg,6))
        self.lmb_eq       = np.zeros(self.ng)
        self.lmb_ineq     = np.zeros(self.nh)
        return
    
    def evaluate_objective(self, xs, us, gs, ys=None):
        """Evaluate the objective function"""
        raise NotImplementedError("Subproblem must be implemented by inherited class!")
    
    def solve_convex_problem(self, xbar, ubar, vbar, ybar=None):
        """Solve the convex subproblem"""
        raise NotImplementedError("Subproblem must be implemented by inherited class!")
    
    def build_linear_model(self, xbar, ubar, vbar):
        for i,ti in enumerate(self.times[:-1]):
            _tspan = (ti, self.times[i+1])
            _x0 = xbar[i,:] + self.B @ ubar[i,:]
            _, _ys = self.integrator.solve(_tspan, _x0, stm=True)

            xf  = _ys[-1,0:self.integrator.nx]
            STM = _ys[-1,self.integrator.nx:self.integrator.nx+self.integrator.nx*self.integrator.nx].reshape(self.integrator.nx,self.integrator.nx)
            self.Phi_A[i,:,:] = STM
            self.Phi_B[i,:,:] = STM @ self.B
            self.Phi_c[i,:]   = xf - self.Phi_A[i,:,:] @ xbar[i,:] - self.Phi_B[i,:,:] @ ubar[i,:]
        return
        
    def evaluate_nonlinear_dynamics(
        self,
        xbar,
        ubar,
        stm = False,
        steps = None,
    ):
        """Evaluate nonlinear dynamics along given state and control history
        
        Args:
            integrator (obj): integrator object
            times (np.array): time grid
            xbar (np.array): state history
            ubar (np.array): control history
            stm (bool): whether to propagate STMs, defaults to False
        """
        assert xbar.shape == (self.N,self.integrator.nx)
        assert ubar.shape == (self.N,self.integrator.nu)

        sols = []
        geq_nl = np.zeros((self.N-1,self.integrator.nx))
        for i,ti in enumerate(self.times[:-1]):
            _tspan = (ti, self.times[i+1])
            if steps is None:
                t_eval = None
            else:
                t_eval = np.linspace(ti, self.times[i+1], steps)
            _x0 = xbar[i,:] + self.B @ ubar[i,:]
            _ts, _ys = self.integrator.solve(_tspan, _x0, stm=stm, t_eval=t_eval)
            sols.append([_ts,_ys])
            geq_nl[i,:] = xbar[i+1,:] - _ys[-1,0:self.integrator.nx]
        return geq_nl, sols
    
    def evaluate_nonlinear_constraints(self, xs, us, gs, ys=None):
        """Evaluate nonlinear constraints
        
        Returns:
            (tuple): tuple of 1D arrays of nonlinear equality and inequality constraints
        """
        return np.zeros(self.ng), np.zeros(self.nh)
    

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
    
