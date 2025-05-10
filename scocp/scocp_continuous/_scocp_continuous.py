"""Sequentially convexified optimal control problem (SCOCP) for continuous dynamics"""

import cvxpy as cp
import numpy as np

from .._misc import get_augmented_lagrangian_penalty


class ContinuousControlSCOCP:
    """Sequentially convexified optimal control problem (SCOCP) for continuous dynamics
    
    Args:
        integrator (obj): integrator object
        times (np.array): time grid
        ng (int): number of nonlinear equality constraints, excluding dynamics constraints
        nh (int): number of nonlinear inequality constraints
        ny (int): number of other variables
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
        ny: int = 0,
        augment_Gamma: bool = False,
        weight: float = 1e2,
        trust_region_radius: float = 0.1,
        solver = cp.CLARABEL,
        verbose_solver: bool = False,
    ):
        assert integrator.impulsive is False, "Continuous control problem must be initialized with an integrator for continuous dynamics"
        assert weight >= 0.0, f"weight must be non-negative, but given {weight}"
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
        # initialize storage
        self.cp_status = "not_solved"
        Nseg = self.N - 1
        if augment_Gamma:
            self.Phi_A = np.zeros((Nseg,self.integrator.nx,self.integrator.nx))
            self.Phi_B = np.zeros((Nseg,self.integrator.nx,self.integrator.nu+1))
            self.Phi_c = np.zeros((Nseg,self.integrator.nx))
            assert self.integrator.nv >= 1,\
                "When formulating SCOCP with `augment_Gamma=True`, `integrator.nv` must be set to at least 1"
        else:
            self.Phi_A = np.zeros((Nseg,self.integrator.nx,self.integrator.nx))
            self.Phi_B = np.zeros((Nseg,self.integrator.nx,self.integrator.nu))
            self.Phi_c = np.zeros((Nseg,self.integrator.nx))

        # initialize multipliers
        self.lmb_dynamics = np.zeros((Nseg,self.integrator.nx))
        self.lmb_eq       = np.zeros(self.ng)
        self.lmb_ineq     = np.zeros(self.nh)
        return
    
    def evaluate_objective(self, xs, us, vs, ys=None):
        """Evaluate the objective function"""
        raise NotImplementedError("Subproblem must be implemented by inherited class!")
    
    def solve_convex_problem(self, xbar, ubar, vbar, ybar = None):
        """Solve the convex subproblem
        
        Args:
            xbar (np.array): `(N, self.integrator.nx)` array of reference state history
            ubar (np.array): `(N-1, self.integrator.nu)` array of reference control history
            vbar (np.array): `(N-1, self.integrator.nv)` array of reference constraint history
            ybar (np.array): `(self.ny,)` other reference variables
        
        Returns:
            (tuple): np.array values of xs, us, gs, xi_dyn, xi_eq, zeta_ineq
        """
        raise NotImplementedError("Subproblem must be implemented by inherited class!")
    
    def build_linear_model(self, xbar, ubar, vbar):
        """Construct linear model for dynamics multiple-shooting constraints within SCP algorithm
        This function computes the Phi_A, Phi_B, and Phi_c matrices and stores them in the class attributes.

        Args:
            xbar (np.array): `(N, self.integrator.nx)` array of state history
            ubar (np.array): `(N-1, self.integrator.nu)` array of control history
            vbar (np.array): `(N-1, self.integrator.nv)` array of constraint history
        """
        assert xbar.shape == (self.N, self.integrator.nx),\
            f"Given incorrect xbar shape {xbar.shape}; should be {(self.N, self.integrator.nx)}"
        assert ubar.shape == (self.N-1, self.integrator.nu),\
            f"Given incorrect ubar shape {ubar.shape}; should be {(self.N-1, self.integrator.nu)}"
        if self.integrator.nv == 0:
            assert vbar.shape == (self.N-1,1), f"Given incorrect vbar shape {vbar.shape}; should be {(self.N-1,1)}"
        else:
            assert vbar.shape == (self.N-1, self.integrator.nv),\
                f"Given incorrect vbar shape {vbar.shape}; should be {(self.N-1, self.integrator.nv)}"
        i_PhiA_end = self.integrator.nx + self.integrator.nx * self.integrator.nx
        for i,ti in enumerate(self.times[:-1]):
            _tspan = (ti, self.times[i+1])
            if self.augment_Gamma:
                _ubar = np.concatenate((ubar[i,:], vbar[i,:]))
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
        
    def evaluate_nonlinear_dynamics(self, xs, us, vs, stm = False, steps = None):
        """Evaluate nonlinear dynamics along given state and control history
        
        Args:
            integrator (obj): integrator object
            times (np.array): time grid
            vs (np.array): state history
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
                _ts, _ys = self.integrator.solve(_tspan, xs[i,:], u=np.concatenate((us[i,:], vs[i,:])), stm=stm, t_eval=t_eval)
            else:
                _ts, _ys = self.integrator.solve(_tspan, xs[i,:], u=us[i,:], stm=stm, t_eval=t_eval)
            sols.append([_ts,_ys])
            geq_nl[i,:] = xs[i+1,:] - _ys[-1,0:self.integrator.nx]
        return geq_nl, sols
    
    def evaluate_nonlinear_constraints(self, xs, us, gs, ys=None):
        """Evaluate nonlinear constraints
        
        Returns:
            (tuple): tuple of 1D arrays of nonlinear equality and inequality constraints
        """
        return np.zeros(self.ng), np.zeros(self.nh)
