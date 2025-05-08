"""Planet-to-planet optimal control problem"""

import cvxpy as cp
import numpy as np
import pykep as pk
from .._misc import get_augmented_lagrangian_penalty
from .._scocp_continuous import ContinuousControlSCOCP

class scocp_pl2pl(ContinuousControlSCOCP):
    """Free-time continuous rendezvous problem class with log-mass dynamics
    
    Note the ordering expected for the state and the control vectors: 

    state = [x,y,z,vx,vy,vz,log(mass),t]
    u = [ax, ay, az, s, Gamma] where s is the dilation factor, Gamma is the control magnitude (at convergence)
    
    """
    def __init__(
        self,
        integrator,
        p0: pk.planet.planet,
        pf: pk.planet.planet,
        mass,
        thrust,
        isp,
        nseg: int,
        t0: float,
        tof: tuple[float, float],
        s_bounds: tuple[float, float],
        g0 = 9.81,
        *args, **kwargs
    ):
        assert isinstance(p0, pk.planet.planet)
        assert isinstance(pf, pk.planet.planet)
        N = nseg + 1
        ng = 0
        nh = N - 1
        augment_Gamma = True
        times = np.linspace(0, 1, N)
        super().__init__(integrator, times, ng=ng, nh=nh, augment_Gamma=augment_Gamma, *args, **kwargs)
        
        self.p0 = p0
        self.pf = pf
        self.mass0 = mass
        self.Tmax = thrust
        self.cex = isp * g0
        self.t0 = t0
        self.tf_bounds = tof
        self.s_bounds = s_bounds
        return
        
    def evaluate_objective(self, xs, us, gs):
        """Evaluate the objective function"""
        return -xs[-1,6]
    
    def solve_convex_problem(self, xbar, ubar, gbar):
        """Solve the convex subproblem
        
        Args:
            xbar (np.array): `(N, self.integrator.nx)` array of reference state history
            ubar (np.array): `(N-1, self.integrator.nu)` array of reference control history
            gbar (np.array): `(N-1, self.integrator.n_gamma)` array of reference constraint history
        
        Returns:
            (tuple): np.array values of xs, us, gs, xi_dyn, xi_eq, zeta_ineq
        """
        N,nx = xbar.shape
        _,nu = ubar.shape
        Nseg = N - 1
        
        xs = cp.Variable((N, nx), name='state')
        us = cp.Variable((Nseg, nu), name='control')
        gs = cp.Variable((Nseg, 1), name='Gamma')
        xis = cp.Variable((Nseg,nx), name='xi')         # slack for dynamics
        zetas = cp.Variable((Nseg,), name='zeta')     # slack for non-convex inequality
        
        penalty = get_augmented_lagrangian_penalty(self.weight, xis, self.lmb_dynamics, zeta=zetas, lmb_ineq=self.lmb_ineq)
        objective_func = -xs[-1,6] + penalty
        constraints_objsoc = [cp.SOC(gs[i,0], us[i,0:3]) for i in range(N-1)]
        
        constraints_dyn = [
            xs[i+1,:] == self.Phi_A[i,:,:] @ xs[i,:] + self.Phi_B[i,:,0:4] @ us[i,:] + self.Phi_B[i,:,4] * gs[i,:] + self.Phi_c[i,:] + xis[i,:]
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
            gs[i,0] - self.Tmax * np.exp(-xbar[i,6]) * (1 - (xs[i,6] - xbar[i,6])) <= zetas[i]
            for i in range(Nseg)
        ]

        convex_problem = cp.Problem(
            cp.Minimize(objective_func),
            constraints_objsoc + constraints_dyn + constraints_trustregion +\
            constraints_initial + constraints_final + constraints_control +\
            constraint_t0 + constraints_tf + constraints_s)
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
