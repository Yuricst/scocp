"""Planet-to-planet optimal control problem"""

import cvxpy as cp
import numpy as np
import pykep as pk

from .._misc import get_augmented_lagrangian_penalty
from .._scocp_continuous import ContinuousControlSCOCP
from ..eoms import rhs_twobody, control_rhs_twobody_logmass_freetf, control_rhs_twobody_logmass_freetf_stm
from .._integrator_scipy import ScipyIntegrator

class CanonicalScales:
    """Canonical scales for the problem
    
    Args:
        MU (float): mass unit in kg
        GM (float): gravitational parameter in m^3/s^2
        DU (float): distance unit in m
        G0 (float): gravitational acceleration in m/s^2
    """
    def __init__(self, MU: float, GM: float=pk.MU_SUN, DU: float=pk.AU, G0: float = 9.81):
        self.MU = MU
        self.GM = GM
        self.DU = DU
        self.VU = np.sqrt(self.GM/self.DU)
        self.TU = self.DU / self.VU
        self.TU2DAY = self.TU / 86400.0
        self.mu = 1.0
        self.G0 = G0
        return
    
    def thrust_si2canonical(self, THRUST: float):
        """Convert thrust in SI units (kg.m/s^2) to canonical units (MU.DU/TU^2)"""
        return THRUST * (1/self.MU)*(self.TU**2/self.DU)  # canonical max thrust
    
    def isp_si2canonical(self, ISP: float):
        return ISP/self.TU


class PlanetTarget:
    """Moving target object from a `pykep.planet` object
    
    Args:
        planet (pykep.planet.planet): pykep planet object
        t0 (float): initial epoch in mjd2000
        DU (float): distance unit in km
        TU (float): time unit to convert non-dimensional time to elapsed time in days
        mu (float): gravitational parameter in km^3/s^2
    """
    def __init__(self, planet: pk.planet.planet, t0_mjd2000: float, TU2DAY: float, DU: float, VU: float, mu: float):
        self.planet = planet
        self.t0_mjd2000 = t0_mjd2000
        self.TU2DAY = TU2DAY
        self.DU = DU
        self.VU = VU
        self.mu = mu
        return
    
    def target_state(self, t: float) -> np.ndarray:
        """Evaluate planet state at time `epoch = self.t0_mjd2000 + t * self.TU` where `epoch` is in mjd2000
        
        Args:
            t (float): elapsed non-dimensional time since `self.t0_mjd2000` in units `self.TU`
        
        Returns:
            (np.array): Cartesian state vector of the target planet at time `epoch`
        """
        r, v = self.planet.eph(self.t0_mjd2000 + t*self.TU2DAY)
        return np.concatenate((np.array(r)/self.DU, np.array(v)/self.VU))
    
    def target_state_derivative(self, t: float) -> np.ndarray:
        """Evaluate planet state derivative at time `self.t0_mjd2000 + t * self.TU`
        
        Args:
            t (float): elapsed non-dimensional time since `self.t0_mjd2000` in units `self.TU`
        
        Returns:
            (np.array): Cartesian state derivative of the target planet at time `epoch`
        """
        state = self.target_final_state(t)
        return rhs_twobody(t, state, self.mu)
    

class scocp_pl2pl(ContinuousControlSCOCP):
    """Free-time continuous rendezvous problem class with log-mass dynamics
    
    Note the ordering expected for the state and the control vectors: 

    state = [x,y,z,vx,vy,vz,log(mass),t]
    u = [ax, ay, az, s, Gamma] where s is the dilation factor, Gamma is the control magnitude (at convergence)
    
    Args:
        integrator (scocp.ScipyIntegrator or scocp.HeyokaIntegrator): integrator object
        canonical_scales (scocp.CanonicalScales): canonical scales object
        p0 (pykep.planet.planet): initial planet object
        pf (pykep.planet.planet): final planet object
        mass (float): initial mass in canonical mass unit
        thrust (float): thrust magnitude in canonical units
        cex (float): exhaust velocity in canonical units
        nseg (int): number of segments (s.t. `N = nseg + 1`)
        t0_mjd2000 (float): initial epoch in mjd2000
        tof_bounds (tuple[float, float]): time of flight bounds
        s_bounds (tuple[float, float]): control dilation factor bounds
    """
    def __init__(
        self,
        integrator,
        canonical_scales,
        p0: pk.planet.planet,
        pf: pk.planet.planet,
        mass,
        thrust,
        cex,
        nseg: int,
        t0_mjd2000: float,
        tof_bounds: tuple[float, float],
        s_bounds: tuple[float, float],
        vinf_dep: float = 0.0,
        vinf_arr: float = 0.0,
        *args,
        **kwargs
    ):
        # define problem parameters and inherit parent class
        N = nseg + 1
        ng = 6
        nh = N - 1
        augment_Gamma = True
        times = np.linspace(0, 1, N)
        super().__init__(integrator, times, ng=ng, nh=nh, augment_Gamma=augment_Gamma, *args, **kwargs)

        # define additional attributes
        self.canonical_scales = canonical_scales
        self.p0 = p0
        self.pf = pf
        self.mass0 = mass
        self.Tmax = thrust
        self.cex = cex
        self.t0_mjd2000 = t0_mjd2000
        self.tof_bounds = tof_bounds
        self.s_bounds = s_bounds
        self.vinf_dep = vinf_dep
        self.vinf_arr = vinf_arr

        # set initial state based on initial planet state
        r0_dim, v0_dim = self.p0.eph(self.t0_mjd2000)
        self.x0 = np.zeros(7)
        self.x0[0:3] = np.array(r0_dim)/self.canonical_scales.DU
        self.x0[3:6] = np.array(v0_dim)/self.canonical_scales.VU
        self.x0[6] = np.log(self.mass0)

        # construct initial target object
        self.target_initial = PlanetTarget(
            p0,
            t0_mjd2000 = self.t0_mjd2000,
            TU2DAY = self.canonical_scales.TU2DAY,
            DU = self.canonical_scales.DU,
            VU = self.canonical_scales.VU,
            mu = self.canonical_scales.mu,
        )

        # construct final target object
        self.target_final = PlanetTarget(
            pf,
            t0_mjd2000 = self.t0_mjd2000,
            TU2DAY = self.canonical_scales.TU2DAY,
            DU = self.canonical_scales.DU,
            VU = self.canonical_scales.VU,
            mu = self.canonical_scales.mu,
        )
        return
    
    def get_initial_orbit(self, steps: int=100):
        """Get state history of initial orbit over one period
        
        Args:
            steps (int): number of steps
        
        Returns:
            (tuple): tuple of 1D array of time and `(steps,6)` array ofstate history
        """
        oe0 = pk.ic2par(self.x0[0:3], self.x0[3:6], 1.0)
        period = 2*np.pi*np.sqrt(oe0[0]**3/1.0)
        t_eval = np.linspace(0, period, steps)
        states = np.zeros((steps, 6))
        for i, t in enumerate(t_eval):
            states[i,0:3], states[i,3:6] = pk.propagate_lagrangian(self.x0[0:3], self.x0[3:6], t, mu=self.canonical_scales.mu)
        return t_eval, states
    
    def get_final_orbit(self, steps: int=100):
        """Get state history of final orbit over one period
        
        Args:
            steps (int): number of steps
        
        Returns:
            (tuple): tuple of 1D array of time and `(steps,6)` array of state history
        """
        xf0 = self.target_final.target_state(0.0)
        oef = pk.ic2par(xf0[0:3], xf0[3:6], 1.0)
        period = 2*np.pi*np.sqrt(oef[0]**3/1.0)
        t_eval = np.linspace(0, period, steps)
        states = np.zeros((steps, 6))
        for i, t in enumerate(t_eval):
            states[i,0:3], states[i,3:6] = pk.propagate_lagrangian(xf0[0:3], xf0[3:6], t, mu=self.canonical_scales.mu)
        return t_eval, states
    
    def get_vinf(self, xs):
        """Evaluate the objective function
        
        Args:
            xs (np.array): `(N,8)` array of state history
        
        Returns:
            (tuple): tuple of `(3,)` array of v-infinity vectors at departure and arrival
        """
        assert xs.shape == (self.N, 8), f"xs must be a `(N,8)` array, but given {xs.shape}"
        vinf_dep = xs[0,3:6] - self.x0[3:6]
        vinf_arr = xs[-1,3:6] - self.target_final.target_state(xs[-1,7])[3:6]
        return vinf_dep, vinf_arr
    
    def get_initial_guess(self, tof_guess: float):
        """Construct initial guess based on linear interpolation along orbital elements
        
        Args:
            tof_guess (float): guess for time of flight
        
        Returns:
            (tuple): tuple of `(N,8)` array of state history, `(N-1,4)` array of control history, and `(N-1,1)` array of constraint history
        """
        # initial orbital elements
        oe0 = pk.ic2par(self.x0[0:3], self.x0[3:6], self.canonical_scales.mu)

        # final orbital elements
        xf_guess = self.target_final.target_state(tof_guess)
        oef = pk.ic2par(xf_guess[0:3], xf_guess[3:6], self.canonical_scales.mu)

        elements = np.concatenate((
            np.linspace(oe0[0], oef[0], self.N).reshape(-1,1),
            np.linspace(oe0[1], oef[1], self.N).reshape(-1,1),
            np.linspace(oe0[2], oef[2], self.N).reshape(-1,1),
            np.linspace(oe0[3], oef[3], self.N).reshape(-1,1),
            np.linspace(oe0[4], oef[4], self.N).reshape(-1,1),
            np.linspace(oe0[5], oef[5], self.N).reshape(-1,1),
        ), axis=1)

        elements[:,5] = np.linspace(oe0[5], oef[5]+2*np.pi, self.N)
        xbar = np.zeros((self.N,8))
        xbar[:,0:6]  = np.array([np.concatenate(pk.par2ic(E,self.canonical_scales.mu)) for E in elements])
        xbar[:,6]    = np.log(np.linspace(1.0, 0.5, self.N))  # initial guess for log-mass
        xbar[0,0:7]  = self.x0[0:7]          # overwrite initial state
        xbar[-1,0:6] = xf_guess[0:6]          # overwrite final state
        xbar[:,7]    = np.linspace(0, tof_guess, self.N)        # initial guess for time

        sbar_initial = tof_guess * np.ones((self.N-1,1))
        # ubar = np.concatenate((np.divide(np.diff(xbar[:,3:6], axis=0), np.diff(times_guess)[:,None]), sbar_initial), axis=1)
        ubar = np.concatenate((np.zeros((self.N-1,3)), sbar_initial), axis=1)
        gbar = np.sum(ubar[:,0:3], axis=1).reshape(-1,1)
        return xbar, ubar, gbar
    
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
        
        xs      = cp.Variable((N, nx), name='state')
        us      = cp.Variable((Nseg, nu), name='control')
        gs      = cp.Variable((Nseg, 1), name='Gamma')
        xis_dyn = cp.Variable((Nseg,nx), name='xi_dyn')     # slack for dynamics
        xis     = cp.Variable((self.ng,), name='xi')        # slack for target state
        zetas   = cp.Variable((Nseg,), name='zeta')         # slack for non-convex inequality

        # additional variables for initial and final v-infinity vectors
        vinf_dep_vec = cp.Variable((3,), name='vinf_dep_vec')
        vinf_dep_mag = cp.Variable((1,), name='vinf_dep_mag')
        vinf_arr_vec = cp.Variable((3,), name='vinf_arr_vec')
        vinf_arr_mag = cp.Variable((1,), name='vinf_arr_mag')
                
        penalty = get_augmented_lagrangian_penalty(
            self.weight,
            xis_dyn,
            self.lmb_dynamics,
            xi=xis,
            lmb_eq=self.lmb_eq,
            zeta=zetas,
            lmb_ineq=self.lmb_ineq
        )
        objective_func = -xs[-1,6] + penalty
        constraints_objsoc = [cp.SOC(gs[i,0], us[i,0:3]) for i in range(N-1)]
        
        constraints_dyn = [
            xs[i+1,:] == self.Phi_A[i,:,:] @ xs[i,:] + self.Phi_B[i,:,0:4] @ us[i,:] + self.Phi_B[i,:,4] * gs[i,:] + self.Phi_c[i,:] + xis_dyn[i,:]
            for i in range(Nseg)
        ]

        constraints_trustregion = [
            xs[i,:] - xbar[i,:] <= self.trust_region_radius for i in range(N)
        ] + [
            xs[i,:] - xbar[i,:] >= -self.trust_region_radius for i in range(N)
        ]

        constraints_initial = [
            xs[0,0:3]                   == self.x0[0:3],
            xs[0,3:6] + vinf_dep_vec[:] == self.x0[3:6],
            xs[0,6]                     == self.x0[6],
        ]
        constraints_final = [
            xs[-1,0:6] - self.target_final.target_state(xbar[-1,7]) - self.target_final.target_state_derivative(xbar[-1,7]) * (xs[-1,7] - xbar[-1,7]) == xis[0:6]
        ]
        constraints_vinf_mag = [
            cp.SOC(vinf_dep_mag[0], vinf_dep_vec[:]),       # connect v-infinity vector to magnitude
            vinf_dep_mag[0] <= self.vinf_dep,               # magnitude upper-bounded
            cp.SOC(vinf_arr_mag[0], vinf_arr_vec[:]),       # connect v-infinity vector to magnitude
            vinf_arr_mag[0] <= self.vinf_arr,               # magnitude upper-bounded
        ]
        
        constraint_t0 = [xs[0,7] == 0.0]
        constraints_tf = [self.tof_bounds[0] <= xs[-1,7],
                          xs[-1,7] <= self.tof_bounds[1]]
        constraints_s = [self.s_bounds[0] <= us[i,3] for i in range(Nseg)] + [us[i,3] <= self.s_bounds[1] for i in range(Nseg)]
        
        constraints_control = [
            gs[i,0] - self.Tmax * np.exp(-xbar[i,6]) * (1 - (xs[i,6] - xbar[i,6])) <= zetas[i]
            for i in range(Nseg)
        ]

        convex_problem = cp.Problem(
            cp.Minimize(objective_func),
            constraints_objsoc + constraints_dyn + constraints_trustregion +\
            constraints_initial + constraints_final + constraints_vinf_mag +constraints_control +\
            constraint_t0 + constraints_tf + constraints_s)
        convex_problem.solve(solver = self.solver, verbose = self.verbose_solver)
        self.cp_status = convex_problem.status
        return xs.value, us.value, gs.value, xis_dyn.value, xis.value, zetas.value
    
    def evaluate_nonlinear_constraints(self, xs, us, gs):
        """Evaluate nonlinear constraints
        
        Returns:
            (tuple): tuple of 1D arrays of nonlinear equality and inequality constraints
        """
        g_eq = xs[-1,0:6] - self.target_final.target_state(xs[-1,7])
        h_ineq = np.array([
            max(gs[i,0] - self.Tmax * np.exp(-xs[i,6]), 0.0) for i in range(self.N-1)
        ])
        return g_eq, h_ineq
