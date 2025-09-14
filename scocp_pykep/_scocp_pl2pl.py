"""Planet-to-planet optimal control problem"""

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


class PlanetTarget:
    """Moving target object from a `pykep.planet` object
    
    Args:
        planet (pykep.planet.planet): pykep planet object
        t0 (float): initial epoch in mjd2000
        DU (float): distance unit in km
        TU (float): time unit to convert non-dimensional time to elapsed time in days
        mu (float): gravitational parameter in km^3/s^2
    """
    def __init__(self, planet, t0_mjd2000: float, TU2DAY: float, DU: float, VU: float, mu: float):
        self.planet = planet
        self.t0_min = t0_mjd2000
        self.TU2DAY = TU2DAY
        self.DU = DU
        self.VU = VU
        self.mu = mu
        return
    
    def target_state(self, t: float) -> np.ndarray:
        """Evaluate planet state at time `epoch = self.t0_min + t * self.TU` where `epoch` is in mjd2000
        
        Args:
            t (float): elapsed non-dimensional time since `self.t0_min` in units `self.TU`
        
        Returns:
            (np.array): Cartesian state vector of the target planet at time `epoch`
        """
        r, v = self.planet.eph(self.t0_min + t*self.TU2DAY)
        return np.concatenate((np.array(r)/self.DU, np.array(v)/self.VU))
    
    def target_state_derivative(self, t: float) -> np.ndarray:
        """Evaluate planet state derivative at time `self.t0_min + t * self.TU`
        
        Args:
            t (float): elapsed non-dimensional time since `self.t0_min` in units `self.TU`
        
        Returns:
            (np.array): Cartesian state derivative of the target planet at time `epoch`
        """
        state = self.target_state(t)
        return rhs_twobody(t, state, self.mu)
    

class scocp_pl2pl_logmass(ContinuousControlSCOCP):
    """Free-time continuous rendezvous problem class with log-mass dynamics
    
    Note the ordering expected for the state and the control vectors: 

    state = [x,y,z,vx,vy,vz,log(mass),t]
    u = [ax, ay, az, s, Gamma] where s is the dilation factor, Gamma is the control magnitude (at convergence)
    
    Args:
        integrator (scocp.ScipyIntegrator or scocp.HeyokaIntegrator): integrator object
        p0 (pykep.planet.planet): initial planet object
        pf (pykep.planet.planet): final planet object
        mass (float): initial mass in kg
        max_thrust (float): thrust magnitude in Newtons
        isp (float): specific impulse in seconds
        nseg (int): number of segments (s.t. `N = nseg + 1`)
        t0_mjd2000 (float): initial epoch in mjd2000
        tf_bounds (tuple[float, float]): bounds on final time in days
        s_bounds (tuple[float, float]): control dilation factor bounds in days
        vinf_dep (float): max v-infinity vector magnitude at departure, in km/s
        vinf_arr (float): max v-infinity vector magnitude at arrival, in km/s
        mass_scaling (float): scaling factor for mass, in kg. If `None`, then `mass_scaling = mass`.
        r_scaling (float): scaling factor for distance, in m
        v_scaling (float): scaling factor for velocity, in m/s
        g0 (float): gravitaty acceleration at Earth surface, in m/s^2
        uniform_dilation (bool): if `True`, then the dilation factor is uniform, otherwise it is variable
    """
    def __init__(
        self,
        integrator,
        p0,
        pf,
        mass = 1500.0,
        mu_SI = pk.MU_SUN,
        max_thrust = 0.45,
        isp = 3000.0,
        nseg: int = 30,
        t0_bounds: tuple[float, float] = [6700.0, 6800.0],
        tf_bounds: tuple[float, float] = [6900.0, 7700.0],
        s_bounds: tuple[float, float] = [1.0, 5e3],
        vinf_dep: float = 3.0,
        vinf_arr: float = 0.0,
        mass_scaling = None,
        r_scaling = pk.AU,
        v_scaling = pk.EARTH_VELOCITY,
        g0 = pk.G0,
        uniform_dilation = True,
        *args,
        **kwargs
    ):
        # define problem parameters and inherit parent class
        N = nseg + 1
        ng = 12
        ny = 6              # v-infinity vectors
        nh = N - 1
        augment_Gamma = True
        times = np.linspace(0, 1, N)
        super().__init__(integrator, times, ng=ng, nh=nh, ny=ny, augment_Gamma=augment_Gamma, *args, **kwargs)

        # scaling parameters
        if mass_scaling is None:
            self.mass_scaling = 1.0 * mass
        else:
            self.mass_scaling = 1.0 * mass_scaling
        self.r_scaling    = r_scaling
        self.v_scaling    = v_scaling
        self.t_scaling    = r_scaling / v_scaling
        self.TU2DAY       = self.t_scaling / 86400.0
        self.mu           = mu_SI / (self.v_scaling**2 * self.r_scaling)

        # define additional attributes
        self.p0         = p0
        self.pf         = pf
        self.mass0      = mass / self.mass_scaling
        self.max_thrust = max_thrust * (1/self.mass_scaling)*(self.t_scaling**2/self.r_scaling)
        self.cex        = isp * g0 * (self.t_scaling/self.r_scaling)
        self.t0_min     = t0_bounds[0]
        self.t0_bounds  = [(t0_bounds[0] - self.t0_min) / self.TU2DAY,
                           (t0_bounds[1] - self.t0_min) / self.TU2DAY]
        self.tf_bounds  = np.array(tf_bounds) / self.TU2DAY
        # self.tf_bounds  = [(tf_bounds[0] - self.t0_min) / self.TU2DAY,
        #                    (tf_bounds[1] - self.t0_min) / self.TU2DAY]
        self.s_bounds   = np.array(s_bounds) / self.TU2DAY
        self.vinf_dep   = vinf_dep * 1e3 / self.v_scaling
        self.vinf_arr   = vinf_arr * 1e3 / self.v_scaling
        self.uniform_dilation = uniform_dilation

        # set initial state based on initial planet state
        r0_dim, v0_dim = self.p0.eph(self.t0_min)
        self.x0 = np.zeros(7)
        self.x0[0:3] = np.array(r0_dim)/self.r_scaling
        self.x0[3:6] = np.array(v0_dim)/self.v_scaling
        self.x0[6] = np.log(self.mass0)

        # construct initial target object
        self.target_initial = PlanetTarget(
            p0,
            t0_mjd2000 = self.t0_min,
            TU2DAY = self.TU2DAY,
            DU = self.r_scaling,
            VU = self.v_scaling,
            mu = self.mu,
        )

        # construct final target object
        self.target_final = PlanetTarget(
            pf,
            t0_mjd2000 = self.t0_min,
            TU2DAY = self.TU2DAY,
            DU = self.r_scaling,
            VU = self.v_scaling,
            mu = self.mu,
        )

        # setup storage for v-infinity vectors
        self.vinf_dep_vec = np.zeros(3,)
        self.vinf_arr_vec = np.zeros(3,)
        return
    
    def get_initial_orbit(self, steps: int=100):
        """Get state history of initial orbit over one period
        
        Args:
            steps (int): number of steps
        
        Returns:
            (tuple): tuple of 1D array of time and `(steps,6)` array ofstate history
        """
        x0 = self.target_initial.target_state(0.0)
        oe0 = rv2mee(np.array(x0), self.mu)
        period = 2*np.pi*np.sqrt(oe0[0]**3 / self.mu)
        t_eval = np.linspace(0, period, steps)
        states = np.zeros((steps, 6))
        for i, t in enumerate(t_eval):
            #states[i,0:3], states[i,3:6] = pk.propagate_lagrangian(self.x0[0:3], self.x0[3:6], t, mu=self.mu)
            states[i,:] = keplerder_nostm(self.mu, self.x0, 0.0, t)
        return t_eval, states
    
    def get_final_orbit(self, steps: int=100):
        """Get state history of final orbit over one period
        
        Args:
            steps (int): number of steps
        
        Returns:
            (tuple): tuple of 1D array of time and `(steps,6)` array of state history
        """
        xf0 = self.target_final.target_state(0.0)
        oef = rv2mee(xf0, self.mu)
        period = 2*np.pi*np.sqrt(oef[0]**3 / self.mu)
        t_eval = np.linspace(0, period, steps)
        states = np.zeros((steps, 6))
        for i, t in enumerate(t_eval):
            #states[i,0:3], states[i,3:6] = pk.propagate_lagrangian(xf0[0:3], xf0[3:6], t, mu=self.mu)
            states[i,:] = keplerder_nostm(self.mu, xf0, 0.0, t)
        return t_eval, states
    
    def get_initial_guess(self, t0_guess: float, tf_guess: float, Nrev_guess: float = 1.0):
        """Construct initial guess based on linear interpolation along orbital elements
        
        Args:
            t0_guess_si (float): guess for loiter time
            tf_guess_si (float): guess for final arrival time
            Nrev_guess (float): guess for number of revolutions
        
        Returns:
            (tuple): tuple of `(N,8)` array of state history, `(N-1,4)` array of control history, and `(N-1,1)` array of constraint history
        """
        # re-scale time
        t0_guess = t0_guess / self.TU2DAY
        tf_guess = tf_guess / self.TU2DAY
        
        # initial orbital elements
        x0 = self.target_initial.target_state(t0_guess)
        oe0 = rv2mee(np.array(x0), self.mu)

        # final orbital elements
        xf_guess = self.target_final.target_state(tf_guess)
        oef = rv2mee(xf_guess, self.mu)
        
        elements = np.concatenate((
            np.linspace(oe0[0], oef[0], self.N).reshape(-1,1),
            np.linspace(oe0[1], oef[1], self.N).reshape(-1,1),
            np.linspace(oe0[2], oef[2], self.N).reshape(-1,1),
            np.linspace(oe0[3], oef[3], self.N).reshape(-1,1),
            np.linspace(oe0[4], oef[4], self.N).reshape(-1,1),
            np.linspace(oe0[5], oe0[5] + 2*np.pi * Nrev_guess, self.N).reshape(-1,1),
        ), axis=1)

        elements[:,5] = np.linspace(oe0[5], oef[5]+2*np.pi, self.N)
        xbar = np.zeros((self.N,8))
        xbar[:,0:6]  = np.array([mee2rv(E,self.mu) for E in elements])
        xbar[:,6]    = np.log(np.linspace(1.0, 0.5, self.N))  # initial guess for log-mass
        xbar[0,0:6]  = x0[0:6]                # overwrite initial state
        xbar[0,6]    = np.log(self.mass0)
        xbar[-1,0:6] = xf_guess[0:6]          # overwrite final state
        xbar[:,7]    = np.linspace(t0_guess, tf_guess, self.N)        # initial guess for time

        sbar_initial = (tf_guess - t0_guess) * np.ones((self.N-1,1))
        # ubar = np.concatenate((np.divide(np.diff(xbar[:,3:6], axis=0), np.diff(times_guess)[:,None]), sbar_initial), axis=1)
        ubar = np.concatenate((np.zeros((self.N-1,3)), sbar_initial), axis=1)
        vbar = np.sum(ubar[:,0:3], axis=1).reshape(-1,1)
        return xbar, ubar, vbar
    
    def evaluate_objective(self, xs, us, gs, ys=None):
        """Evaluate the objective function"""
        return -xs[-1,6]
    
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
        N,nx = xbar.shape
        _,nu = ubar.shape
        Nseg = N - 1
        
        xs      = cp.Variable((N, nx), name='state')
        us      = cp.Variable((Nseg, nu), name='control')
        gs      = cp.Variable((Nseg, 1), name='Gamma')
        ys      = cp.Variable((self.ny,), name='v-infinity vectors')
        xis_dyn = cp.Variable((Nseg,nx), name='xi_dyn')     # slack for dynamics
        xis     = cp.Variable((self.ng,), name='xi')        # slack for target state
        zetas   = cp.Variable((Nseg,), name='zeta')         # slack for non-convex inequality
        
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
        
        # constraints on dynamics for state and control
        constraints_dyn = [
            xs[i+1,:] == self.Phi_A[i,:,:] @ xs[i,:] + self.Phi_B[i,:,0:4] @ us[i,:] + self.Phi_B[i,:,4] * gs[i,:] + self.Phi_c[i,:] + xis_dyn[i,:]
            for i in range(Nseg)
        ]
        constraints_control = [gs[i,0] - self.max_thrust * np.exp(-xbar[i,6]) * (1 - (xs[i,6] - xbar[i,6])) <= zetas[i] for i in range(Nseg)]

        if self.uniform_dilation:
            constraints_dilation = [us[i,3] == us[0,3] for i in range(1,Nseg)]
        else:
            constraints_dilation = []

        # trust region constraints 
        constraints_trustregion = [
            xs[i,0:7] - xbar[i,0:7] <= self.trust_region_radius_x for i in range(N)
        ] + [
            xs[i,0:7] - xbar[i,0:7] >= -self.trust_region_radius_x for i in range(N)
        ]
        if self.trust_region_radius_u is not None:
            constraints_trustregion += [
                us[i,:] - ubar[i,:] <=  self.trust_region_radius_u for i in range(Nseg)
            ] + [
                us[i,:] - ubar[i,:] >= -self.trust_region_radius_u for i in range(Nseg)
            ]

        # boundary conditions
        constraints_boundary = [
            xs[0,0:6] - np.concatenate((np.zeros((3,3)), np.eye(3))) @ ys[0:3] \
                - self.target_initial.target_state(xbar[0,7]) - self.target_initial.target_state_derivative(xbar[0,7]) * (xs[0,7] - xbar[0,7]) == xis[0:6],
            xs[0,6] == np.log(self.mass0),
            xs[-1,0:6] - np.concatenate((np.zeros((3,3)), np.eye(3))) @ ys[3:6] \
                - self.target_final.target_state(xbar[-1,7]) - self.target_final.target_state_derivative(xbar[-1,7]) * (xs[-1,7] - xbar[-1,7]) == xis[6:12]
        ]
        constraints_vinf_mag = [
            cp.SOC(self.vinf_dep, ys[0:3]),
            cp.SOC(self.vinf_arr, ys[3:6]),
        ]
        
        # constraints on times
        if abs(self.t0_bounds[1] - self.t0_bounds[0]) < 1e-12:
            constraints_t0 = [xs[0,7] == self.t0_bounds[0]]
        else:
            constraints_t0 = [self.t0_bounds[0] <= xs[0,7],
                              xs[0,7] <= self.t0_bounds[1]]
        constraints_tf = [self.tf_bounds[0] <= xs[-1,7],
                          xs[-1,7] <= self.tf_bounds[1]]
        constraints_s = [self.s_bounds[0] <= us[i,3] for i in range(Nseg)] + [us[i,3] <= self.s_bounds[1] for i in range(Nseg)]

        convex_problem = cp.Problem(
            cp.Minimize(objective_func),
            constraints_objsoc + constraints_dyn + constraints_trustregion +\
            constraints_boundary + constraints_vinf_mag + constraints_control +\
            constraints_t0 + constraints_tf + constraints_s + constraints_dilation)
        convex_problem.solve(solver = self.solver, verbose = self.verbose_solver)
        self.cp_status = convex_problem.status
        return xs.value, us.value, gs.value, ys.value, xis_dyn.value, xis.value, zetas.value
    
    def evaluate_nonlinear_constraints(self, xs, us, gs, ys):
        """Evaluate nonlinear constraints
        
        Returns:
            (tuple): tuple of 1D arrays of nonlinear equality and inequality constraints
        """
        g_eq = np.concatenate((
            xs[0,0:3] - self.target_initial.target_state(xs[0,7])[0:3],
            xs[0,3:6] - ys[0:3] - self.target_initial.target_state(xs[0,7])[3:6],
            xs[-1,0:3] - self.target_final.target_state(xs[-1,7])[0:3],
            xs[-1,3:6] - ys[3:6] - self.target_final.target_state(xs[-1,7])[3:6],
        ))
        h_ineq = np.array([
            max(gs[i,0] - self.max_thrust * np.exp(-xs[i,6]), 0.0) for i in range(self.N-1)
        ])
        return g_eq, h_ineq

    def process_solution(
        self,
        solution,
        r_scaling = None,
        v_scaling = None,
        t_scaling = None,
        a_scaling = None,
        mass_scaling = None,
        convert_t_to_mjd2000 = True,
        dense_output = False,
        dt_day = None,
        return_acceleration_control = False,
    ):
        """Get times, states, and controls from solution object
        
        Args:
            solution (scocp.Solution): solution object returned by `scocp.SCvxStar.solve()` for this problem
            r_scaling (float): scaling factor for distance, defaults to `self.r_scaling`
            v_scaling (float): scaling factor for velocity, defaults to `self.v_scaling`
            t_scaling (float): scaling factor for time, defaults to `self.t_scaling`
            a_scaling (float): scaling factor for acceleration, defaults to `r_scaling/t_scaling**2`
            mass_scaling (float): scaling factor for mass, defaults to `self.mass_scaling`
            convert_t_to_mjd2000 (bool): if `True`, then convert times to mjd2000, defaults to `True`
            dense_output (bool): if `True`, then return dense output, defaults to `False`
            dt_day (float): if `dense_output` is `True`, then this is the time step in days for sampling the solution, defaults to `None`
            return_acceleration_control (bool): if `True`, then return acceleration control, otherwise return thrust controls

        Returns:
            (tuple): tuple of 1D array of times, `(N,7)` array of states, and `(N-1,3)` array of controls
        """
        if r_scaling is None:
            r_scaling = self.r_scaling
        if v_scaling is None:
            v_scaling = self.v_scaling
        if t_scaling is None:
            t_scaling = self.t_scaling
        if mass_scaling is None:
            mass_scaling = self.mass_scaling
        if a_scaling is None:
            a_scaling = r_scaling/t_scaling**2

        if dense_output is False:
            if convert_t_to_mjd2000:
                times = solution.x[:,7] * t_scaling + self.t0_min
            else:
                times = solution.x[:,7] * t_scaling

            states = np.concatenate((
                solution.x[:,0:3] * r_scaling,
                solution.x[:,3:6] * v_scaling,
                np.exp(solution.x[:,6]).reshape(-1,1) * mass_scaling,
            ), axis=1)
            if return_acceleration_control:
                controls = solution.u[:,0:3] * a_scaling       # acceleration control
            else:
                controls = np.divide(solution.u[:,0:3] * a_scaling, states[:,6].reshape(-1,1))

        else:
            if dt_day is None:
                steps = None
            else:
                dt_node_min = np.min(np.diff(solution.x[:,7]))
                steps = 100 #int(np.ceil(dt_node_min*self.TU2DAY / dt_day*86400))
            _, sols = self.evaluate_nonlinear_dynamics(solution.x, solution.u, solution.v, steps=steps)
            times = []
            states = []
            controls = []
            for (idx, (_ts, _ys)) in enumerate(sols):
                _states = np.concatenate((
                    _ys[:,0:3] * r_scaling,
                    _ys[:,3:6] * v_scaling,
                    np.exp(_ys[:,6]).reshape(-1,1) * mass_scaling,
                ), axis=1)
                times.append(np.array(_ts) * t_scaling)
                states.append(_states)
                if return_acceleration_control:
                    controls.append(np.tile(solution.u[idx,0:3] * a_scaling, (len(_ts), 1)))
                else:
                    controls.append(np.tile(solution.u[idx,0:3] * a_scaling, (len(_ts), 1)) / _states[:,6].reshape(-1,1))
            times = np.concatenate(times)
            states = np.concatenate(states)
            controls = np.concatenate(controls)
        return times, states, controls


class scocp_pl2pl(ContinuousControlSCOCP):
    """Free-time continuous rendezvous problem class with mass dynamics
    
    Note the ordering expected for the state and the control vectors: 
        * state = `[x,y,z,vx,vy,vz,mass,t]`
        * control = `[ux,uy,uz,s,v]` where:
            * `s` is the dilation factor,
            * `ux,uy,uz` is the thrust throttle, bounded between -1 and 1,
            * `v` is the control magnitude (at convergence), bounded between 0 and 1

    The objective should be one of the following:
        * `"mf"`: maximize final mass (i.e. minimum fuel problem)
        * `"tf"`: minimize arrival time (i.e. earliest arrival)
        * `"tof"`: minimize time of flight (i.e. minimum time in transit)

    The solution of this optimal control problem obeys the stark model dynamics, i.e. we assume a zeroth-order hold on the thrust inputs.
    
    Args:
        integrator (scocp.ScipyIntegrator or scocp.HeyokaIntegrator): integrator object
        p0 (pykep.planet.planet): initial planet object
        pf (pykep.planet.planet): final planet object
        mass (float): initial mass in kg
        mu_SI (float): gravitational parameter of the central body, in m^3/s^2
        nseg (int): number of segments (s.t. `N = nseg + 1`)
        t0_mjd2000 (float): initial epoch in mjd2000
        tf_bounds (tuple[float, float]): bounds on final time in days
        s_bounds (tuple[float, float]): control dilation factor bounds in days
        vinf_dep (float): max v-infinity vector magnitude at departure, in km/s
        vinf_arr (float): max v-infinity vector magnitude at arrival, in km/s
        mass_scaling (float): scaling factor for mass, in kg. If `None`, then `mass_scaling = mass`.
        r_scaling (float): scaling factor for distance, in m
        v_scaling (float): scaling factor for velocity, in m/s
        uniform_dilation (bool): if `True`, then the dilation factor is uniform, otherwise it is variable
        objective_type (str): objective type, one of `"mf"`, `"tf"`, or `"tof"`
    """
    def __init__(
        self,
        integrator,
        p0,
        pf,
        mass = 1500.0,
        mu_SI = pk.MU_SUN,
        nseg: int = 30,
        t0_bounds: tuple[float, float] = [6700.0, 6800.0],
        tf_bounds: tuple[float, float] = [6900.0, 7700.0],
        s_bounds: tuple[float, float] = [1.0, 5e3],
        vinf_dep: float = 3.0,
        vinf_arr: float = 0.0,
        mass_scaling = None,
        r_scaling = pk.AU,
        v_scaling = pk.EARTH_VELOCITY,
        uniform_dilation = True,
        objective_type = "mf",
        *args,
        **kwargs
    ):
        objective_types = ["mf", "tf", "tof"]
        assert objective_type in objective_types, f"objective_type must be one of {objective_types}"
        self.objective_type = objective_type

        # define problem parameters and inherit parent class
        N = nseg + 1
        ng = 12
        ny = 6              # v-infinity vectors
        nh = 0
        augment_Gamma = True
        times = np.linspace(0, 1, N)
        super().__init__(integrator, times, ng=ng, nh=nh, ny=ny, augment_Gamma=augment_Gamma, *args, **kwargs)

        # scaling parameters
        if mass_scaling is None:
            self.mass_scaling = 1.0 * mass
        else:
            self.mass_scaling = 1.0 * mass_scaling
        self.r_scaling    = r_scaling
        self.v_scaling    = v_scaling
        self.t_scaling    = r_scaling / v_scaling
        self.TU2DAY       = self.t_scaling / 86400.0
        self.mu           = mu_SI / (self.v_scaling**2 * self.r_scaling)

        # define additional attributes
        self.p0         = p0
        self.pf         = pf
        self.mass0      = mass / self.mass_scaling
        self.t0_min     = t0_bounds[0]
        self.t0_bounds  = [(t0_bounds[0] - self.t0_min) / self.TU2DAY,
                           (t0_bounds[1] - self.t0_min) / self.TU2DAY]
        self.tf_bounds  = [(tf_bounds[0] - self.t0_min) / self.TU2DAY,
                           (tf_bounds[1] - self.t0_min) / self.TU2DAY]
        self.s_bounds   = np.array(s_bounds) / self.TU2DAY
        self.vinf_dep   = vinf_dep * 1e3 / self.v_scaling
        self.vinf_arr   = vinf_arr * 1e3 / self.v_scaling
        self.uniform_dilation = uniform_dilation

        # set initial state based on initial planet state
        r0_dim, v0_dim = self.p0.eph(self.t0_min)
        self.x0 = np.zeros(6)
        self.x0[0:3] = np.array(r0_dim)/self.r_scaling
        self.x0[3:6] = np.array(v0_dim)/self.v_scaling

        # construct initial target object
        self.target_initial = PlanetTarget(
            p0,
            t0_mjd2000 = self.t0_min,
            TU2DAY = self.TU2DAY,
            DU = self.r_scaling,
            VU = self.v_scaling,
            mu = self.mu,
        )

        # construct final target object
        self.target_final = PlanetTarget(
            pf,
            t0_mjd2000 = self.t0_min,
            TU2DAY = self.TU2DAY,
            DU = self.r_scaling,
            VU = self.v_scaling,
            mu = self.mu,
        )

        # setup storage for v-infinity vectors
        self.vinf_dep_vec = np.zeros(3,)
        self.vinf_arr_vec = np.zeros(3,)
        return
    
    def get_initial_orbit(self, steps: int=100):
        """Get state history of initial orbit over one period
        
        Args:
            steps (int): number of steps
        
        Returns:
            (tuple): tuple of 1D array of time and `(steps,6)` array ofstate history
        """
        x0 = self.target_initial.target_state(0.0)
        oe0 = rv2mee(np.array(x0), self.mu)
        period = 2*np.pi*np.sqrt(oe0[0]**3 / self.mu)
        t_eval = np.linspace(0, period, steps)
        states = np.zeros((steps, 6))
        for i, t in enumerate(t_eval):
            #states[i,0:3], states[i,3:6] = pk.propagate_lagrangian(self.x0[0:3], self.x0[3:6], t, mu=self.mu)
            states[i,:] = keplerder_nostm(self.mu, self.x0, 0.0, t)
        return t_eval, states
    
    def get_final_orbit(self, steps: int=100):
        """Get state history of final orbit over one period
        
        Args:
            steps (int): number of steps
        
        Returns:
            (tuple): tuple of 1D array of time and `(steps,6)` array of state history
        """
        xf0 = self.target_final.target_state(0.0)
        oef = rv2mee(xf0, self.mu)
        period = 2*np.pi*np.sqrt(oef[0]**3 / self.mu)
        t_eval = np.linspace(0, period, steps)
        states = np.zeros((steps, 6))
        for i, t in enumerate(t_eval):
            #states[i,0:3], states[i,3:6] = pk.propagate_lagrangian(xf0[0:3], xf0[3:6], t, mu=self.mu)
            states[i,:] = keplerder_nostm(self.mu, xf0, 0.0, t)
        return t_eval, states
    
    def get_initial_guess(self, t0_guess: float, tf_guess: float, Nrev_guess: float = 1.0):
        """Construct initial guess based on linear interpolation along orbital elements
        
        Args:
            t0_guess_si (float): guess for departure time in MJD
            tf_guess_si (float): guess for final arrival time in MJD
            Nrev_guess (float): guess for number of revolutions
        
        Returns:
            (tuple): tuple of `(N,8)` array of state history, `(N-1,4)` array of control history, and `(N-1,1)` array of constraint history
        """
        # re-scale time
        t0_guess = (t0_guess - self.t0_min) / self.TU2DAY
        tf_guess = (tf_guess - self.t0_min) / self.TU2DAY

        # initial orbital elements
        x0 = self.target_initial.target_state(t0_guess)
        oe0 = rv2mee(np.array(x0), self.mu)

        # final orbital elements
        xf_guess = self.target_final.target_state(tf_guess)
        oef = rv2mee(xf_guess, self.mu)
        
        elements = np.concatenate((
            np.linspace(oe0[0], oef[0], self.N).reshape(-1,1),
            np.linspace(oe0[1], oef[1], self.N).reshape(-1,1),
            np.linspace(oe0[2], oef[2], self.N).reshape(-1,1),
            np.linspace(oe0[3], oef[3], self.N).reshape(-1,1),
            np.linspace(oe0[4], oef[4], self.N).reshape(-1,1),
            np.linspace(oe0[5], oe0[5] + 2*np.pi * Nrev_guess, self.N).reshape(-1,1),
        ), axis=1)
        xbar = np.zeros((self.N,8))
        xbar[:,0:6]  = np.array([mee2rv(E,self.mu) for E in elements])
        xbar[:,6]    = np.linspace(1.0, 0.5, self.N)                  # initial guess for log-mass
        xbar[0,0:6]  = x0[0:6]                                        # overwrite initial state
        xbar[0,6]    = self.mass0
        xbar[-1,0:6] = xf_guess[0:6]                                  # overwrite final state
        xbar[:,7]    = np.linspace(t0_guess, tf_guess, self.N)        # initial guess for time

        sbar_initial = (tf_guess - t0_guess) * np.ones((self.N-1,1))
        ubar = np.concatenate((np.zeros((self.N-1,3)), sbar_initial), axis=1)
        vbar = np.sum(ubar[:,0:3], axis=1).reshape(-1,1)
        return xbar, ubar, vbar
    
    def evaluate_objective(self, xs, us, vs, ys=None):
        """Evaluate the objective function"""
        if self.objective_type == "mf":
            return -xs[-1,6]
        elif self.objective_type == "tof":
            return xs[-1,7] - xs[0,7]
        elif self.objective_type == "tf":
            return xs[-1,7]
    
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
        N,nx = xbar.shape
        _,nu = ubar.shape
        Nseg = N - 1
        
        xs      = cp.Variable((N, nx), name='state')
        us      = cp.Variable((Nseg, nu), name='control')
        vs      = cp.Variable((Nseg, 1), name='Gamma')
        ys      = cp.Variable((self.ny,), name='v-infinity vectors')
        xis_dyn = cp.Variable((Nseg,nx), name='xi_dyn')     # slack for dynamics
        xis     = cp.Variable((self.ng,), name='xi')        # slack for target state
        
        penalty = get_augmented_lagrangian_penalty(
            self.weight,
            xis_dyn,
            self.lmb_dynamics,
            xi=xis,
            lmb_eq=self.lmb_eq,
        )
        objective_func = self.evaluate_objective(xs, us, vs, ys) + penalty
        constraints_objsoc = [cp.SOC(vs[i,0], us[i,0:3]) for i in range(N-1)]
        constraints_control = [vs[i,0] <= 1.0 for i in range(Nseg)]
        
        # constraints on dynamics for state and control
        constraints_dyn = [
            xs[i+1,:] == self.Phi_A[i,:,:] @ xs[i,:] + self.Phi_B[i,:,0:4] @ us[i,:] + self.Phi_B[i,:,4] * vs[i,:] + self.Phi_c[i,:] + xis_dyn[i,:]
            for i in range(Nseg)
        ]

        if self.uniform_dilation:
            constraints_dilation = [us[i,3] == us[0,3] for i in range(1,Nseg)]
        else:
            constraints_dilation = []

        # trust region constraints 
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

        # boundary conditions
        constraints_boundary = [
            xs[0,0:6] - np.concatenate((np.zeros((3,3)), np.eye(3))) @ ys[0:3] \
                - self.target_initial.target_state(xbar[0,7]) - self.target_initial.target_state_derivative(xbar[0,7]) * (xs[0,7] - xbar[0,7]) == xis[0:6],
            xs[0,6] == self.mass0,
            xs[-1,0:6] - np.concatenate((np.zeros((3,3)), np.eye(3))) @ ys[3:6] \
                - self.target_final.target_state(xbar[-1,7]) - self.target_final.target_state_derivative(xbar[-1,7]) * (xs[-1,7] - xbar[-1,7]) == xis[6:12]
        ]
        constraints_vinf_mag = [
            cp.SOC(self.vinf_dep, ys[0:3]),
            cp.SOC(self.vinf_arr, ys[3:6]),
        ]
        
        # constraints on times
        if abs(self.t0_bounds[1] - self.t0_bounds[0]) < 1e-12:
            constraints_t0 = [xs[0,7] == self.t0_bounds[0]]
        else:
            constraints_t0 = [self.t0_bounds[0] <= xs[0,7],
                              xs[0,7] <= self.t0_bounds[1]]
        constraints_tf = [self.tf_bounds[0] <= xs[-1,7],
                          xs[-1,7] <= self.tf_bounds[1]]
        constraints_s = [self.s_bounds[0] <= us[i,3] for i in range(Nseg)] + [us[i,3] <= self.s_bounds[1] for i in range(Nseg)]

        convex_problem = cp.Problem(
            cp.Minimize(objective_func),
            constraints_objsoc + constraints_dyn + constraints_trustregion +\
            constraints_boundary + constraints_vinf_mag + constraints_control +\
            constraints_t0 + constraints_tf + constraints_s + constraints_dilation)
        convex_problem.solve(solver = self.solver, verbose = self.verbose_solver)
        self.cp_status = convex_problem.status
        return xs.value, us.value, vs.value, ys.value, xis_dyn.value, xis.value, None
    
    def evaluate_nonlinear_constraints(self, xs, us, vs, ys):
        """Evaluate nonlinear constraints
        
        Returns:
            (tuple): tuple of 1D arrays of nonlinear equality and inequality constraints
        """
        g_eq = np.concatenate((
            xs[0,0:3] - self.target_initial.target_state(xs[0,7])[0:3],
            xs[0,3:6] - ys[0:3] - self.target_initial.target_state(xs[0,7])[3:6],
            xs[-1,0:3] - self.target_final.target_state(xs[-1,7])[0:3],
            xs[-1,3:6] - ys[3:6] - self.target_final.target_state(xs[-1,7])[3:6],
        ))
        return g_eq, np.zeros(self.nh)
    
    def pretty(self, solution):
        """Pretty print the solution"""
        print(f"\n ********* Trajectory summary ********* ")
        print(f"   Objective type       : {self.objective_type}")
        print(f"   Departure            : {self.t0_min + solution.x[0,7]*self.TU2DAY:1.4f} MJD")
        print(f"   Arrival              : {self.t0_min + solution.x[-1,7]*self.TU2DAY:1.4f} MJD")
        print(f"   TOF                  : {solution.x[-1,7]*self.TU2DAY:1.4f} days")
        print(f"   Final mass           : {solution.x[-1,6]*self.mass_scaling:1.4f} kg")
        print(f"   Departure v-infinity : {np.linalg.norm(solution.y[0:3])*self.v_scaling:1.2f} m/s")
        print(f"   Arrival v-infinity   : {np.linalg.norm(solution.y[3:6])*self.v_scaling:1.2f} m/s")
        print(f"\n")
        return

    def process_solution(
        self,
        solution,
        r_scaling = None,
        v_scaling = None,
        t_scaling = None,
        mass_scaling = None,
        convert_t_to_mjd2000 = True,
    ):
        """Get times, states, and controls from solution object, re-scaled back to SI units
        
        Args:
            solution (scocp.Solution): solution object returned by `scocp.SCvxStar.solve()` for this problem
            r_scaling (float): scaling factor for distance, defaults to `self.r_scaling`
            v_scaling (float): scaling factor for velocity, defaults to `self.v_scaling`
            t_scaling (float): scaling factor for time, defaults to `self.t_scaling`
            mass_scaling (float): scaling factor for mass, defaults to `self.mass_scaling`
            convert_t_to_mjd2000 (bool): if `True`, then convert times to mjd2000, defaults to `True`

        Returns:
            (tuple): tuple of 1D array of times, `(N,7)` array of states, and `(N-1,3)` array of controls
        """
        if r_scaling is None:
            r_scaling = self.r_scaling
        if v_scaling is None:
            v_scaling = self.v_scaling
        if t_scaling is None:
            t_scaling = self.TU2DAY
        if mass_scaling is None:
            mass_scaling = self.mass_scaling

        if convert_t_to_mjd2000:
            times = solution.x[:,7] * t_scaling + self.t0_min
        else:
            times = solution.x[:,7] * t_scaling

        states = np.concatenate((
            solution.x[:,0:3] * r_scaling,
            solution.x[:,3:6] * v_scaling,
            solution.x[:,6].reshape(-1,1) * mass_scaling,
        ), axis=1)
        controls = solution.u[:,0:3]
        v_infinities = [
            solution.y[0:3] * v_scaling,
            solution.y[3:6] * v_scaling,
        ]
        return times, states, controls, v_infinities


