"""Test rendezvous with moving target"""

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pykep as pk

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import scocp


def example_pl2pl(get_plot=False):
    """Test SCP continuous transfer with log-mass dynamics"""
    # define canonical parameters
    GM_SUN = pk.MU_SUN           # Sun GM, m^3/s^-2
    MSTAR  = 800.0               # reference spacecraft mass
    ISP    = 3000.0              # specific impulse, s
    THRUST = 0.2                 # max thrust, kg.m/s^2
    G0     = 9.81                # gravity at surface, m/s^2

    DU = pk.AU                   # length scale set to Sun-Earth distance, m
    VU = np.sqrt(GM_SUN / DU)    # velocity scale, m/s
    TU = DU / VU                 # time scale, s

    canonical_scales = scocp.CanonicalScales(MSTAR, GM_SUN, DU)

    # define canonical spacecraft parameters
    m0 = 1.0                                                     # initial mass, in MU
    isp = canonical_scales.isp_si2canonical(ISP)                 # canonical specific impulse, TU
    Tmax = canonical_scales.thrust_si2canonical(THRUST)          # canonical max thrust
    cex = isp * G0*(TU**2/DU)             # canonical exhaust velocity
    print(f"\nCanonical isp: {isp:1.4e} TU, cex: {cex:1.4e} DU/TU, tmax: {Tmax:1.4e} MU.DU/TU^2")

    # define initial and final planets
    pl0 = pk.planet.jpl_lp('earth')
    plf = pk.planet.jpl_lp('mars')

    t0_mjd2000_bounds = [1100.0, 1200.0]    # initial epoch in mjd2000
    TU2DAY = TU / 86400.0  # convert non-dimensional time to elapsed time in days

    # this is the non-dimentional time integrator for solving the OCP
    integrator_01domain = scocp.ScipyIntegrator(
        nx=8,
        nu=4,
        n_gamma=1,
        rhs=scocp.control_rhs_twobody_logmass_freetf,
        rhs_stm=scocp.control_rhs_twobody_logmass_freetf_stm,
        impulsive=False,
        args=((canonical_scales.mu, cex), [0.0,0.0,0.0,1.0,0.0]),
        method='DOP853', reltol=1e-12, abstol=1e-12
    )

    # define transfer problem discretization
    tf_bounds = np.array([100.0, 500.0]) / TU2DAY
    t0_guess = 0.0
    tof_guess = 250.0 / TU2DAY
    N = 30
    s_bounds = [0.01*tof_guess, 10*tof_guess]

    # max v-infinity vector magnitudes
    vinf_dep = 1e3 / VU     # 1000 m/s
    vinf_arr = 500 / VU     # 500 m/s

    # create problem
    problem = scocp.scocp_pl2pl(
        integrator_01domain,
        canonical_scales,
        pl0,
        plf,
        m0,
        Tmax,
        cex,
        N,
        t0_mjd2000_bounds,
        tf_bounds,
        s_bounds,
        vinf_dep,
        vinf_arr,
        weight = 100.0,
    )

    # create initial guess
    print(f"Preparing initial guess...")
    xbar, ubar, gbar = problem.get_initial_guess(t0_guess, tof_guess)
    geq_nl_ig, sols_ig = problem.evaluate_nonlinear_dynamics(xbar, ubar, gbar, steps=5) # evaluate initial guess
    
    # plot initial and final orbits
    initial_orbit_states = problem.get_initial_orbit()
    final_orbit_states = problem.get_final_orbit()

    # setup algorithm & solve
    tol_feas = 1e-10
    tol_opt = 1e-6
    algo = scocp.SCvxStar(problem, tol_opt=tol_opt, tol_feas=tol_feas, rho1=1e-8, r_bounds=[1e-10, 10.0])
    xopt, uopt, gopt, yopt, sols, summary_dict = algo.solve(
        xbar,
        ubar,
        gbar,
        maxiter = 200,
        verbose = True
    )
    assert summary_dict["status"] == "Optimal"
    assert summary_dict["chi"][-1] <= tol_feas
    print(f"Initial guess TOF: {tof_guess*TU2DAY:1.4f}d --> Optimized TOF: {xopt[-1,7]*TU2DAY:1.4f}d (bounds: {tf_bounds[0]*TU2DAY:1.4f}d ~ {tf_bounds[1]*TU2DAY:1.4f}d)")
    x0 = problem.target_initial.target_state(xopt[0,7])
    xf = problem.target_final.target_state(xopt[-1,7])

    # evaluate v-infinity vectors
    vinf_dep_vec, vinf_arr_vec = yopt[0:3], yopt[3:6]
    print(f"||vinf_dep|| = {np.linalg.norm(vinf_dep_vec)*VU:1.4f} m/s (max: {vinf_dep*VU:1.4f} m/s), ||vinf_arr|| = {np.linalg.norm(vinf_arr_vec)*VU:1.4f} m/s (max: {vinf_arr*VU:1.4f} m/s)")

    # evaluate nonlinear violations
    geq_nl_opt, sols = problem.evaluate_nonlinear_dynamics(xopt, uopt, gopt, steps=20)
    print(f"Max dynamics constraint violation: {np.max(np.abs(geq_nl_opt)):1.4e}")
    assert np.max(np.abs(geq_nl_opt)) <= tol_feas
    
    # evaluate solution
    if (get_plot is True) and (summary_dict["status"] != "CPFailed"):
        # plot results
        fig = plt.figure(figsize=(12,7))
        ax = fig.add_subplot(2,3,1,projection='3d')
        ax.set(xlabel="x", ylabel="y", zlabel="z")
        for (_ts, _ys) in sols_ig:
            ax.plot(_ys[:,0], _ys[:,1], _ys[:,2], '--', color='grey')
        for (_ts, _ys) in sols:
            ax.plot(_ys[:,0], _ys[:,1], _ys[:,2], 'b-')
            _us_zoh = scocp.zoh_controls(problem.times, uopt, _ts)
            ax.quiver(_ys[:,0], _ys[:,1], _ys[:,2], _us_zoh[:,0], _us_zoh[:,1], _us_zoh[:,2], color='r', length=1.0)

        ax.scatter(x0[0], x0[1], x0[2], marker='x', color='k', label='Initial state')
        ax.scatter(xf[0], xf[1], xf[2], marker='o', color='k', label='Final state')
        ax.plot(initial_orbit_states[1][:,0], initial_orbit_states[1][:,1], initial_orbit_states[1][:,2], 'k-', lw=0.3)
        ax.plot(final_orbit_states[1][:,0], final_orbit_states[1][:,1], final_orbit_states[1][:,2], 'k-', lw=0.3)
        ax.set_aspect('equal')
        ax.legend()

        ax_m = fig.add_subplot(2,3,2)
        ax_m.grid(True, alpha=0.5)
        for (_ts, _ys) in sols:
            ax_m.plot(_ys[:,7]*canonical_scales.TU2DAY, np.exp(_ys[:,6]), 'b-')
        ax_m.axhline(np.exp(sols[-1][1][-1,6]), color='r', linestyle='--')
        ax_m.text(xopt[0,7]*canonical_scales.TU2DAY, 0.01 + np.exp(sols[-1][1][-1,6]), f"m_f = {np.exp(sols[-1][1][-1,6]):1.4f}", color='r')
        ax_m.set(xlabel="Time, days", ylabel="Mass")
        #ax_m.legend()

        ax_u = fig.add_subplot(2,3,3)
        ax_u.grid(True, alpha=0.5)
        ax_u.step(xopt[:,7]*canonical_scales.TU2DAY, np.concatenate((gopt[:,0], [0.0])), label="Control", where='post', color='k')
        for idx, (_ts, _ys) in enumerate(sols):
            ax_u.plot(_ys[:,7]*canonical_scales.TU2DAY, Tmax/np.exp(_ys[:,6]), color='r', linestyle=':', label="Max accel." if idx == 0 else None)
        ax_u.set(xlabel="Time, days", ylabel="Acceleration")
        ax_u.legend()

        iters = np.arange(len(summary_dict["DeltaJ"]))
        ax_DeltaJ = fig.add_subplot(2,3,4)
        ax_DeltaJ.grid(True, alpha=0.5)
        algo.plot_DeltaJ(ax_DeltaJ, summary_dict)
        ax_DeltaJ.axhline(tol_opt, color='k', linestyle='--', label='tol_opt')
        ax_DeltaJ.legend()

        ax_DeltaL = fig.add_subplot(2,3,5)
        ax_DeltaL.grid(True, alpha=0.5)
        algo.plot_chi(ax_DeltaL, summary_dict)
        ax_DeltaL.axhline(tol_feas, color='k', linestyle='--', label='tol_feas')
        ax_DeltaL.legend()

        # ax_J0 = fig.add_subplot(2,3,6)
        # ax_J0.grid(True, alpha=0.5)
        # algo.plot_J0(ax_J0, summary_dict)
        # ax_J0.legend()
        ax = fig.add_subplot(2,3,6)
        for (_ts, _ys) in sols_ig:
            ax.plot(_ts, _ys[:,7]*canonical_scales.TU2DAY, '--', color='grey')
        for (_ts, _ys) in sols:
            ax.plot(_ts, _ys[:,7]*canonical_scales.TU2DAY, marker="o", ms=2, color='k')
        ax.grid(True, alpha=0.5)
        ax.set(xlabel="tau", ylabel="Time, days")

        plt.tight_layout()
        fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots/example_pl2pl.png"), dpi=300)
    return


if __name__ == "__main__":
    example_pl2pl(get_plot=True)
    plt.show()