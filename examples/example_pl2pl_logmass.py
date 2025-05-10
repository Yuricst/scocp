"""Test rendezvous with moving target"""

import matplotlib.pyplot as plt
import numpy as np
import pykep as pk

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import scocp
import scocp_pykep


def example_pl2pl(use_heyoka=False, get_plot=False):
    """Test SCP continuous transfer with log-mass dynamics"""
    # define canonical parameters
    GM_SUN = pk.MU_SUN           # Sun GM, m^3/s^-2
    MSTAR  = 800.0               # reference spacecraft mass
    ISP    = 3000.0              # specific impulse, s
    THRUST = 0.2                 # max thrust, kg.m/s^2
    G0     = pk.G0               # gravity at surface, m/s^2

    DU = pk.AU                   # length scale set to Sun-Earth distance, m
    VU = np.sqrt(GM_SUN / DU)    # velocity scale, m/s
    TU = DU / VU                 # time scale, s

    # define initial and final planets
    pl0 = pk.planet.jpl_lp('earth')
    plf = pk.planet.jpl_lp('mars')

    t0_mjd2000_bounds = [1100.0, 1200.0]    # initial epoch in mjd2000
    TU2DAY = TU / 86400.0                   # convert non-dimensional time to elapsed time in days

    # define transfer problem discretization
    tf_bounds = [100.0, 500.0]
    t0_guess = 0.0
    tf_guess = 250.0
    N = 30
    s_bounds = [0.01*tf_guess, 10*tf_guess]

    # max v-infinity vector magnitudes
    vinf_dep = 1e3/1e3     # 1000 m/s
    vinf_arr = 500/1e3     # 500 m/s

    # this is the non-dimentional time integrator for solving the OCP
    mu = GM_SUN / (VU**2 * DU)      # canonical gravitational constant
    cex = ISP * G0 * (TU/DU)        # canonical exhaust velocity of thruster
    if use_heyoka:
        ta_dyn, ta_dyn_aug = scocp_pykep.get_heyoka_integrator_twobody(mu, cex, tol=1e-12, verbose=True)
        integrator_01domain = scocp_pykep.HeyokaIntegrator(
            nx=8,
            nu=4,
            nv=1,
            ta=ta_dyn,
            ta_stm=ta_dyn_aug,
            impulsive=False
        )
    else:
        integrator_01domain = scocp.ScipyIntegrator(
            nx=8,
            nu=4,
            nv=1,
            rhs=scocp.control_rhs_twobody_logmass_freetf,
            rhs_stm=scocp.control_rhs_twobody_logmass_freetf_stm,
            impulsive=False,
            args=((mu, cex),                # canonical gravitational constant & exhaust velocity
                [0.0,0.0,0.0,1.0,0.0]     # place-holder for control vector: [ax,ay,az,s,v]
            ),
            method='DOP853', reltol=1e-12, abstol=1e-12
        )

    # create problem
    problem = scocp_pykep.scocp_pl2pl_logmass(
        integrator_01domain,
        pl0,
        plf,
        MSTAR,
        pk.MU_SUN,
        THRUST,
        ISP,
        N,
        t0_mjd2000_bounds,
        tf_bounds,
        s_bounds,
        vinf_dep,
        vinf_arr,
        r_scaling = pk.AU,
        v_scaling = pk.EARTH_VELOCITY,
        weight = 100.0,
    )
    # create initial guess
    print(f"Preparing initial guess...")
    xbar, ubar, vbar = problem.get_initial_guess(t0_guess, tf_guess)
    geq_nl_ig, sols_ig = problem.evaluate_nonlinear_dynamics(xbar, ubar, vbar, steps=5)   # evaluate initial guess

    # setup algorithm & solve
    tol_feas = 1e-10
    tol_opt = 1e-6
    algo = scocp.SCvxStar(problem, tol_opt=tol_opt, tol_feas=tol_feas, rho1=1e-8, r_bounds=[1e-10, 10.0])
    solution = algo.solve(
        xbar,
        ubar,
        vbar,
        maxiter = 100,
        verbose = True
    )
    xopt, uopt, vopt, yopt, sols, summary_dict = solution.x, solution.u, solution.v, solution.y, solution.sols, solution.summary_dict
    assert summary_dict["status"] == "Optimal"
    assert summary_dict["chi"][-1] <= tol_feas
    print(f"Initial guess TOF: {tf_guess*TU2DAY:1.4f}d --> Optimized TOF: {xopt[-1,7]*TU2DAY:1.4f}d (bounds: {tf_bounds[0]*TU2DAY:1.4f}d ~ {tf_bounds[1]*TU2DAY:1.4f}d)")
    x0 = problem.target_initial.target_state(xopt[0,7])
    xf = problem.target_final.target_state(xopt[-1,7])

    # evaluate v-infinity vectors
    vinf_dep_vec, vinf_arr_vec = yopt[0:3], yopt[3:6]
    print(f"||vinf_dep|| = {np.linalg.norm(vinf_dep_vec)*VU:1.4f} m/s (max: {vinf_dep*VU:1.4f} m/s), ||vinf_arr|| = {np.linalg.norm(vinf_arr_vec)*VU:1.4f} m/s (max: {vinf_arr*VU:1.4f} m/s)")

    # evaluate nonlinear violations
    geq_nl_opt, sols = problem.evaluate_nonlinear_dynamics(xopt, uopt, vopt, steps=8)
    print(f"Max dynamics constraint violation: {np.max(np.abs(geq_nl_opt)):1.4e}")
    assert np.max(np.abs(geq_nl_opt)) <= tol_feas

    # extract solution
    ts_mjd2000, states, controls = problem.process_solution(solution, dense_output=True, dt_day = 1.0)
    print(f"ts_mjd2000.shape = {ts_mjd2000.shape}")
    print(f"states.shape = {states.shape}")
    print(f"controls.shape = {controls.shape}")

    # initial and final orbits
    initial_orbit_states = problem.get_initial_orbit()
    final_orbit_states = problem.get_final_orbit()

    # fig = plt.figure(figsize=(12,5))
    # ax = fig.add_subplot(1,3,1, projection='3d')
    # ax.plot(states[:,0], states[:,1], states[:,2], 'b-')
    # ax.scatter(x0[0]*pk.AU, x0[1]*pk.AU, x0[2]*pk.AU, marker='x', color='k', label='Initial state')
    # ax.scatter(xf[0]*pk.AU, xf[1]*pk.AU, xf[2]*pk.AU, marker='o', color='k', label='Final state')
    # ax.plot(initial_orbit_states[1][:,0]*pk.AU, initial_orbit_states[1][:,1]*pk.AU, initial_orbit_states[1][:,2]*pk.AU, 'k-', lw=0.3)
    # ax.plot(final_orbit_states[1][:,0]*pk.AU, final_orbit_states[1][:,1]*pk.AU, final_orbit_states[1][:,2]*pk.AU, 'k-', lw=0.3)   
    # ax.set(xlabel="x", ylabel="y", zlabel="z")
    # ax.legend()

    # ax_mass = fig.add_subplot(1,3,2)
    # ax_mass.plot(ts_mjd2000, states[:,6])
    # ax_mass.set(xlabel="Time, days", ylabel="Mass")
    # ax_mass.grid(True, alpha=0.5)
    
    # ax_control = fig.add_subplot(1,3,3)
    # ax_control.plot(ts_mjd2000, controls[:,0], label="ux")
    # ax_control.plot(ts_mjd2000, controls[:,1], label="uy")
    # ax_control.plot(ts_mjd2000, controls[:,2], label="uz")
    # ax_control.set(xlabel="Time, days", ylabel="Control")
    # ax_control.grid(True, alpha=0.5)
    # ax_control.legend()
    # plt.tight_layout()
    
    # evaluate solution
    if (get_plot is True) and (summary_dict["status"] != "CPFailed"):
        # initial and final orbits
        initial_orbit_states = problem.get_initial_orbit()
        final_orbit_states = problem.get_final_orbit()

        # plot results
        fig = plt.figure(figsize=(12,7))
        ax = fig.add_subplot(2,3,1,projection='3d')
        ax.set(xlabel="x", ylabel="y", zlabel="z")
        for (_ts, _ys) in sols_ig:
            ax.plot(_ys[:,0], _ys[:,1], _ys[:,2], '--', color='grey')
        for (_ts, _ys) in sols:
            ax.plot(_ys[:,0], _ys[:,1], _ys[:,2], 'b-')
            _us_zoh = scocp.zoh_controls(problem.times, uopt, _ts)
            ax.quiver(_ys[:,0], _ys[:,1], _ys[:,2], _us_zoh[:,0], _us_zoh[:,1], _us_zoh[:,2], color='r', length=2.0)

        ax.scatter(x0[0], x0[1], x0[2], marker='x', color='k', label='Initial state')
        ax.scatter(xf[0], xf[1], xf[2], marker='o', color='k', label='Final state')
        ax.plot(initial_orbit_states[1][:,0], initial_orbit_states[1][:,1], initial_orbit_states[1][:,2], 'k-', lw=0.3)
        ax.plot(final_orbit_states[1][:,0], final_orbit_states[1][:,1], final_orbit_states[1][:,2], 'k-', lw=0.3)
        # ax.set_aspect('equal')
        ax.legend()

        ax_m = fig.add_subplot(2,3,2)
        ax_m.grid(True, alpha=0.5)
        for (_ts, _ys) in sols:
            ax_m.plot(_ys[:,7]*problem.TU2DAY, np.exp(_ys[:,6]), 'b-')
        ax_m.axhline(np.exp(sols[-1][1][-1,6]), color='r', linestyle='--')
        ax_m.text(xopt[0,7]*problem.TU2DAY, 0.01 + np.exp(sols[-1][1][-1,6]), f"m_f = {np.exp(sols[-1][1][-1,6]):1.4f}", color='r')
        ax_m.set(xlabel="Time, days", ylabel="Mass")
        #ax_m.legend()

        ax_u = fig.add_subplot(2,3,3)
        ax_u.grid(True, alpha=0.5)
        ax_u.step(xopt[:,7]*problem.TU2DAY, np.concatenate((vopt[:,0], [0.0]))*VU/TU, label="Control", where='post', color='k')
        ax_u.step(xopt[:,7]*problem.TU2DAY, np.concatenate((uopt[:,0], [0.0]))*VU/TU, label="u", where='post', color='r')
        ax_u.step(xopt[:,7]*problem.TU2DAY, np.concatenate((uopt[:,1], [0.0]))*VU/TU, label="v", where='post', color='g')
        ax_u.step(xopt[:,7]*problem.TU2DAY, np.concatenate((uopt[:,2], [0.0]))*VU/TU, label="w", where='post', color='b')
        for idx, (_ts, _ys) in enumerate(sols):
            ax_u.plot(_ys[:,7]*problem.TU2DAY, THRUST/(MSTAR*np.exp(_ys[:,6])), color='r', linestyle=':', label="Max accel." if idx == 0 else None)
        ax_u.set(xlabel="Time, days", ylabel="Control acceleration, m/s^2")
        ax_u.legend()

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
            ax.plot(_ts, _ys[:,7]*problem.TU2DAY, '--', color='grey')
        for (_ts, _ys) in sols:
            ax.plot(_ts, _ys[:,7]*problem.TU2DAY, marker="o", ms=2, color='k')
        ax.grid(True, alpha=0.5)
        ax.set(xlabel="tau", ylabel="Time, days")

        plt.tight_layout()
        fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots/example_pl2pl_logmass.png"), dpi=300)
    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--heyoka", default=1, type=int)
    parser.add_argument("--plot", default=1, type=int)
    args = parser.parse_args()
    example_pl2pl(use_heyoka=bool(args.heyoka), get_plot=bool(args.plot))
    plt.show()