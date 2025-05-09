"""Test rendezvous with moving target"""

import matplotlib.pyplot as plt
import numpy as np
import pykep as pk

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import scocp
import scocp_pykep


def example_pl2pl(get_plot=False):
    """Test SCP continuous transfer with log-mass dynamics"""
    # define canonical parameters
    GM_SUN = pk.MU_SUN           # Sun GM, m^3/s^-2
    ISP    = 3000.0              # specific impulse, s
    G0     = pk.G0               # gravity at surface, m/s^2

    DU = pk.AU                   # length scale set to Sun-Earth distance, m
    VU = np.sqrt(GM_SUN / DU)    # velocity scale, m/s
    TU = DU / VU                 # time scale, s

    # define initial and final planets
    pl0 = pk.planet.jpl_lp('earth')
    plf = pk.planet.jpl_lp('mars')

    # this is the non-dimentional time integrator for solving the OCP
    integrator_01domain = scocp.ScipyIntegrator(
        nx=8,
        nu=4,
        nv=1,
        rhs=scocp.control_rhs_twobody_logmass_freetf,
        rhs_stm=scocp.control_rhs_twobody_logmass_freetf_stm,
        impulsive=False,
        args=((
            GM_SUN / (VU**2 * DU),          # canonical gravitational constant
            ISP * G0 * (TU/DU)),            # canonical exhaust velocity of thruster
            [0.0,0.0,0.0,1.0,0.0]           # place-holder for control vector: [ax,ay,az,s,v]
        ),
        method='DOP853', reltol=1e-12, abstol=1e-12
    )
    
    # create problem
    problem = scocp_pykep.scocp_pl2pl(
        integrator_01domain,
        pl0,
        plf,
    )
    
    # create initial guess
    print(f"Preparing initial guess...")
    xbar, ubar, vbar = problem.get_initial_guess(t0_guess=0.0, tf_guess=250.0)
    geq_nl_ig, sols_ig = problem.evaluate_nonlinear_dynamics(xbar, ubar, vbar, steps=5)   # evaluate initial guess

    # setup algorithm & solve
    tol_feas = 1e-10
    tol_opt = 1e-6
    algo = scocp.SCvxStar(problem, tol_opt=tol_opt, tol_feas=tol_feas, rho1=1e-8)#, alpha2=1.5)
    solution = algo.solve(
        xbar,
        ubar,
        vbar,
        maxiter = 150,
        verbose = True
    )
    xopt, uopt, vopt, yopt, sols, summary_dict = solution.x, solution.u, solution.v, solution.y, solution.sols, solution.summary_dict
    assert summary_dict["status"] == "Optimal"
    assert summary_dict["chi"][-1] <= tol_feas
    x0 = problem.target_initial.target_state(xopt[0,7])
    xf = problem.target_final.target_state(xopt[-1,7])

    # evaluate nonlinear violations
    geq_nl_opt, sols = problem.evaluate_nonlinear_dynamics(xopt, uopt, vopt, steps=20)
    print(f"Max dynamics constraint violation: {np.max(np.abs(geq_nl_opt)):1.4e}")
    assert np.max(np.abs(geq_nl_opt)) <= tol_feas
    
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
            ax.quiver(_ys[:,0], _ys[:,1], _ys[:,2], _us_zoh[:,0], _us_zoh[:,1], _us_zoh[:,2], color='r', length=5.0)

        ax.scatter(x0[0], x0[1], x0[2], marker='x', color='k', label='Initial state')
        ax.scatter(xf[0], xf[1], xf[2], marker='o', color='k', label='Final state')
        ax.plot(initial_orbit_states[1][:,0], initial_orbit_states[1][:,1], initial_orbit_states[1][:,2], 'k-', lw=0.3)
        ax.plot(final_orbit_states[1][:,0], final_orbit_states[1][:,1], final_orbit_states[1][:,2], 'k-', lw=0.3)
        ax.set_aspect('equal')
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
        for idx, (_ts, _ys) in enumerate(sols):
            ax_u.plot(_ys[:,7]*problem.TU2DAY, problem.max_thrust/(problem.mass_scaling*np.exp(_ys[:,6])),
                      color='r', linestyle=':', label="Max accel." if idx == 0 else None)
        ax_u.set(xlabel="Time, days", ylabel="Acceleration, m/s^2")
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
        
        ax = fig.add_subplot(2,3,6)
        for (_ts, _ys) in sols_ig:
            ax.plot(_ts, _ys[:,7]*problem.TU2DAY, '--', color='grey')
        for (_ts, _ys) in sols:
            ax.plot(_ts, _ys[:,7]*problem.TU2DAY, marker="o", ms=2, color='k')
        ax.grid(True, alpha=0.5)
        ax.set(xlabel="tau", ylabel="Time, days")

        plt.tight_layout()
    return


if __name__ == "__main__":
    example_pl2pl(get_plot=True)
    plt.show()