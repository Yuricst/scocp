"""Test SCP continuous transfer with log-mass dynamics

For OCP with mass dynamics, make sure to do the following:
- Set `impulsive = False` in the integrator
- Set `augment_Gamma = True` in the problem class
"""

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import scocp


def test_scp_scipy_logmass(get_plot=False):
    """Test SCP continuous transfer with log-mass dynamics"""
    mu = 1.215058560962404e-02
    cex = 0.8       # exhaust velocity
    parameters_ode = (mu, cex)

    # we create the time-domain integrator for plotting etc.
    integrator_timedomain = scocp.ScipyIntegrator(
        nx=7,
        nu=3,
        rhs=scocp.control_rhs_cr3bp_logmass,
        rhs_stm=scocp.control_rhs_cr3bp_logmass_stm,
        args=(parameters_ode,[0.0,0.0,0.0,0.0]),
        method='DOP853',
        reltol=1e-12,
        abstol=1e-12
    )
    
    # this is the non-dimentional time integrator for solving the OCP
    integrator_01domain = scocp.ScipyIntegrator(
        nx=8,
        nu=4,
        nv=1,
        rhs=scocp.control_rhs_cr3bp_logmass_freetf,
        rhs_stm=scocp.control_rhs_cr3bp_logmass_freetf_stm,
        impulsive=False,
        args=(parameters_ode,[0.0,0.0,0.0,1.0,0.0]),
        method='DOP853', reltol=1e-12, abstol=1e-12)
    
    # propagate uncontrolled and controlled dynamics
    x0 = np.array([
        1.0809931218390707E+00,
        0.0,
        -2.0235953267405354E-01,
        0.0,
        -1.9895001215078018E-01,
        0.0,
        np.log(1.0),                        # initial log-mass (fixed)
    ])
    period_0 = 2.3538670417546639E+00
    sol_lpo0 = integrator_timedomain.solve([0, period_0], x0, get_ODESolution=True)

    xf = np.array([
        1.1648780946517576,
        0.0,
        -1.1145303634437023E-1,
        0.0,
        -2.0191923237095796E-1,
        0.0,
        np.log(0.5),                        # final log-mass (guess)
    ])
    period_f = 3.3031221822879884
    sol_lpo1 = integrator_timedomain.solve([0, period_f], xf, get_ODESolution=True)

    # transfer problem discretization
    N = 40
    tf_bounds = np.array([period_0, 1.3 * period_f])
    tf_guess = period_f
    s_bounds = [0.01*tf_guess, 10*tf_guess]
    
    times_guess = np.linspace(0, tf_guess, N)    # initial guess
    taus = np.linspace(0, 1, N)
    Tmax = 0.37637494800142673      # max thrust

    # create subproblem
    problem = scocp.FreeTimeContinuousRdvLogMass(
        x0, xf, Tmax, tf_bounds, s_bounds, N, integrator_01domain, taus, augment_Gamma=True,
        weight = 10.0
    )

    # create initial guess
    print(f"Preparing initial guess...")
    sol_initial = integrator_timedomain.solve([0, times_guess[-1]], x0, t_eval=times_guess, get_ODESolution=True)
    sol_final  = integrator_timedomain.solve([0, times_guess[-1]], xf, t_eval=times_guess, get_ODESolution=True)

    alphas = np.linspace(1,0,N)
    xbar = (np.multiply(sol_initial.y, np.tile(alphas, (7,1))) + np.multiply(sol_final.y, np.tile(1-alphas, (7,1)))).T
    xbar[:,6]   = np.log(np.linspace(1.0, 0.5, N))  # initial guess for log-mass
    xbar[0,:]   = x0                 # overwrite initial state
    xbar[-1,:]  = xf                 # overwrite final state
    xbar = np.concatenate((xbar,  times_guess.reshape(-1,1)), axis=1)    # append initial guess for time

    sbar_initial = tf_guess * np.ones((N-1,1))
    ubar = np.concatenate((np.divide(np.diff(xbar[:,3:6], axis=0), np.diff(times_guess)[:,None]), sbar_initial), axis=1)
    vbar = np.sum(ubar[:,0:3], axis=1).reshape(-1,1)
    print(f"Initial guess objective: {problem.evaluate_objective(xbar, ubar, vbar):1.4e}")

    geq_nl_ig, sols_ig = problem.evaluate_nonlinear_dynamics(xbar, ubar, vbar, steps=5)
    print(f"np.linalg.norm(geq_nl_ig) = {np.linalg.norm(geq_nl_ig)}")
    
    # solve subproblem
    print(f"ubar.shape = {ubar.shape}, xbar.shape = {xbar.shape}, vbar.shape = {vbar.shape}")
    print(f"problem.Phi_A.shape = {problem.Phi_A.shape}, problem.Phi_B.shape = {problem.Phi_B.shape}, problem.Phi_c.shape = {problem.Phi_c.shape}")
    problem.solve_convex_problem(xbar, ubar, vbar)
    assert problem.cp_status == "optimal"

    # setup algorithm & solve
    tol_feas = 1e-8
    tol_opt = 1e-6
    algo = scocp.SCvxStar(problem, tol_opt=tol_opt, tol_feas=tol_feas)
    solution = algo.solve(
        xbar,
        ubar,
        vbar,
        maxiter = 250,
        verbose = True
    )
    xopt, uopt, vopt, yopt, sols, summary_dict = solution.x, solution.u, solution.v, solution.y, solution.sols, solution.summary_dict
    assert summary_dict["status"] == "Optimal"
    assert summary_dict["chi"][-1] <= tol_feas
    print(f"Initial guess TOF: {tf_guess:1.4f} --> Optimized TOF: {xopt[-1,7]:1.4f} (bounds: {tf_bounds[0]:1.4f} ~ {tf_bounds[1]:1.4f})")
    
    # evaluate nonlinear violations
    geq_nl_opt, sols = problem.evaluate_nonlinear_dynamics(xopt, uopt, vopt, steps=20)
    assert np.max(np.abs(geq_nl_opt)) <= tol_feas
    
    # evaluate solution
    if (get_plot is True) and (summary_dict["status"] != "CPFailed"):
        # plot results
        fig = plt.figure(figsize=(12,7))
        ax = fig.add_subplot(2,3,1,projection='3d')
        for (_ts, _ys) in sols_ig:
            ax.plot(_ys[:,0], _ys[:,1], _ys[:,2], '--', color='grey')
        for (_ts, _ys) in sols:
            ax.plot(_ys[:,0], _ys[:,1], _ys[:,2], 'b-')

            # interpolate control
            _us_zoh = scocp.zoh_controls(taus, uopt, _ts)
            ax.quiver(_ys[:,0], _ys[:,1], _ys[:,2], _us_zoh[:,0], _us_zoh[:,1], _us_zoh[:,2], color='r', length=0.1)

        ax.scatter(x0[0], x0[1], x0[2], marker='x', color='k', label='Initial state')
        ax.scatter(xf[0], xf[1], xf[2], marker='o', color='k', label='Final state')
        ax.plot(sol_lpo0.y[0,:], sol_lpo0.y[1,:], sol_lpo0.y[2,:], 'k-', lw=0.3)
        ax.plot(sol_lpo1.y[0,:], sol_lpo1.y[1,:], sol_lpo1.y[2,:], 'k-', lw=0.3)
        ax.set_aspect('equal')
        ax.legend()

        ax_m = fig.add_subplot(2,3,2)
        ax_m.grid(True, alpha=0.5)
        for (_ts, _ys) in sols:
            ax_m.plot(_ys[:,7], np.exp(_ys[:,6]), 'b-')
        ax_m.axhline(np.exp(sols[-1][1][-1,6]), color='r', linestyle='--')
        ax_m.text(0.0, 0.01 + np.exp(sols[-1][1][-1,6]), f"m_f = {np.exp(sols[-1][1][-1,6]):1.4f}", color='r')
        ax_m.set(xlabel="Time", ylabel="Mass")
        ax_m.legend()

        ax_u = fig.add_subplot(2,3,3)
        ax_u.grid(True, alpha=0.5)
        ax_u.step(xopt[:,7], np.concatenate((vopt[:,0], [0.0])), label="Control", where='post', color='k')
        for idx, (_ts, _ys) in enumerate(sols):
            ax_u.plot(_ys[:,7], Tmax/np.exp(_ys[:,6]), color='r', linestyle=':', label="Max accel." if idx == 0 else None)
        ax_u.set(xlabel="Time", ylabel="Acceleration")
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
            ax.plot(_ts, _ys[:,7], '--', color='grey')
        for (_ts, _ys) in sols:
            ax.plot(_ts, _ys[:,7], marker="o", ms=2, color='k')
        ax.grid(True, alpha=0.5)
        ax.set(xlabel="tau", ylabel="time")

        plt.tight_layout()
        fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots/scp_scipy_logmass_freetf_transfer.png"), dpi=300)
    return


if __name__ == "__main__":
    test_scp_scipy_logmass(get_plot=True)
    plt.show()