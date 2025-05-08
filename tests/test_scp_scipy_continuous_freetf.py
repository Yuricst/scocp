"""Test SCP continuous transfer with free final time"""

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import scocp


def test_freetf_integrator(get_plot=False):
    """Test SCP continuous transfer"""
    mu = 1.215058560962404e-02

    # we create the time-domain integrator for plotting etc.
    integrator_timedomain = scocp.ScipyIntegrator(
        nx=6, nu=3, rhs=scocp.rhs_cr3bp, rhs_stm=scocp.rhs_cr3bp_stm, args=(mu,),
        method='DOP853', reltol=1e-12, abstol=1e-12
    )
    
    # this is the non-dimentional time integrator for solving the OCP
    integrator_01domain = scocp.ScipyIntegrator(
        nx=7,        # state is [x,y,z,vx,vy,vz,t]
        nu=4,        # controls are [ax,ay,az,s]    
        rhs=scocp.control_rhs_cr3bp_freetf, rhs_stm=scocp.control_rhs_cr3bp_freetf_stm,
        impulsive=False,
        args=(mu,[0.0,0.0,0.0,1.0]),   # last argument is dummy placeholder
        method='DOP853', reltol=1e-12, abstol=1e-12
    )

    # propagate uncontrolled and controlled dynamics
    x0 = np.array([
        1.0809931218390707E+00,
        0.0,
        -2.0235953267405354E-01,
        0.0,
        -1.9895001215078018E-01,
        0.0])
    period_0 = 2.3538670417546639E+00
    sol_lpo0 = integrator_timedomain.solve([0, period_0/2], x0, get_ODESolution=True)

    # integrate with time-dilation factor
    x0_tau = np.concatenate((x0, [0.0]))
    print(f"x0_tau = {x0_tau}")
    sol_lpo_tau = integrator_01domain.solve([0, 1.0], x0_tau,u=np.array([0.0, 0.0, 0.0, period_0/2]),get_ODESolution=True)
    print(f"sol_lpo_tau.y[:,-1] = {sol_lpo_tau.y[:,-1]}")

    assert np.allclose(sol_lpo0.y[0:6,-1], sol_lpo_tau.y[0:6,-1], atol=1e-11)

    if get_plot:
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(1,2,1,projection='3d')
        ax.plot(sol_lpo0.y[0,:], sol_lpo0.y[1,:], sol_lpo0.y[2,:], label="lpo0", lw=4.0)
        ax.plot(sol_lpo_tau.y[0,:], sol_lpo_tau.y[1,:], sol_lpo_tau.y[2,:], label="lpo_tau", lw=1.0)
        ax.legend()
        ax.set_aspect('equal')
        
        ax = fig.add_subplot(1,2,2)
        ax.plot(sol_lpo_tau.t, sol_lpo_tau.y[6,:], marker="o", ms=2, color='k')
        ax.axhline(sol_lpo0.t[0], color='r', linestyle='--', label="t0")
        ax.axhline(sol_lpo0.t[-1], color='g', linestyle='--', label="tf")
        ax.set(xlabel="tau", ylabel="time")
        ax.grid(True, alpha=0.5)
    return



def test_scp_scipy_freetf(get_plot=False):
    """Test SCP continuous transfer"""
    mu = 1.215058560962404e-02

    # we create the time-domain integrator for plotting etc.
    integrator_timedomain = scocp.ScipyIntegrator(
        nx=6, nu=3, rhs=scocp.rhs_cr3bp, rhs_stm=scocp.rhs_cr3bp_stm, args=(mu,),
        method='DOP853', reltol=1e-12, abstol=1e-12
    )
    
    # this is the non-dimentional time integrator for solving the OCP
    integrator_01domain = scocp.ScipyIntegrator(
        nx=7,        # state is [x,y,z,vx,vy,vz,t]
        nu=4,        # controls are [ax,ay,az,s]    
        rhs=scocp.control_rhs_cr3bp_freetf,
        rhs_stm=scocp.control_rhs_cr3bp_freetf_stm,
        impulsive=False,
        args=(mu,[0.0,0.0,0.0,1.0]),   # last argument is dummy placeholder
        method='DOP853', reltol=1e-12, abstol=1e-12
    )

    # propagate uncontrolled and controlled dynamics
    x0 = np.array([
        1.0809931218390707E+00,
        0.0,
        -2.0235953267405354E-01,
        0.0,
        -1.9895001215078018E-01,
        0.0])
    period_0 = 2.3538670417546639E+00
    sol_lpo0 = integrator_timedomain.solve([0, period_0], x0, get_ODESolution=True)

    xf = np.array([
        1.1648780946517576,
        0.0,
        -1.1145303634437023E-1,
        0.0,
        -2.0191923237095796E-1,
        0.0])
    period_f = 3.3031221822879884
    sol_lpo1 = integrator_timedomain.solve([0, period_f], xf, get_ODESolution=True)

    # transfer problem discretization
    N = 40
    tf_bounds = np.array([period_0, period_f])
    tf_guess = np.mean(tf_bounds)
    s_bounds = [0.01*tf_guess, 10*tf_guess]
    
    times_guess = np.linspace(0, tf_guess, N)    # initial guess
    taus = np.linspace(0, 1, N)
    umax = 0.3  # max acceleration

    # create subproblem
    problem = scocp.FreeTimeContinuousRdv(x0, xf, umax, tf_bounds, s_bounds, integrator_01domain, taus)

    # create initial guess
    print(f"Preparing initial guess...")
    sol_initial = integrator_timedomain.solve([0, tf_guess], x0, t_eval=times_guess, get_ODESolution=True)
    sol_final  = integrator_timedomain.solve([0, tf_guess], xf, t_eval=times_guess, get_ODESolution=True)

    alphas = np.linspace(1,0,N)
    xbar = (np.multiply(sol_initial.y, np.tile(alphas, (6,1))) + np.multiply(sol_final.y, np.tile(1-alphas, (6,1)))).T
    xbar[0,:] = x0  # overwrite initial state
    xbar[-1,:] = xf # overwrite final state
    xbar = np.concatenate((xbar,  np.linspace(0, tf_guess, N).reshape(-1,1)), axis=1)
    
    sbar_initial = tf_guess * np.ones((N-1,1))
    ubar = np.concatenate((np.zeros((N-1,3)), sbar_initial), axis=1)
    vbar = np.sum(ubar[:,0:3], axis=1).reshape(-1,1)

    # check initial guess
    _, sols_ig = problem.evaluate_nonlinear_dynamics(xbar, ubar, vbar, steps=5)
    # geq_nl_initial, sols_initial = problem.evaluate_nonlinear_dynamics(xbar, ubar, vbar)
    # fig = plt.figure(figsize=(10,5))
    # ax = fig.add_subplot(1,2,1,projection='3d')
    # for (_ts, _ys) in sols_initial:
    #     ax.plot(_ys[:,0], _ys[:,1], _ys[:,2], 'k-', lw=0.3)
    # ax.scatter(x0[0], x0[1], x0[2], marker='x', color='k', label='Initial state')
    # ax.scatter(xf[0], xf[1], xf[2], marker='o', color='k', label='Final state')
    # ax.scatter(xbar[:,0], xbar[:,1], xbar[:,2], marker='o', color='r', label='Initial guess')
    # ax.plot(sol_lpo0.y[0,:], sol_lpo0.y[1,:], sol_lpo0.y[2,:], 'k-', lw=0.3)
    # ax.plot(sol_lpo1.y[0,:], sol_lpo1.y[1,:], sol_lpo1.y[2,:], 'k-', lw=0.3)
    # ax.set_aspect('equal')
    # ax = fig.add_subplot(1,2,2)
    # for (_ts,_ys) in sols_initial:
    #     ax.plot(_ts, _ys[:,6], marker="o", ms=2, color='k')
    # ax.grid(True, alpha=0.5)

    # solve subproblem
    problem.solve_convex_problem(xbar, ubar, vbar)
    assert problem.cp_status == "optimal"

    # setup algorithm & solve
    tol_feas = 1e-10
    tol_opt = 1e-4
    algo = scocp.SCvxStar(problem, tol_opt=tol_opt, tol_feas=tol_feas)
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
    print(f"Initial guess TOF: {tf_guess:1.4f} --> Optimized TOF: {xopt[-1,6]:1.4f} (bounds: {tf_bounds[0]:1.4f} ~ {tf_bounds[1]:1.4f})")

    # evaluate nonlinear violations
    geq_nl_opt, sols = problem.evaluate_nonlinear_dynamics(xopt, uopt, vopt, steps=20)
    
    # evaluate solution
    if (get_plot is True) and (summary_dict["status"] != "CPFailed"):    
        # plot results
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(2,3,1,projection='3d')
        for (_ts, _ys) in sols_ig:
            ax.plot(_ys[:,0], _ys[:,1], _ys[:,2], '--', color='grey')
        for (_ts, _ys) in sols:
            ax.plot(_ys[:,0], _ys[:,1], _ys[:,2], 'b-')

            # interpolate control
            _us_zoh = scocp.zoh_controls(taus, uopt, _ts)
            ax.quiver(_ys[:,0], _ys[:,1], _ys[:,2], _us_zoh[:,0], _us_zoh[:,1], _us_zoh[:,2], color='r', length=0.5)

        ax.scatter(x0[0], x0[1], x0[2], marker='x', color='k', label='Initial state')
        ax.scatter(xf[0], xf[1], xf[2], marker='o', color='k', label='Final state')
        ax.plot(sol_lpo0.y[0,:], sol_lpo0.y[1,:], sol_lpo0.y[2,:], 'k-', lw=0.3)
        ax.plot(sol_lpo1.y[0,:], sol_lpo1.y[1,:], sol_lpo1.y[2,:], 'k-', lw=0.3)
        ax.set_aspect('equal')
        ax.legend()

        ax_u = fig.add_subplot(2,3,2)
        ax_u.grid(True, alpha=0.5)
        ax_u.step(xopt[:,6], np.concatenate((vopt[:,0], [0.0])), label="Gamma", where='post', color='k')
        ax_u.set(xlabel="Time", ylabel="Control")
        ax_u.legend()

        ax = fig.add_subplot(2,3,3)
        for (_ts, _ys) in sols_ig:
            ax.plot(_ts, _ys[:,6], '--', color='grey')
        for (_ts, _ys) in sols:
            ax.plot(_ts, _ys[:,6], marker="o", ms=2, color='k')
        ax.grid(True, alpha=0.5)
        ax.set(xlabel="tau", ylabel="time")

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

        plt.tight_layout()
        fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots/scp_scipy_scipy_freetf.png"), dpi=300)
    return


if __name__ == "__main__":
    #test_freetf_integrator(get_plot=True)
    test_scp_scipy_freetf(get_plot=True)
    plt.show()