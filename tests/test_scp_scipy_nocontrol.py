"""Using scocp for multiple shooting, without any control"""

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import scocp

def test_scp_scipy_nocontrol(get_plot = False):
    """Test SCP without control"""
    mu = 1.215058560962404e-02
    integrator = scocp.ScipyIntegrator(nx=6, nu=0, rhs=scocp.rhs_cr3bp, rhs_stm=scocp.rhs_cr3bp_stm, args=(mu,),
                                       impulsive=True, method='DOP853', reltol=1e-12, abstol=1e-12)
    
    # initial state of true, sought orbit
    x0 = np.array([
        1.0809931218390707E+00,
        0.0,
        -2.0235953267405354E-01,
        0.0,
        -1.9895001215078018E-01,
        0.0])
    period = 2.3538670417546639E+00

    # construct discretized times and evaluate true solution along it for testing
    N = 10
    times = np.linspace(0, period, N)
    sol = integrator.solve([0, period], x0, get_ODESolution=True, t_eval=times)

    # create initial guess by perturbing true solution
    xbar = np.array([
        x_true + 0.05 * np.random.randn(6)
        for x_true in sol.y[:6, :].T
    ])
    xbar[0] = x0
    xbar[-1] = x0
    print(f"xbar.shape = {xbar.shape}")

    # create problem for a periodic trajectory
    problem = scocp.FixedTimeBallisticTrajectory(x0, x0, 0.0, integrator, times)

    # solve subproblem
    ubar = np.zeros((N-1,0))
    vbar = np.zeros((N-1,0))
    problem.solve_convex_problem(xbar, ubar, vbar)
    assert problem.cp_status == "optimal", f"problem.cp_status = {problem.cp_status}"

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
    # assert summary_dict["status"] == "Optimal"
    # assert summary_dict["chi"][-1] <= tol_feas
    print(f"ubar.shape = {ubar.shape}, uopt.shape = {uopt.shape}")
    
    # evaluate solution
    if (get_plot is True) and (summary_dict["status"] != "CPFailed"):
        _, sols_ig = problem.evaluate_nonlinear_dynamics(xbar, ubar, vbar, steps=5)
    
        # plot results
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(2,2,1,projection='3d')
        for (_ts, _ys) in sols_ig:
            ax.plot(_ys[:,0], _ys[:,1], _ys[:,2], '--', color='grey')
        for (_ts, _ys) in sols:
            ax.plot(_ys[:,0], _ys[:,1], _ys[:,2], 'b-')

        ax.scatter(x0[0], x0[1], x0[2], marker='x', color='k', label='Initial state')
        # ax.scatter(xf[0], xf[1], xf[2], marker='o', color='k', label='Final state')
        ax.set_aspect('equal')
        ax.legend()

        # ax_u = fig.add_subplot(2,2,2)
        # ax_u.grid(True, alpha=0.5)
        # ax_u.step(times, np.concatenate((vopt[:,0], [0.0])), label="Gamma", where='post', color='k')
        # ax_u.set(xlabel="Time", ylabel="Control")
        # ax_u.legend()

        ax_DeltaJ = fig.add_subplot(2,2,3)
        ax_DeltaJ.grid(True, alpha=0.5)
        algo.plot_DeltaJ(ax_DeltaJ, summary_dict)
        ax_DeltaJ.axhline(tol_opt, color='k', linestyle='--', label='tol_opt')
        ax_DeltaJ.legend()

        ax_DeltaL = fig.add_subplot(2,2,4)
        ax_DeltaL.grid(True, alpha=0.5)
        algo.plot_chi(ax_DeltaL, summary_dict)
        ax_DeltaL.axhline(tol_feas, color='k', linestyle='--', label='tol_feas')
        ax_DeltaL.legend()

        plt.tight_layout()
    return


if __name__ == "__main__":
    test_scp_scipy_nocontrol(get_plot = True)
    plt.show()