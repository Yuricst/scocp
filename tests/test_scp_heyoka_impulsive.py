"""Test SCP impulsive transfer"""

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import scocp


def test_scp_scipy_impulsive(get_plot=False):
    """Test SCP impulsive transfer"""
    ta_dyn, ta_dyn_aug = scocp.get_heyoka_integrator_cr3bp(mu=1.215058560962404e-02, tol=1e-12)
    integrator = scocp.HeyokaIntegrator(nx=6, nu=3, ta=ta_dyn, ta_stm=ta_dyn_aug, impulsive=True)
    
    # propagate uncontrolled and controlled dynamics
    x0 = np.array([
        1.0809931218390707E+00,
        0.0,
        -2.0235953267405354E-01,
        0.0,
        -1.9895001215078018E-01,
        0.0])
    period_0 = 2.3538670417546639E+00
    sol_lpo0 = integrator.solve([0.0, period_0], x0, t_eval=np.linspace(0.0, period_0, 100))

    xf = np.array([
        1.1648780946517576,
        0.0,
        -1.1145303634437023E-1,
        0.0,
        -2.0191923237095796E-1,
        0.0])
    period_f = 3.3031221822879884
    sol_lpo1 = integrator.solve([0.0, period_f], xf, t_eval=np.linspace(0.0, period_f, 100))

    # transfer problem discretization
    N = 20
    tf = (period_0 + period_f) / 2
    times = np.linspace(0, tf, N)

    # create subproblem
    problem = scocp.FixedTimeImpulsiveRdv(x0, xf, integrator, times)

    # create initial guess
    print(f"Preparing initial guess...")
    sol_initial = integrator.solve([0, times[-1]], x0, t_eval=times)
    sol_final  = integrator.solve([0, times[-1]], xf, t_eval=times)

    alphas = np.linspace(1,0,N)
    xbar = (np.multiply(sol_initial[1].T, np.tile(alphas, (6,1))) + np.multiply(sol_final[1].T, np.tile(1-alphas, (6,1)))).T
    xbar[0,:] = x0  # overwrite initial state
    xbar[-1,:] = xf # overwrite final state
    ubar = np.zeros((N,3))

    # solve subproblem
    gbar = np.sum(ubar, axis=1).reshape(-1,1)
    _, _, _, _, _, _ = problem.solve_convex_problem(xbar, ubar, gbar)
    assert problem.cp_status == "optimal"

    # setup algorithm & solve
    tol_feas = 1e-10
    tol_opt = 1e-4
    algo = scocp.SCvxStar(problem, tol_opt=tol_opt, tol_feas=tol_feas)
    xopt, uopt, gopt, sols, summary_dict = algo.solve(
        xbar,
        ubar,
        gbar,
        maxiter = 100,
        verbose = True
    )
    assert summary_dict["status"] == "Optimal"
    assert summary_dict["chi"][-1] <= tol_feas

    # evaluate nonlinear violations
    geq_nl_opt, sols = problem.evaluate_nonlinear_dynamics(xopt, uopt, steps=20)
    assert np.max(np.abs(geq_nl_opt)) <= tol_feas

    # evaluate solution
    if (get_plot is True) and (summary_dict["status"] != "CPFailed"):
        _, sols_ig = problem.evaluate_nonlinear_dynamics(xbar, ubar, steps=20)
    
        # plot results
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(2,2,1,projection='3d')
        for (_ts, _ys) in sols_ig:
            ax.plot(_ys[:,0], _ys[:,1], _ys[:,2], '--', color='grey')
        for (_ts, _ys) in sols:
            ax.plot(_ys[:,0], _ys[:,1], _ys[:,2], 'b-')
        ax.scatter(x0[0], x0[1], x0[2], marker='x', color='k', label='Initial state')
        ax.scatter(xf[0], xf[1], xf[2], marker='o', color='k', label='Final state')
        ax.plot(sol_lpo0[1].T[0,:], sol_lpo0[1].T[1,:], sol_lpo0[1].T[2,:], 'k-', lw=0.3)
        ax.plot(sol_lpo1[1].T[0,:], sol_lpo1[1].T[1,:], sol_lpo1[1].T[2,:], 'k-', lw=0.3)
        ax.quiver(xopt[:,0], xopt[:,1], xopt[:,2], uopt[:,0], uopt[:,1], uopt[:,2], color='r', length=1.0)
        ax.set_aspect('equal')
        ax.legend()

        ax_u = fig.add_subplot(2,2,2)
        ax_u.grid(True, alpha=0.5)
        ax_u.stem(times, gopt, markerfmt='D', label="Gamma")
        ax_u.set(xlabel="Time", ylabel="Control")
        ax_u.legend()

        ax_DeltaJ = fig.add_subplot(2,2,3)
        ax_DeltaJ.grid(True, alpha=0.5)
        ax_DeltaJ.plot(np.abs(summary_dict["DeltaJ"]), marker="o", color="k", ms=3)
        ax_DeltaJ.axhline(tol_opt, color='r', linestyle='--', label='tol_opt')
        ax_DeltaJ.set(yscale='log', xlabel='Iter.', ylabel='|DeltaJ|')
        ax_DeltaJ.legend()

        ax_DeltaL = fig.add_subplot(2,2,4)
        ax_DeltaL.grid(True, alpha=0.5)
        ax_DeltaL.plot(summary_dict["chi"], marker="o", color="k", ms=3)
        ax_DeltaL.axhline(tol_feas, color='r', linestyle='--', label='tol_feas')
        ax_DeltaL.set(yscale='log', xlabel='Iter.', ylabel='chi')
        ax_DeltaL.legend()

        plt.tight_layout()
        fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots/scp_heyoka_impulsive_transfer.png"), dpi=300)
        plt.show()
    return


if __name__ == "__main__":
    test_scp_scipy_impulsive(get_plot=True)
    plt.show()