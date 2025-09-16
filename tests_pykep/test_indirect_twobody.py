"""Indirect method for two-body problem"""

import matplotlib.pyplot as plt
import numpy as np
import pykep as pk

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import scocp
import scocp_pykep


def test_indirect_twobody(verbose = False, get_plot = False):
    # construst taylor integrators
    ta, ta_var = scocp_pykep.get_heyoka_integrator_twobody_indirect(tol = 1e-12, tol_var = 1e-12)
    H_func, u_func, rho_func, i_vers_func = scocp_pykep.get_twobody_indirect_functions()

    # Test the shooting function. Setting the parameters
    x0 = np.array([
        0.22801737222709886, -0.9352406973722779, -0.09383706869181557,
        0.9767094578416712, 0.3357560395100114, 0.033687972124946564,
    ])

    xf = np.array([
        -1.0629748432590669, -0.12588746003418305, 0.10549259166260608,
        0.09337624079153128, -0.9702436228220527, -0.016231606546965493,
    ])

    # create integrator for SCP
    integrator = scocp_pykep.HeyokaIntegrator(nx=14, nu=0, nv=0, ta=ta, ta_stm=ta_var, impulsive=True)

    # Units
    L = 149597870700.0  # AU
    MU = 1.3271244004127942e20  # m^3/s^2 (gravitational parameter of the Sun)
    T = np.sqrt(L**3 / MU)  # Orbital period at 1AU will then be 2pi

    V = L / T
    ACC = L / T / T
    M = 1500

    # Lets set the parameters
    mu = 1.0
    eps = 1e-5
    c1 = 0.6 / ACC / M
    c2 = (2500.*pk.G0) / V
    ta.pars[:4] = [mu, c1, c2, eps]

    ta_var.pars[:] = [mu, c1, c2, eps, 1.0]
    tof = 250.0 * 86400 / T

    # construct SCP
    N = 3
    times = np.linspace(0.0, tof, N)

    xf_dict = {
        i: xf[i] for i in range(6)
    }
    problem = scocp.IndirectOptimalControl(
        list(x0) + [1.0], xf_dict, integrator, times,
        l1_penalty = False
    )
    
    # construct initial guess
    _ig_rv_orbit0 = np.zeros((len(times),6))
    _ig_rv_orbitf = np.zeros((len(times),6))

    for i in range(len(times)):
        [_ig_rv_orbit0[i,0:3], _ig_rv_orbit0[i,3:6]] = pk.propagate_lagrangian(
            rv=[x0[0:3],x0[3:6]], tof = times[i], mu = mu, stm = False
        )
        [_ig_rv_orbitf[i,0:3], _ig_rv_orbitf[i,3:6]] = pk.propagate_lagrangian(
            rv=[xf[0:3],xf[3:6]], tof = times[i], mu = mu, stm = False
        )

    alphas = np.linspace(1,0,N)
    xbar = np.hstack((_ig_rv_orbit0, np.ones((N,1)), np.ones((N,7))))
    xbar[0,:6] = x0                  # overwrite initial state
    xbar[-1,:6] = xf                 # overwrite final state
    xbar[-1,13] = 0.0                # transversality condition
    ubar = np.zeros((N,0))
    vbar = np.zeros((N,0))

    # test solve convex subproblem
    problem.solve_convex_problem(xbar, ubar, vbar)
    assert problem.cp_status == "optimal"

    # setup algorithm & solve
    tol_feas = 1e-10
    tol_opt = 1e-2
    algo = scocp.SCvxStar(problem, tol_opt=tol_opt, tol_feas=tol_feas)

    print(ubar.shape)

    solution = algo.solve(
        xbar,
        ubar,
        maxiter = 100,
        verbose = verbose
    )

    # evaluate solution
    g_opt, sols_opt = problem.evaluate_nonlinear_dynamics(solution.x, solution.u, solution.v, steps=50)
    # assert np.max(np.abs(g_opt)) <= tol_feas

    # evaluate optimality history
    ts_plot = []
    H_hist = []
    umag_hist = []
    ivec_hist = []

    for (ts,rvls) in sols_opt:
        #u_func(rvls[0,:14], pars=ta.pars)
        ts_plot.append(ts)
        for (t,rvl) in zip(ts,rvls):
            H_hist.append(H_func(rvl[:14], pars=ta_var.pars))
            umag_hist.append(u_func(rvl[:14], pars=ta_var.pars))
            ivec_hist.append(i_vers_func(rvl[10:13]))

    ts_plot = np.concatenate(ts_plot)
    ivec_hist = np.array(ivec_hist)

    if get_plot:
        # -------------------------- plot of solution --------------------------
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(121, projection='3d')
        axu = fig.add_subplot(122)
        
        # propagate initial & final orbits
        times_eval = np.linspace(0.0, tof, 200)
        sol_orbit0 = np.zeros((len(times_eval),6))
        sol_orbitf = np.zeros((len(times_eval),6))
        for i in range(len(times_eval)):
            [sol_orbit0[i,0:3], sol_orbit0[i,3:6]] = pk.propagate_lagrangian(
                rv=[x0[0:3],x0[3:6]], tof = times_eval[i], mu = mu, stm = False
            )
            [sol_orbitf[i,0:3], sol_orbitf[i,3:6]] = pk.propagate_lagrangian(
                rv=[xf[0:3],xf[3:6]], tof = times_eval[i], mu = mu, stm = False
            )

        ax.plot(sol_orbit0[:,0], sol_orbit0[:,1], sol_orbit0[:,2], color='k', lw=0.75)
        ax.plot(sol_orbitf[:,0], sol_orbitf[:,1], sol_orbitf[:,2], color='k', lw=0.75)
        ax.plot(x0[0], x0[1], x0[2], '^k')
        ax.plot(xf[0], xf[1], xf[2], 'vk')
        for (ts,rvls) in sols_opt:
            _umags = []
            for (t,rvl) in zip(ts,rvls):
                _umags.append(u_func(rvl[:14], pars=ta_var.pars))
            ax.scatter(rvls[:,0], rvls[:,1], rvls[:,2], c=np.array(_umags), vmin=0, vmax=1, marker='.', s=4, cmap='bwr')
            ax.plot(rvls[:,0], rvls[:,1], rvls[:,2], color='grey', lw=0.75)
        ax.set_aspect('equal')

        axu.plot(ts_plot, ivec_hist[:,0], color='r', label='i_x')
        axu.plot(ts_plot, ivec_hist[:,1], color='b', label='i_y')
        axu.plot(ts_plot, ivec_hist[:,2], color='g', label='i_z')
        axu.plot(ts_plot, umag_hist, color='k', label='||u||')
        axu.legend()
        plt.tight_layout()

        # -------------------------- plot to diagnose SCP --------------------------
        fig_scp = plt.figure(figsize=(10,6))
        ax_DeltaJ = fig_scp.add_subplot(2,3,1)
        ax_DeltaJ.grid(True, alpha=0.5)
        algo.plot_DeltaJ(ax_DeltaJ, solution.summary_dict)
        ax_DeltaJ.axhline(tol_opt, color='k', linestyle='--', label='tol_opt')
        ax_DeltaJ.legend()

        ax_DeltaL = fig_scp.add_subplot(2,3,2)
        ax_DeltaL.grid(True, alpha=0.5)
        algo.plot_chi(ax_DeltaL, solution.summary_dict)
        ax_DeltaL.axhline(tol_feas, color='k', linestyle='--', label='tol_feas')
        ax_DeltaL.legend()

        ax_J0 = fig_scp.add_subplot(2,3,3)
        ax_J0.grid(True, alpha=0.5)
        algo.plot_J0(ax_J0, solution.summary_dict)
        ax_J0.legend()

        ax_w = fig_scp.add_subplot(2,3,4)
        ax_w.grid(True, alpha=0.5)
        algo.plot_w(ax_w, solution.summary_dict)

        ax_tr = fig_scp.add_subplot(2,3,5)
        ax_tr.grid(True, alpha=0.5)
        algo.plot_trust_region_radius_x(ax_tr, solution.summary_dict)
        ax_tr.legend()
        plt.tight_layout()
    return


if __name__=="__main__":
    test_indirect_twobody(verbose = True, get_plot = True)
    plt.show()