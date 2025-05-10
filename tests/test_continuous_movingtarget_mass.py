"""Test rendezvous with moving target"""

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import scocp


def test_scp_scipy_logmass(get_plot=False):
    """Test SCP continuous transfer with log-mass dynamics"""
    # define canonical parameters
    GM_SUN = 132712000000.44     # Sun GM, km^3/s^-2
    MSTAR  = 1000.0              # reference spacecraft mass
    ISP    = 3000.0              # specific impulse, s
    THRUST = 0.3                 # max thrust, kg.m/s^2
    G0     = 9.81                # gravity at surface, m/s^2

    DU = 149.6e6                 # length scale set to Sun-Earth distance, km
    VU = np.sqrt(GM_SUN / DU)    # velocity scale, km/s
    TU = DU / VU                 # time scale, s

    isp = ISP/TU                                # canonical specific impulse, TU
    c1 = THRUST * (1/MSTAR)*(TU**2/(1e3*DU))  # canonical max thrust
    c2 = c1/(isp * G0*(TU**2/(1e3*DU)) )

    mu = 1.0
    parameters_ode = (mu, c1, c2)

    print(f"\nCanonical c1: {c1:1.4e}, c2: {c2:1.4e}")

    # we create the time-domain integrator for plotting etc.
    integrator_timedomain = scocp.ScipyIntegrator(
        nx=6,
        nu=0,
        rhs=scocp.rhs_twobody,
        rhs_stm=scocp.rhs_twobody_stm,
        args=(mu,),
        method='DOP853',
        reltol=1e-12,
        abstol=1e-12
    )
    
    # this is the non-dimentional time integrator for solving the OCP
    integrator_01domain = scocp.ScipyIntegrator(
        nx=8,
        nu=4,
        nv=1,
        rhs=scocp.control_rhs_twobody_mass_freetf,
        rhs_stm=scocp.control_rhs_twobody_mass_freetf_stm,
        impulsive=False,
        args=(parameters_ode,[0.0,0.0,0.0,1.0,0.0]),
        method='DOP853', reltol=1e-12, abstol=1e-12)
    
    # propagate uncontrolled and controlled dynamics
    oe0 = np.array([1., 0.0, np.deg2rad(1), 1.5, 0.0, np.deg2rad(0)])
    x0 = np.concatenate((scocp.kep2rv(oe0, mu), [1.0]))
    period_0 = 2*np.pi*np.sqrt(oe0[0]**3/mu)
    sol_lpo0 = integrator_timedomain.solve([0, period_0], x0[0:6], get_ODESolution=True)

    oef0 = np.array([1.5, 0.01, np.deg2rad(3), 0.3, 0.0, np.deg2rad(130)])
    xf0 = scocp.kep2rv(oef0, mu)
    period_f = 2*np.pi*np.sqrt(oef0[0]**3/mu)
    sol_lpo1 = integrator_timedomain.solve([0, period_f], xf0[0:6], get_ODESolution=True)

    # transfer problem discretization
    N = 40
    tf_bounds = np.array([period_0, 1.3 * period_f])
    tf_guess = 2*np.pi
    s_bounds = [0.01*tf_guess, 10*tf_guess]
    
    times_guess = np.linspace(0, tf_guess, N)    # initial guess
    taus = np.linspace(0, 1, N)

    # create target object
    def eval_target_state(t: float) -> np.ndarray:
        return scocp.keplerder_nostm(mu, xf0, 0.0, t, tol=1e-12, maxiter=10)
    
    def eval_target_state_derivative(t: float) -> np.ndarray:
        state = eval_target_state(t)
        return scocp.rhs_twobody(t, state, mu)
    
    target = scocp.MovingTarget(eval_target_state, eval_target_state_derivative)

    # evaluate target state
    xf_guess = target.eval_state(tf_guess)

    # create subproblem
    problem = scocp.FreeTimeContinuousMovingTargetRdvMass(
        x0, target, c1, tf_bounds, s_bounds, N, integrator_01domain, taus, augment_Gamma=True,
        weight = 10.0
    )

    # create initial guess
    print(f"Preparing initial guess...")
    elements = np.concatenate((
        np.linspace(oe0[0], oef0[0], N).reshape(-1,1),
        np.linspace(oe0[1], oef0[1], N).reshape(-1,1),
        np.linspace(oe0[2], oef0[2], N).reshape(-1,1),
        np.linspace(oe0[3], oef0[3], N).reshape(-1,1),
        np.linspace(oe0[4], oef0[4], N).reshape(-1,1),
        np.linspace(oe0[5], oef0[5], N).reshape(-1,1),
    ), axis=1)
    elements[:,5] = np.linspace(oe0[5], oef0[5]+2*np.pi, N)
    xbar = np.zeros((N,8))
    xbar[:,0:6] = np.array([scocp.kep2rv(E,mu) for E in elements])
    xbar[:,6]   = np.linspace(1.0, 0.5, N)  # initial guess for log-mass
    xbar[0,0:7]   = x0[0:7]                 # overwrite initial state
    xbar[-1,0:6]  = xf_guess[0:6]           # overwrite final state
    xbar[:,7]   = times_guess               # initial guess for time

    sbar_initial = tf_guess * np.ones((N-1,1))
    ubar = np.concatenate((np.zeros((N-1,3)), sbar_initial), axis=1)
    vbar = np.sum(ubar[:,0:3], axis=1).reshape(-1,1)
    geq_nl_ig, sols_ig = problem.evaluate_nonlinear_dynamics(xbar, ubar, vbar, steps=5) # evaluate initial guess

    # # plot initial guess
    # fig = plt.figure(figsize=(12,5))
    # ax = fig.add_subplot(1,2,1,projection='3d')
    # ax.plot(sol_lpo0.y[0,:], sol_lpo0.y[1,:], sol_lpo0.y[2,:], 'k-', lw=0.3)
    # ax.plot(sol_lpo1.y[0,:], sol_lpo1.y[1,:], sol_lpo1.y[2,:], 'k-', lw=0.3)
    # ax.scatter(xbar[:,0], xbar[:,1], xbar[:,2], marker='o', color='k', label='Nodes')
    # for (ts, ys) in sols_ig:
    #     ax.plot(ys[:,0], ys[:,1], ys[:,2], 'b-')
    # ax.legend()

    # ax_m = fig.add_subplot(1,2,2)
    # ax_m.plot(xbar[:,7], xbar[:,6], 'k-')
    # ax_m.set(xlabel="Time", ylabel="Mass")
    # ax_m.legend()
    
    # solve subproblem
    print(f"ubar.shape = {ubar.shape}, xbar.shape = {xbar.shape}, vbar.shape = {vbar.shape}")
    print(f"problem.Phi_A.shape = {problem.Phi_A.shape}, problem.Phi_B.shape = {problem.Phi_B.shape}, problem.Phi_c.shape = {problem.Phi_c.shape}")
    problem.solve_convex_problem(xbar, ubar, vbar)
    assert problem.cp_status == "optimal", f"CP status: {problem.cp_status}"

    # setup algorithm & solve
    tol_feas = 1e-8
    tol_opt = 1e-6
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
    print(f"Initial guess TOF: {tf_guess:1.4f} --> Optimized TOF: {xopt[-1,7]:1.4f} (bounds: {tf_bounds[0]:1.4f} ~ {tf_bounds[1]:1.4f})")
    xf = target.eval_state(xopt[-1,7])

    # trajectory summary
    print(f"  Departure time : {xopt[0,7]*TU/86400.0:1.4f} days")
    print(f"  Arrival time   : {xopt[-1,7]*TU/86400.0:1.4f} days")
    print(f"  TOF            : {(xopt[-1,7] - xopt[0,7])*TU/86400.0:1.4f} days")
    print(f"  Initial mass   : {xopt[0,6]*MSTAR:1.4f} kg")
    print(f"  Final mass     : {xopt[-1,6]*MSTAR:1.4f} kg")

    # evaluate nonlinear violations
    geq_nl_opt, sols = problem.evaluate_nonlinear_dynamics(xopt, uopt, vopt, steps=5)
    assert np.max(np.abs(geq_nl_opt)) <= tol_feas
    
    # # evaluate solution
    if (get_plot is True) and (summary_dict["status"] != "CPFailed"):
        # plot results
        fig = plt.figure(figsize=(12,7))
        ax = fig.add_subplot(2,3,1,projection='3d')
        ax.set(xlabel="x", ylabel="y", zlabel="z")
        for (_ts, _ys) in sols_ig:
            ax.plot(_ys[:,0], _ys[:,1], _ys[:,2], '--', color='grey')
        for (_ts, _ys) in sols:
            ax.plot(_ys[:,0], _ys[:,1], _ys[:,2], 'b-')

            # interpolate control
            _us_zoh = scocp.zoh_controls(taus, uopt, _ts)
            ax.quiver(_ys[:,0], _ys[:,1], _ys[:,2], _us_zoh[:,0], _us_zoh[:,1], _us_zoh[:,2], color='r', length=0.1)

        ax.scatter(x0[0], x0[1], x0[2], marker='x', color='k', label='Initial state')
        ax.scatter(xf_guess[0], xf_guess[1], xf_guess[2], marker='^', color='k', label='Final state (guess)')
        ax.scatter(xf[0], xf[1], xf[2], marker='o', color='k', label='Final state')
        ax.plot(sol_lpo0.y[0,:], sol_lpo0.y[1,:], sol_lpo0.y[2,:], 'k-', lw=0.3)
        ax.plot(sol_lpo1.y[0,:], sol_lpo1.y[1,:], sol_lpo1.y[2,:], 'k-', lw=0.3)
        #ax.set_aspect('equal')
        ax.legend()

        ax_m = fig.add_subplot(2,3,2)
        ax_m.grid(True, alpha=0.5)
        for (_ts, _ys) in sols:
            ax_m.plot(_ys[:,7], _ys[:,6], 'b-')
        ax_m.axhline(sols[-1][1][-1,6], color='r', linestyle='--')
        ax_m.text(0.0, 0.01 + sols[-1][1][-1,6], f"m_f = {sols[-1][1][-1,6]:1.4f}", color='r')
        ax_m.set(xlabel="Time", ylabel="Mass")
        ax_m.legend()

        ax_u = fig.add_subplot(2,3,3)
        ax_u.grid(True, alpha=0.5)
        ax_u.step(xopt[:,7], np.concatenate((vopt[:,0], [0.0])), label="Gamma", where='post', color='k')
        ax_u.step(xopt[:,7], np.concatenate((uopt[:,0], [0.0])), label="ux", where='post', color='r')
        ax_u.step(xopt[:,7], np.concatenate((uopt[:,1], [0.0])), label="uy", where='post', color='b')
        ax_u.step(xopt[:,7], np.concatenate((uopt[:,2], [0.0])), label="uz", where='post', color='g')
        ax_u.set(xlabel="Time", ylabel="Control throttle")
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
        fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots/twobody_mass_freetf_rdv.png"), dpi=300)
    return


if __name__ == "__main__":
    test_scp_scipy_logmass(get_plot=True)
    plt.show()