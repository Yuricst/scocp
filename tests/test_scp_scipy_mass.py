"""Test SCP continuous transfer in CR3BP with mass dynamics"""

import copy
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import scocp


def print_matrix(matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            print(f"{matrix[i,j]: 1.4e}", end=" ")
        print()
    print()
    return

def test_eom_cr3bp_mass(verbose=False):
    mu = 1.215058560962404e-02
    c1 = 0.1
    c2 = 0.1
    uvec = [0.2, 0.3, 0.4, np.sqrt(0.2**2 + 0.3**2 + 0.4**2)]
    integrator = scocp.ScipyIntegrator(nx=7, nu=3, nv=1,
                                       rhs=scocp.control_rhs_cr3bp_mass,
                                       rhs_stm=scocp.control_rhs_cr3bp_mass_stm,
                                       impulsive=False, args=((mu,c1,c2),uvec),
                                       method='DOP853', reltol=1e-12, abstol=1e-12)

    # integrate STMs numerically
    x0 = np.array([
        1.0809931218390707E+00,
        0.0,
        -2.0235953267405354E-01,
        0.0,
        -1.9895001215078018E-01,
        0.0,
        1.0])
    period_0 = 2.3538670417546639E+00
    sol_stm = integrator.solve([0, period_0], x0, get_ODESolution=True, stm=True)
    # print(f"sol_stm.y.shape = {sol_stm.y.shape}")

    xf = sol_stm.y[0:7,-1]
    Phi_A = sol_stm.y[7:56,-1].reshape(7,7)
    Phi_B = sol_stm.y[56:84,-1].reshape(7,4)
    if verbose:
        print(f"xf = \n{xf}\n")
        print(f"Phi_A = \n")
        print_matrix(Phi_A)
        print(f"Phi_B = \n")
        print_matrix(Phi_B)

    # numerically compute Phi_A, Phi_B
    Phi_A_num = np.zeros((7,7))
    Phi_B_num = np.zeros((7,4))
    h = 1e-6
    for i in range(7):
        x0_plus = copy.deepcopy(x0)
        x0_plus[i] += h
        sol_plus = integrator.solve([0, period_0], x0_plus, get_ODESolution=True, stm=False)

        x0_minus = copy.deepcopy(x0)
        x0_minus[i] -= h
        sol_minus = integrator.solve([0, period_0], x0_minus, get_ODESolution=True, stm=False)
        Phi_A_num[:,i] = (sol_plus.y[0:7,-1] - sol_minus.y[0:7,-1]) / (2*h)

    for i in range(4):
        integrator_copy = copy.deepcopy(integrator)
        uvec_plus = copy.deepcopy(uvec)
        uvec_plus[i] += h
        integrator_copy.args[-1][:] = uvec_plus[:]
        sol_plus = integrator_copy.solve([0, period_0], x0, get_ODESolution=True, stm=False)

        integrator_copy = copy.deepcopy(integrator)
        uvec_minus = copy.deepcopy(uvec)
        uvec_minus[i] -= h
        integrator_copy.args[-1][:] = uvec_minus[:]
        sol_minus = integrator_copy.solve([0, period_0], x0, get_ODESolution=True, stm=False)
        Phi_B_num[:,i] = (sol_plus.y[0:7,-1] - sol_minus.y[0:7,-1]) / (2*h)

    if verbose:
        print(f"Phi_A_num = \n")
        print_matrix(Phi_A_num)
        print(f"Phi_B_num = \n")
        print_matrix(Phi_B_num)

        print(f"Max diff in Phi_A = {np.max(np.abs(Phi_A_num - Phi_A)):1.4e}")
        print(f"Max diff in Phi_B = {np.max(np.abs(Phi_B_num - Phi_B)):1.4e}")
    assert np.max(np.abs(Phi_A_num - Phi_A)) <= h
    assert np.max(np.abs(Phi_B_num - Phi_B)) <= h
    return


def test_scp_scipy_mass(get_plot=False):
    """Test SCP continuous transfer"""
    mu = 1.215058560962404e-02
    c1 = 0.1
    c2 = 0.1
    integrator = scocp.ScipyIntegrator(nx=7, nu=3, nv=1,
                                       rhs=scocp.control_rhs_cr3bp_mass,
                                       rhs_stm=scocp.control_rhs_cr3bp_mass_stm,
                                       impulsive=False, args=((mu,c1,c2),[0.0,0.0,0.0,0.0]),
                                       method='DOP853', reltol=1e-12, abstol=1e-12)
    
    # propagate uncontrolled and controlled dynamics
    x0 = np.array([
        1.0809931218390707E+00,
        0.0,
        -2.0235953267405354E-01,
        0.0,
        -1.9895001215078018E-01,
        0.0,
        1.0])
    period_0 = 2.3538670417546639E+00
    sol_lpo0 = integrator.solve([0, period_0], x0, get_ODESolution=True)

    xf = np.array([
        1.1648780946517576,
        0.0,
        -1.1145303634437023E-1,
        0.0,
        -2.0191923237095796E-1,
        0.0,
        1.0])
    period_f = 3.3031221822879884
    sol_lpo1 = integrator.solve([0, period_f], xf, get_ODESolution=True)
    
    # transfer problem discretization
    N = 50
    tf = (period_0 + period_f) / 2
    times = np.linspace(0, tf, N)

    # create subproblem
    trust_region_radius_x = 0.1
    trust_region_radius_u = None
    problem = scocp.FixedTimeContinuousRdvMass(x0, xf[0:6], c1, c2, integrator, times,
                                           augment_Gamma=True,
                                           trust_region_radius_x=trust_region_radius_x,
                                           trust_region_radius_u=trust_region_radius_u)

    # create initial guess
    print(f"Preparing initial guess...")
    sol_initial = integrator.solve([0, times[-1]], x0, t_eval=times, get_ODESolution=True)
    sol_final  = integrator.solve([0, times[-1]], xf, t_eval=times, get_ODESolution=True)

    alphas = np.linspace(1,0,N)
    xbar = (np.multiply(sol_initial.y, np.tile(alphas, (7,1))) + np.multiply(sol_final.y, np.tile(1-alphas, (7,1)))).T
    xbar[:,6] = np.linspace(1.0, 0.8, N)
    xbar[0,:] = x0              # overwrite initial state
    xbar[-1,:6] = xf[:6]        # overwrite final state
    ubar = np.zeros((N-1,3))

    # solve subproblem
    vbar = np.sum(ubar, axis=1).reshape(-1,1)
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

    # evaluate nonlinear violations
    geq_nl_opt, sols = problem.evaluate_nonlinear_dynamics(xopt, uopt, vopt, steps=5)
    assert np.max(np.abs(geq_nl_opt)) <= tol_feas
    
    # evaluate solution
    if (get_plot is True) and (summary_dict["status"] != "CPFailed"):
        _, sols_ig = problem.evaluate_nonlinear_dynamics(xbar, ubar, vbar, steps=5)
    
        # plot results
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(2,3,1,projection='3d')
        for (_ts, _ys) in sols_ig:
            ax.plot(_ys[:,0], _ys[:,1], _ys[:,2], '--', color='grey')
        for (_ts, _ys) in sols:
            ax.plot(_ys[:,0], _ys[:,1], _ys[:,2], 'b-')

            # interpolate control
            arrow_scale = 0.1
            _us_zoh = scocp.zoh_controls(times, uopt, _ts)
            ax.quiver(_ys[:,0], _ys[:,1], _ys[:,2],
                _us_zoh[:,0]*arrow_scale, _us_zoh[:,1]*arrow_scale, _us_zoh[:,2]*arrow_scale, color='r', length=0.5)

        ax.scatter(x0[0], x0[1], x0[2], marker='x', color='k', label='Initial state')
        ax.scatter(xf[0], xf[1], xf[2], marker='o', color='k', label='Final state')
        ax.plot(sol_lpo0.y[0,:], sol_lpo0.y[1,:], sol_lpo0.y[2,:], 'k-', lw=0.3)
        ax.plot(sol_lpo1.y[0,:], sol_lpo1.y[1,:], sol_lpo1.y[2,:], 'k-', lw=0.3)
        ax.set_aspect('equal')
        ax.legend()

        ax_u = fig.add_subplot(2,3,2)
        ax_u.grid(True, alpha=0.5)
        ax_u.step(times, np.concatenate((vopt[:,0], [0.0])), label="Gamma", where='post', color='k')
        ax_u.set(xlabel="Time", ylabel="Control")
        ax_u.legend()

        ax_m = fig.add_subplot(2,3,3)
        ax_m.grid(True, alpha=0.5)
        for (_ts, _ys) in sols:
            ax_m.plot(_ts, _ys[:,6], 'k-')
        ax_m.set(xlabel="Time", ylabel="Mass")
        ax_m.legend()

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
        fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots/scp_scipy_mass_transfer.png"), dpi=300)
    return


if __name__ == "__main__":
    test_eom_cr3bp_mass(verbose=True)
    test_scp_scipy_mass(get_plot=True)
    plt.show()