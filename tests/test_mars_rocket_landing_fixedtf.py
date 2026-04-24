"""Regression test for the fixed-time 3-DoF Mars rocket landing benchmark."""

import os
import sys

import cvxpy as cp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import scocp


JULIA_REFERENCE_COST = 429.4077878390665
JULIA_REFERENCE_FINAL_MASS = 1531.2327108960533


def test_mars_rocket_landing_fixedtf(tmp_path):
    problem = scocp.FixedTimeMars3DoFRocketLanding(solver=cp.CLARABEL, verbose_solver=False)
    solution = problem.solve()

    assert solution.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    assert np.isclose(solution.cost, JULIA_REFERENCE_COST, rtol=1e-4, atol=1e-3)
    assert np.isclose(
        solution.mass[-1],
        JULIA_REFERENCE_FINAL_MASS,
        rtol=1e-4,
        atol=1e-3,
    )
    assert np.linalg.norm(solution.r[-1]) <= 1e-6
    assert np.linalg.norm(solution.v[-1]) <= 1e-6

    control_t, control, min_accel, max_accel = problem._control_plot_series(solution)
    assert control_t.shape == solution.times[:-1].shape
    assert control.shape == solution.xi.shape
    assert min_accel.shape == solution.xi.shape
    assert max_accel.shape == solution.xi.shape
    assert np.allclose(control, solution.xi)
    assert np.all(control >= min_accel - 1e-5)
    assert np.all(control <= max_accel + 1e-5)
    assert np.max(max_accel) < 10.0
    assert np.all(solution.mass[:-1] >= problem.m_dry - 1e-6)

    glide_values = (problem._H_gs @ solution.r.T).T
    assert np.max(glide_values) <= 1e-6
    assert np.max(np.linalg.norm(solution.v, axis=1)) <= problem.v_max + 1e-6

    simulation = problem.simulate(solution, dt=5e-2)
    assert np.linalg.norm(simulation.r[-1]) <= 5e-1
    assert np.linalg.norm(simulation.v[-1]) <= 5e-2

    plot_path = tmp_path / "mars_rocket_landing_3d_fixedtf.png"
    fig = problem.plot_summary(solution, sim=simulation, path=str(plot_path))
    assert plot_path.exists()
    fig.clf()
