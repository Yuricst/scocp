"""Test integrator class"""

import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import scocp
import scocp_pykep


def test_heyoka_integrator_cr3bp_impulsive():
    """Test `HeyokaIntegrator` class"""
    mu = 1.215058560962404e-02
    ta_dyn, ta_dyn_aug = scocp_pykep.get_heyoka_integrator_cr3bp(mu=mu, tol=1e-12)
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

    ts, ys = integrator.solve([0.0, period_0], x0)
    assert ys.shape == (2,6)
    assert np.max(np.abs((ys[0,:] - x0))) < 1e-11
    return


def test_heyoka_integrator_cr3bp_continuous():
    """Test `HeyokaIntegrator` class"""
    mu = 1.215058560962404e-02
    ta_dyn, ta_dyn_aug = scocp_pykep.get_heyoka_integrator_cr3bp(mu=mu, tol=1e-12, impulsive=False)
    integrator = scocp.HeyokaIntegrator(nx=6, nu=3, ta=ta_dyn, ta_stm=ta_dyn_aug, impulsive=False)

    # propagate uncontrolled and controlled dynamics
    x0 = np.array([
        1.0809931218390707E+00,
        0.0,
        -2.0235953267405354E-01,
        0.0,
        -1.9895001215078018E-01,
        0.0])
    period_0 = 2.3538670417546639E+00

    ts, ys = integrator.solve([0.0, period_0], x0, stm=True)
    assert ys.shape == (2,60)
    assert np.max(np.abs((ys[0,0:6] - x0))) < 1e-11
    return


if __name__ == "__main__":
    test_heyoka_integrator_cr3bp_impulsive()
