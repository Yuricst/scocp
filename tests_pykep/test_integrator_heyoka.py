"""Test integrator class"""

import numpy as np
import pykep as pk

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import scocp
import scocp_pykep


def test_heyoka_integrator_cr3bp_impulsive():
    """Test `HeyokaIntegrator` class"""
    mu = 1.215058560962404e-02
    ta_dyn, ta_dyn_aug = scocp_pykep.get_heyoka_integrator_cr3bp(mu=mu, tol=1e-12)
    integrator = scocp_pykep.HeyokaIntegrator(nx=6, nu=3, ta=ta_dyn, ta_stm=ta_dyn_aug, impulsive=True)

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
    integrator = scocp_pykep.HeyokaIntegrator(nx=6, nu=3, ta=ta_dyn, ta_stm=ta_dyn_aug, impulsive=False)

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


def test_heyoka_integrator_twobody():
    """Test `HeyokaIntegrator` class"""
    # define canonical parameters
    GM_SUN = pk.MU_SUN           # Sun GM, m^3/s^-2
    MSTAR  = 800.0               # reference spacecraft mass
    ISP    = 3000.0              # specific impulse, s
    THRUST = 0.2                 # max thrust, kg.m/s^2
    G0     = pk.G0               # gravity at surface, m/s^2

    DU = pk.AU                   # length scale set to Sun-Earth distance, m
    VU = np.sqrt(GM_SUN / DU)    # velocity scale, m/s
    TU = DU / VU                 # time scale, s

    mu = GM_SUN / (VU**2 * DU)
    cex = ISP * G0 * (TU/DU)
    
    ta_dyn, ta_dyn_aug = scocp_pykep.get_heyoka_integrator_twobody_logmass(mu, cex, tol=1e-12, verbose=True)

    itg_heyoka = scocp_pykep.HeyokaIntegrator(
        nx=8,
        nu=4,
        nv=1,
        ta=ta_dyn,
        ta_stm=ta_dyn_aug,
        impulsive=False
    )
    
    # test against scipy integrator
    itg_scipy = scocp.ScipyIntegrator(
        nx=8,
        nu=4,
        nv=1,
        rhs=scocp.control_rhs_twobody_logmass_freetf,
        rhs_stm=scocp.control_rhs_twobody_logmass_freetf_stm,
        impulsive=False,
        args=(
            (mu, cex),                  # canonical gravitational param. & exhaust velocity of thruster
            [0.0,0.0,0.0,1.0,0.0]       # place-holder for control vector: [ax,ay,az,s,v]
        ),
        method='DOP853', reltol=1e-12, abstol=1e-12
    )

    # solve
    x0 = np.array([1.0, 0.2, 0.1, 0.2, 0.92, -0.5, np.log(1.0), 0.0])
    u = np.array([0.03, -0.04, 0.04, 1.0, np.sqrt(0.03**2 + -0.04**2 + 0.04**2)])
    ts, ys = itg_heyoka.solve([0.0, 0.1], x0, u=u)
    ts_scipy, ys_scipy = itg_scipy.solve([0.0, 0.1], x0, u=u)

    # checks
    print(f"max diff: {np.max(np.abs(ys[-1,:] - ys_scipy[-1,:])):1.4e}")
    assert np.abs(ts[-1] - ts_scipy[-1]) < 1e-15
    assert np.max(np.abs(ys[-1,:] - ys_scipy[-1,:])) < 1e-11
    return

if __name__ == "__main__":
    test_heyoka_integrator_twobody()
