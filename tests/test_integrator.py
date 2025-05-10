"""Test integrator class"""

import matplotlib.pyplot as plt
import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import scocp

def test_scipy_integrator_cr3bp_impulsive():
    """Test `ScipyIntegrator` class for impulsive CR3BP dynamics"""
    mu = 1.215058560962404e-02
    integrator = scocp.ScipyIntegrator(nx=6, nu=3, rhs=scocp.rhs_cr3bp, rhs_stm=scocp.rhs_cr3bp_stm, args=(mu,),
                                       method='DOP853', reltol=1e-12, abstol=1e-12)

    # propagate uncontrolled and controlled dynamics
    x0 = np.array([
        1.0809931218390707E+00,
        0.0,
        -2.0235953267405354E-01,
        0.0,
        -1.9895001215078018E-01,
        0.0])
    period_0 = 2.3538670417546639E+00
    sol_lpo0 = integrator.solve([0.0, period_0], x0, stm=True, get_ODESolution=True)

    assert sol_lpo0.y[0:6,-1].shape == (6,)
    assert np.max(np.abs((sol_lpo0.y[0:6,-1] - x0))) < 1e-11
    return


def test_scipy_integrator_cr3bp_continuous():
    """Test `ScipyIntegrator` class for continuous CR3BP dynamics"""
    mu = 1.215058560962404e-02
    integrator = scocp.ScipyIntegrator(nx=6, nu=3, rhs=scocp.control_rhs_cr3bp, rhs_stm=scocp.control_rhs_cr3bp_stm,
                                       impulsive=False, args=(mu,[0.0,0.0,0.0]),
                                       method='DOP853', reltol=1e-12, abstol=1e-12)

    # propagate uncontrolled and controlled dynamics
    x0 = np.array([
        1.0809931218390707E+00,
        0.0,
        -2.0235953267405354E-01,
        0.0,
        -1.9895001215078018E-01,
        0.0])
    period_0 = 2.3538670417546639E+00
    sol_lpo0 = integrator.solve([0.0, period_0], x0, stm=False, get_ODESolution=True)

    assert sol_lpo0.y[0:6,-1].shape == (6,)
    assert np.max(np.abs((sol_lpo0.y[0:6,-1] - x0))) < 1e-11
    return


def test_scipy_integrator_twobody_mass_freetf():
    """Test `ScipyIntegrator` class for twobody dynamics with mass and time"""
    mu = 1.0
    c1 = 1e-2
    c2 = 1e-2
    integrator = scocp.ScipyIntegrator(nx=8, nu=5, rhs=scocp.control_rhs_twobody_mass_freetf,
                                       rhs_stm=scocp.control_rhs_twobody_mass_freetf_stm,
                                       impulsive=False, args=((mu, c1, c2), [0.0, 0.0, 0.0, 1.0, 0.0]),
                                       method='DOP853', reltol=1e-12, abstol=1e-12)
    
    
    x0 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.02, 1.0, 0.0])   # [r,v,mass,t]
    tspan = [0.0, 2*np.pi]
    u = np.array([0.04, 0.5, -0.2, tspan[1], np.sqrt(0.04**2 + 0.05**2 + 0.02**2)])
    sol_lpo0 = integrator.solve(tspan, x0, u=u, stm=True, get_ODESolution=True)

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1,2,1, projection='3d')
    ax.plot(sol_lpo0.y[0,:], sol_lpo0.y[1,:], sol_lpo0.y[2,:])
    ax = fig.add_subplot(1,2,2)
    ax.plot(sol_lpo0.y[7,:], sol_lpo0.y[6,:])
    plt.tight_layout()
    return


if __name__ == "__main__":
    test_scipy_integrator_twobody_mass_freetf()
    plt.show()
