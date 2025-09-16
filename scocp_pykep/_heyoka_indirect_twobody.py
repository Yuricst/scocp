"""Indirect optimal control two-body problem integrator with heyoka"""

import heyoka as hy
import numpy as np


def get_heyoka_integrator_twobody_indirect(tol = 1e-12, tol_var = 1e-12):
    """Built taylor integrator for two-body problem with indirect method

    Parameters are: `hy.par[0:5] = [mu, c1, c2, epsilon, lambda0]`
    
    Returns:
        (tuple): tuple of taylor integrators for state + costate & state + costate + STM
    """
    # The state
    x, y, z, vx, vy, vz, m = hy.make_vars("x", "y", "z", "vx", "vy", "vz", "m")
    # The costate
    lx, ly, lz, lvx, lvy, lvz, lm = hy.make_vars(
        "lx", "ly", "lz", "lvx", "lvy", "lvz", "lm"
    )
    # The controls
    u, ix, iy, iz = hy.make_vars("u", "ix", "iy", "iz")

    # Useful expressions
    r3 = (x**2 + y**2 + z**2) ** (1.5)
    lv_norm = hy.sqrt(lvx**2 + lvy**2 + lvz**2) 
    
    # Vectors for convenience of math manipulation
    lr = np.array([lx, ly, lz])
    lv = np.array([lvx, lvy, lvz])
    r = np.array([x, y, z])
    v = np.array([vx, vy, vz])
    i_vers = np.array([ix, iy, iz])

    # Dynamics
    fr = v
    fv = hy.par[1] * u / m * i_vers - (hy.par[0] / r3) * r
    fm = -hy.par[1] / hy.par[2] * u

    # Hamiltonian
    H_full = lr @ fr + lv @ fv + lm * fm + hy.par[4] * hy.par[1] / hy.par[2] * (u - hy.par[3] * hy.log(u * (1 - u)))
    # This is the shooting function must be derived manually per case
    rho = 1 - hy.par[2] * lv_norm / m / hy.par[4] - lm / hy.par[4]

    # Augmented equations of motion
    rhs = [
        hy.diff(H_full, var)
        for var in [lx, ly, lz, lvx, lvy, lvz, lm, x, y, z, vx, vy, vz, m]
    ]
    for j in range(7, 14):
        rhs[j] = -rhs[j]             # flip the sign for costates

    # We apply Pontryagin minimum principle (primer vector and u^* = 2eps / (rho + 2eps + sqrt(rho^2+4*eps^2)))
    argmin_H_full = {
        ix: -lvx / lv_norm,
        iy: -lvy / lv_norm,
        iz: -lvz / lv_norm,
        u: 2.
        * hy.par[3]
        / (rho + 2. * hy.par[3] + hy.sqrt(rho * rho + 4. * hy.par[3] * hy.par[3])),
    }
    rhs = hy.subs(rhs, argmin_H_full)       # substitute optimal control expressions into eom

    # We assemble the Taylor adaptive integrator
    full_state = [x, y, z, vx, vy, vz, m, lx, ly, lz, lvx, lvy, lvz, lm]
    sys = [(var, dvar) for var, dvar in zip(full_state, rhs)]
    ta = hy.taylor_adaptive(sys, state=[1.0] * 14, tol=tol)

    # variational integrator (i.e. "state" + STM)
    vsys = hy.var_ode_sys(                           
            sys,                                  # dynamical system
            full_state,                           # all variables
            # [lx, ly, lz, lvx, lvy, lvz, lm],      # Only for state variables
            order = 1)                            # STM

    # ta must come from vsys
    ta_var = hy.taylor_adaptive(
            vsys,
            state = [1.0] * 14 + [0.0] * (14*14),
            # state = [1.0] * 14 + [0.0] * (14*7),     
            time  = 0.0,
            tol = tol_var,                          
            compact_mode = True)
    return ta, ta_var


def get_twobody_indirect_functions():
    """Get functions for two-body indirect problem
    
    Returns:
        (tuple): tuple of functions for Hamiltonian, control, shooting function, and thrust direction
    """
    # The state
    x, y, z, vx, vy, vz, m = hy.make_vars("x", "y", "z", "vx", "vy", "vz", "m")
    # The costate
    lx, ly, lz, lvx, lvy, lvz, lm = hy.make_vars(
        "lx", "ly", "lz", "lvx", "lvy", "lvz", "lm"
    )
    # The controls
    u, ix, iy, iz = hy.make_vars("u", "ix", "iy", "iz")

    # Useful expressions
    r3 = (x**2 + y**2 + z**2) ** (1.5)
    lv_norm = hy.sqrt(lvx**2 + lvy**2 + lvz**2) 

    # Vectors for convenience of math manipulation
    lr = np.array([lx, ly, lz])
    lv = np.array([lvx, lvy, lvz])
    r = np.array([x, y, z])
    v = np.array([vx, vy, vz])
    i_vers = np.array([ix, iy, iz])

    # Dynamics
    fr = v
    fv = hy.par[1] * u / m * i_vers - (hy.par[0] / r3) * r
    fm = -hy.par[1] / hy.par[2] * u

    # Hamiltonian
    H_full = lr @ fr + lv @ fv + lm * fm + hy.par[4] * hy.par[1] / hy.par[2] * (u - hy.par[3] * hy.log(u * (1 - u)))
    # This is the shooting function must be derived manually per case
    rho = 1 - hy.par[2] * lv_norm / m / hy.par[4] - lm / hy.par[4]

    # Augmented equations of motion
    rhs = [
        hy.diff(H_full, var)
        for var in [lx, ly, lz, lvx, lvy, lvz, lm, x, y, z, vx, vy, vz, m]
    ]
    for j in range(7, 14):
        rhs[j] = -rhs[j]             # flip the sign for costates

    # We apply Pontryagin minimum principle (primer vector and u^* = 2eps / (rho + 2eps + sqrt(rho^2+4*eps^2)))
    argmin_H_full = {
        ix: -lvx / lv_norm,
        iy: -lvy / lv_norm,
        iz: -lvz / lv_norm,
        u: 2.
        * hy.par[3]
        / (rho + 2. * hy.par[3] + hy.sqrt(rho * rho + 4. * hy.par[3] * hy.par[3])),
    }
    H = hy.subs(H_full, argmin_H_full)

    # We compile the Hamiltonian into a C function (to be called with pars = [mu, c1, c2, eps, l0])
    H_func = hy.cfunc([H], [x, y, z, vx, vy, vz, m, lx, ly, lz, lvx, lvy, lvz, lm])
    # We compile the thrust direction
    u_func = hy.cfunc(
        [argmin_H_full[u]], [x, y, z, vx, vy, vz, m, lx, ly, lz, lvx, lvy, lvz, lm]
    )
    # We compile the SF
    rho_func = hy.cfunc([rho], [x, y, z, vx, vy, vz, m, lx, ly, lz, lvx, lvy, lvz, lm])
    # We compile also the thrust direction
    i_vers_func = hy.cfunc(
        [argmin_H_full[ix], argmin_H_full[iy], argmin_H_full[iz]], [lvx, lvy, lvz]
    )
    return H_func, u_func, rho_func, i_vers_func

