"""Two-body dybamics integrator with heyoka"""

import time 
import heyoka as hy
import numpy as np


def get_heyoka_integrator_twobody(
    mu,
    cex,
    tol = 1e-12,
    high_accuracy = True,
    verbose = False,
):
    """Get heyoka integrator for two-body dynamics with mass and time
    
    state = [x,y,z,vx,vy,vz,log(mass),t] 
    u = [ax,ay,az,s,v]
    
    Paramers are:
    hy.par = [mu, cex, ax, ay, az, s, Gamma]
    where the last 5 entries are control inputs.
    """
    tstart = time.time()

    # Create the symbolic variables.
    symbols_state = ["x", "y", "z", "vx", "vy", "vz", "logm", "t"]
    x = np.array(hy.make_vars(*symbols_state))
    nx = len(x)     # dimension of state
    nu = 5          # dimension of control inputs

    # This will contain the r.h.s. of the equations
    r = (x[0]**2 + x[1]**2 + x[2]**2)**(1/2.)
    f = []
    f.append(hy.par[5] * x[3])                                       # dx/dtau
    f.append(hy.par[5] * x[4])                                       # dy/dtau
    f.append(hy.par[5] * x[5])                                       # dz/dtau
    f.append(hy.par[5] * (-hy.par[0] * x[0] / r**3 + hy.par[2]))     # dvx/dtau
    f.append(hy.par[5] * (-hy.par[0] * x[1] / r**3 + hy.par[3]))     # dvy/dtau
    f.append(hy.par[5] * (-hy.par[0] * x[2] / r**3 + hy.par[4]))     # dvz/dtau
    f.append(hy.par[5] * (-1/hy.par[1] * hy.par[6]))                 # dlog(mass)/dtau
    f.append(hy.par[5])                                              # dt/dtau

    # construct dynamics
    dyn = []
    for state, rhs in zip(x,f):
        dyn.append((state, rhs))

    # ----------------------------- Integrator for state only ----------------------------- #
    if verbose:
        print(f"Building integrator for state only", end=" ... ")
    ta_dyn = hy.taylor_adaptive(
        dyn,                                                  # dynamics
        [-0.45, 0.80, 0.00, -0.80, -0.45, 0.58, 0.0, 0.0],    # initial conditions
        tol = tol,
        high_accuracy = high_accuracy,
    )
    ta_dyn.pars[0] = mu
    ta_dyn.pars[1] = cex
    ta_dyn.pars[2] = 0.0
    ta_dyn.pars[3] = 0.0
    ta_dyn.pars[4] = 0.0
    ta_dyn.pars[5] = 1.0
    ta_dyn.pars[6] = 0.0
    if verbose:
        print(f"Done! Elapsed time: {time.time() - tstart:1.4f} seconds")

    # ---------------------------- Integrator for state + STM ---------------------------- #
    if verbose:
        print(f"Building integrator for state + STM", end=" ... ")
    # define derivative of Phi_A
    symbols_PhiA = []
    for i in range(nx):
        for j in range(nx):
            symbols_PhiA.append("PhiA_"+str(i)+str(j))  
    PhiA = np.array(hy.make_vars(*symbols_PhiA)).reshape((nx,nx))

    dfdx = []
    for i in range(nx):
        for j in range(nx):
            dfdx.append(hy.diff(f[i],x[j]))
    dfdx = np.array(dfdx).reshape((nx,nx))
    dPhiAdt = dfdx@PhiA       # (variational) equations of motion

    dfdu = []
    for i in range(nx):
        for j in range(nu):
            dfdu.append(hy.diff(f[i],hy.par[j+2]))
    dfdu = np.array(dfdu).reshape((nx,nu))

    # define derivative of PhiB
    symbols_PhiB = []
    for i in range(nx):
        for j in range(nu):
            symbols_PhiB.append("PhiB_"+str(i)+str(j))
    PhiB = np.array(hy.make_vars(*symbols_PhiB)).reshape((nx,nu))
    dPhiBdt = dfdx @ PhiB + dfdu

    dyn_aug = []
    for state, rhs in zip(x,f):
        dyn_aug.append((state, rhs))
    for state, rhs in zip(PhiA.reshape((nx*nx,)),dPhiAdt.reshape((nx*nx,))):
        dyn_aug.append((state, rhs))
    for state, rhs in zip(PhiB.reshape((nx*nu,)),dPhiBdt.reshape((nx*nu,))):
        dyn_aug.append((state, rhs))
    
    # construct integrator for augmented dynamics
    ta_dyn_aug = hy.taylor_adaptive(
        dyn_aug,                                                # augmented dynamics
        [-0.45, 0.80, 0.00, -0.80, -0.45, 0.58, 0.0, 0.0] +\
            list(np.eye(nx).flatten()) +\
            list(np.zeros(nx*nu).flatten()),                    # initial conditions
        tol = tol,
        high_accuracy = high_accuracy,
    )
    ta_dyn_aug.pars[0] = mu
    ta_dyn_aug.pars[1] = cex
    ta_dyn_aug.pars[2] = 0.0
    ta_dyn_aug.pars[3] = 0.0
    ta_dyn_aug.pars[4] = 0.0
    ta_dyn_aug.pars[5] = 1.0
    ta_dyn_aug.pars[6] = 0.0
    if verbose:
        print(f"Done! Elapsed time: {time.time() - tstart:1.4f} seconds")
    return ta_dyn, ta_dyn_aug