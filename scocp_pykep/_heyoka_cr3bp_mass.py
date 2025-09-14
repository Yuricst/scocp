"""CR3BP equations of motion + mass with heyoka"""

import heyoka as hy
import numpy as np

def get_heyoka_integrator_cr3bp_mass(
    mu,
    tol = 1e-12,
    high_accuracy = True,
    impulsive = True,
):
    """Build heyoka taylor adaptive integrator objects for CR3BP + mass
    
    States are `[x,y,z,vx,vy,vz,mass]`

    Controls are `hy.par[-4:] = [ux, uy, uz, umag]` where `umag` is the control throttle magnitude.

    Args:
        mu (float): CR3BP mass ratio
        c1 (float): maximum thrust
        c2 (float): exhaust velocity, Isp * g0
    """
    # Create the symbolic variables.
    symbols_state = ["x", "y", "z", "vx", "vy", "vz", "mass"]
    x = np.array(hy.make_vars(*symbols_state))
    npar = 2    # non-control parameters are [c1, c2]
    c1, c2 = hy.par[0], hy.par[1]

    # This will contain the r.h.s. of the equations
    f = []
    r1 = ((x[0] + mu)**2 + x[1]**2 + x[2]**2)**(1/2.)
    r2 = ((x[0] - 1 + mu)**2 + x[1]**2 + x[2]**2)**(1/2.)

    # The equations of motion
    f.append(x[3])
    f.append(x[4])
    f.append(x[5])
    if impulsive is True:
        f.append(x[0] - (1-mu)/r1**3 * (x[0]+mu) - mu/r2**3 * (x[0]-1+mu) + 2*x[4])
        f.append(x[1] - (1-mu)/r1**3 * x[1]      - mu/r2**3 * x[1]        - 2*x[3])
        f.append(     - (1-mu)/r1**3 * x[2]      - mu/r2**3 * x[2])
    else:
        f.append(x[0] - (1-mu)/r1**3 * (x[0]+mu) - mu/r2**3 * (x[0]-1+mu) + 2*x[4] + c1/x[6] * hy.par[npar+0])
        f.append(x[1] - (1-mu)/r1**3 * x[1]      - mu/r2**3 * x[1]        - 2*x[3] + c1/x[6] * hy.par[npar+1])
        f.append(     - (1-mu)/r1**3 * x[2]      - mu/r2**3 * x[2]                 + c1/x[6] * hy.par[npar+2])
    f.append(-c1/c2 * hy.par[npar+3])
    
    # construct integrator for dynamics only
    dyn = []
    for state, rhs in zip(x,f):
        dyn.append((state, rhs))

    # construct integrator for state dynamics
    ta_dyn = hy.taylor_adaptive(
        dyn,                                             # dynamics
        [-0.45, 0.80, 0.00, -0.80, -0.45, 0.58, 1.0],    # initial conditions
        tol = tol,
        high_accuracy = high_accuracy,
    )

    # define derivative of Phi_A
    symbols_phi = []
    for i in range(7):
        for j in range(7):
            symbols_phi.append("phi_"+str(i)+str(j))  
    phi = np.array(hy.make_vars(*symbols_phi)).reshape((7,7))

    dfdx = []
    for i in range(7):
        for j in range(7):
            dfdx.append(hy.diff(f[i],x[j]))
    dfdx = np.array(dfdx).reshape((7,7))
    dphidt = dfdx@phi       # (variational) equations of motion

    if impulsive:
        dyn_aug = []
        for state, rhs in zip(x,f):
            dyn_aug.append((state, rhs))
        for state, rhs in zip(phi.reshape((49,)),dphidt.reshape((49,))):
            dyn_aug.append((state, rhs))

        # construct integrator for augmented dynamics
        ta_dyn_aug = hy.taylor_adaptive(
            dyn_aug,                                                                      # augmented dynamics
            [-0.45, 0.80, 0.00, -0.80, -0.45, 0.58, 1.0] + list(np.eye(7).flatten()),     # initial conditions
            tol = tol,
            high_accuracy = high_accuracy,
        )

    else:
        dfdu = []
        for i in range(7):
            for j in range(4):
                dfdu.append(hy.diff(f[i],hy.par[npar+j]))
        dfdu = np.array(dfdu).reshape((7,4))
        
        # define derivative of PhiB
        symbols_PhiB = []
        for i in range(7):
            for j in range(4):
                symbols_PhiB.append("PhiB_"+str(i)+str(j))  
        PhiB = np.array(hy.make_vars(*symbols_PhiB)).reshape((7,4))
        dPhiBdt = dfdx @ PhiB + dfdu

        dyn_aug = []
        for state, rhs in zip(x,f):
            dyn_aug.append((state, rhs))
        for state, rhs in zip(phi.reshape((49,)),dphidt.reshape((49,))):
            dyn_aug.append((state, rhs))
        for state, rhs in zip(PhiB.reshape((28,)),dPhiBdt.reshape((28,))):
            dyn_aug.append((state, rhs))
        
        # construct integrator for augmented dynamics
        ta_dyn_aug = hy.taylor_adaptive(
            dyn_aug,                                    # augmented dynamics
            [-0.45, 0.80, 0.00, -0.80, -0.45, 0.58, 1.0] +\
                list(np.eye(7).flatten()) +\
                list(np.zeros(7*4).flatten()),                          # initial conditions
            tol = tol,
            high_accuracy = high_accuracy,
        )
    return ta_dyn, ta_dyn_aug
