"""CR3BP equations of motion with heyoka"""

import heyoka as hy
import numpy as np

def get_heyoka_integrator_cr3bp(
    mu,
    tol = 1e-12,
    high_accuracy = True,
    impulsive = True,
):
    """Build heyoka taylor adaptive integrator objects for CR3BP
    
    Controls are `hy.par[0:3] = [ax, ay, az]`.
    """
    # Create the symbolic variables.
    symbols_state = ["x", "y", "z", "vx", "vy", "vz"]
    x = np.array(hy.make_vars(*symbols_state))

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
        f.append(x[0] - (1-mu)/r1**3 * (x[0]+mu) - mu/r2**3 * (x[0]-1+mu) + 2*x[4] + hy.par[0])
        f.append(x[1] - (1-mu)/r1**3 * x[1]      - mu/r2**3 * x[1]        - 2*x[3] + hy.par[1])
        f.append(     - (1-mu)/r1**3 * x[2]      - mu/r2**3 * x[2]                 + hy.par[2])

    # construct integrator for dynamics only
    dyn = []
    for state, rhs in zip(x,f):
        dyn.append((state, rhs))

    # construct integrator for state dynamics
    ta_dyn = hy.taylor_adaptive(
        dyn,                                        # dynamics
        [-0.45, 0.80, 0.00, -0.80, -0.45, 0.58],    # initial conditions
        tol = tol,
        high_accuracy = high_accuracy,
    )

    # define derivative of Phi_A
    symbols_phi = []
    for i in range(6):
        for j in range(6):
            symbols_phi.append("phi_"+str(i)+str(j))  
    phi = np.array(hy.make_vars(*symbols_phi)).reshape((6,6))

    dfdx = []
    for i in range(6):
        for j in range(6):
            dfdx.append(hy.diff(f[i],x[j]))
    dfdx = np.array(dfdx).reshape((6,6))
    dphidt = dfdx@phi       # (variational) equations of motion

    if impulsive:
        dyn_aug = []
        for state, rhs in zip(x,f):
            dyn_aug.append((state, rhs))
        for state, rhs in zip(phi.reshape((36,)),dphidt.reshape((36,))):
            dyn_aug.append((state, rhs))

        # construct integrator for augmented dynamics
        ta_dyn_aug = hy.taylor_adaptive(
            dyn_aug,                                                                 # augmented dynamics
            [-0.45, 0.80, 0.00, -0.80, -0.45, 0.58] + list(np.eye(6).flatten()),     # initial conditions
            tol = tol,
            high_accuracy = high_accuracy,
        )

    else:
        # define derivative of PhiB
        symbols_PhiB = []
        for i in range(6):
            for j in range(3):
                symbols_PhiB.append("PhiB_"+str(i)+str(j))  
        PhiB = np.array(hy.make_vars(*symbols_PhiB)).reshape((6,3))
        dPhiBdt = dfdx @ PhiB + np.vstack((np.zeros((3,3)),np.eye(3)))

        dyn_aug = []
        for state, rhs in zip(x,f):
            dyn_aug.append((state, rhs))
        for state, rhs in zip(phi.reshape((36,)),dphidt.reshape((36,))):
            dyn_aug.append((state, rhs))
        for state, rhs in zip(PhiB.reshape((18,)),dPhiBdt.reshape((18,))):
            dyn_aug.append((state, rhs))
        
        # construct integrator for augmented dynamics
        ta_dyn_aug = hy.taylor_adaptive(
            dyn_aug,                                    # augmented dynamics
            [-0.45, 0.80, 0.00, -0.80, -0.45, 0.58] +\
                list(np.eye(6).flatten()) +\
                list(np.zeros(6*3).flatten()),                          # initial conditions
            tol = tol,
            high_accuracy = high_accuracy,
        )
    return ta_dyn, ta_dyn_aug



def get_heyoka_integrator_cr3bp_freetf(
    mu,
    tol = 1e-12,
    high_accuracy = True,
    impulsive = True,
):
    """Build heyoka taylor adaptive integrator objects for CR3BP with free final time
    
    Controls are `hy.par[0:4] = [ax, ay, az, s]` where s is the time-dilation factor.
    """
    # Create the symbolic variables.
    symbols_state = ["x", "y", "z", "vx", "vy", "vz", "t"]
    x = np.array(hy.make_vars(*symbols_state))

    # This will contain the r.h.s. of the equations
    f = []
    r1 = ((x[0] + mu)**2 + x[1]**2 + x[2]**2)**(1/2.)
    r2 = ((x[0] - 1 + mu)**2 + x[1]**2 + x[2]**2)**(1/2.)

    # The equations of motion
    f.append(hy.par[3] * x[3])
    f.append(hy.par[3] * x[4])
    f.append(hy.par[3] * x[5])
    if impulsive is True:
        f.append(hy.par[3] * (x[0] - (1-mu)/r1**3 * (x[0]+mu) - mu/r2**3 * (x[0]-1+mu) + 2*x[4]))
        f.append(hy.par[3] * (x[1] - (1-mu)/r1**3 * x[1]      - mu/r2**3 * x[1]        - 2*x[3]))
        f.append(hy.par[3] * (     - (1-mu)/r1**3 * x[2]      - mu/r2**3 * x[2]))
    else:
        f.append(hy.par[3] * (x[0] - (1-mu)/r1**3 * (x[0]+mu) - mu/r2**3 * (x[0]-1+mu) + 2*x[4] + hy.par[0]))
        f.append(hy.par[3] * (x[1] - (1-mu)/r1**3 * x[1]      - mu/r2**3 * x[1]        - 2*x[3] + hy.par[1]))
        f.append(hy.par[3] * (     - (1-mu)/r1**3 * x[2]      - mu/r2**3 * x[2]                 + hy.par[2]))
    f.append(hy.par[3])

    # construct integrator for dynamics only
    dyn = []
    for state, rhs in zip(x,f):
        dyn.append((state, rhs))

    # construct integrator for state dynamics
    ta_dyn = hy.taylor_adaptive(
        dyn,                                             # dynamics
        [-0.45, 0.80, 0.00, -0.80, -0.45, 0.58, 0.0],    # initial conditions
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
            [-0.45, 0.80, 0.00, -0.80, -0.45, 0.58, 0.0] + list(np.eye(7).flatten()),     # initial conditions
            tol = tol,
            high_accuracy = high_accuracy,
        )

    else:
        # define derivative of PhiB
        symbols_PhiB = []
        for i in range(7):
            for j in range(4):
                symbols_PhiB.append("PhiB_"+str(i)+str(j))  
        PhiB = np.array(hy.make_vars(*symbols_PhiB)).reshape((7,4))
        B_accel = np.hstack((
            np.vstack((np.zeros((3,3)), hy.par[3]*np.eye(3))),              # usual 6-by-3
            np.array([_el/hy.par[3] for _el in f[0:6]]).reshape(-1,1),      # 6-by-1 dynamics, without the time-dilation factor
        ))
        dPhiBdt = dfdx @ PhiB + np.vstack((B_accel, np.array([[0.0,0.0,0.0,1.0]])))

        dyn_aug = []
        for state, rhs in zip(x,f):                                             # state derivatives
            dyn_aug.append((state, rhs))
        for state, rhs in zip(phi.reshape((49,)),dphidt.reshape((49,))):        # state STM derivatives
            dyn_aug.append((state, rhs))
        for state, rhs in zip(PhiB.reshape((28,)),dPhiBdt.reshape((28,))):      # control STM derivatives
            dyn_aug.append((state, rhs))
        
        # construct integrator for augmented dynamics
        ta_dyn_aug = hy.taylor_adaptive(
            dyn_aug,                                                    # augmented dynamics
            [-0.45, 0.80, 0.00, -0.80, -0.45, 0.58, 0.0] +\
                list(np.eye(7).flatten()) +\
                list(np.zeros(7*4).flatten()),                          # initial conditions
            tol = tol,
            high_accuracy = high_accuracy,
        )
    return ta_dyn, ta_dyn_aug