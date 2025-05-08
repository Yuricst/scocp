"""CR3BP equations of motion with scipy"""

from numba import njit, float64
import numpy as np


@njit(float64[:,:](float64[:], float64), cache=True)
def gravity_gradient_cr3bp(rvec, mu):
    """Compute gravity gradient matrix for CR3BP in the rotating frame.

    Args:
        rvec (np.array): position vector of spacecraft
        mu (float): CR3BP parameter, i.e. scaled mass of secondary body

    Returns:
        np.array: 3-by-3 gravity gradient matrix
    """
    # unpack rvec
    x,y,z = rvec
    r1 = np.sqrt((x+mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x-(1-mu))**2 + y**2 + z**2)

    # define entries of the gravity gradient matrix
    r1_5 = r1**5
    r2_5 = r2**5
    G00 = 3*(1-mu)*(mu+x)**2/r1_5- (1-mu)/r1**3 + 3*mu*(x-1+mu)**2/r2_5 - mu/r2**3 + 1
    G01 = 3*(1-mu)*y*(mu+x)/r1_5+ 3*mu*y*(x-1+mu)/r2_5
    G02 = 3*(1-mu)*z*(mu+x)/r1_5+ 3*mu*z*(x-1+mu)/r2_5

    G10 = 3*(1-mu)*y*(mu+x)/r1_5+ 3*mu*y*(x-1+mu)/r2_5
    G11 = 3*(1-mu)*y**2/r1_5+ 3*mu*y**2/r2_5 - (1-mu)/r1**3 - mu/r2**3 + 1
    G12 = 3*(1-mu)*y*z/r1_5+ 3*mu*y*z/r2_5

    G20 = 3*(1-mu)*z*(mu+x)/r1_5+ 3*mu*z*(x-1+mu)/r2_5
    G21 = 3*(1-mu)*y*z/r1_5+ 3*mu*y*z/r2_5
    G22 = 3*(1-mu)*z**2/r1_5+ 3*mu*z**2/r2_5 - (1-mu)/r1**3 - mu/r2**3
    return np.array([[G00, G01, G02], [G10, G11, G12], [G20, G21, G22]])


@njit(float64[:](float64, float64[:], float64), cache=True)
def rhs_cr3bp(t, state, mu):
    """Equation of motion in CR3BP, formulated for scipy.integrate.solve=ivp(), compatible with njit

    Args:
        t (float): time
        state (np.array): 1D array of Cartesian state, length 6
        mu (float): CR3BP parameter

    Returns:
        (np.array): 1D array of derivative of Cartesian state
    """
    # unpack positions
    x = state[0]
    y = state[1]
    z = state[2]
    # unpack velocities
    vx = state[3]
    vy = state[4]
    vz = state[5]
    # compute radii to each primary
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2 + z ** 2)
    # setup vector for dX/dt
    deriv = np.zeros((6,))
    # position derivatives
    deriv[0] = vx
    deriv[1] = vy
    deriv[2] = vz
    # velocity derivatives
    deriv[3] = (
        2 * vy + x - ((1 - mu) / r1 ** 3) * (mu + x) + (mu / r2 ** 3) * (1 - mu - x)
    )
    deriv[4] = -2 * vx + y - ((1 - mu) / r1 ** 3) * y - (mu / r2 ** 3) * y
    deriv[5] = -((1 - mu) / r1 ** 3) * z - (mu / r2 ** 3) * z
    return deriv


def rhs_cr3bp_stm(t, state, mu):
    """Equation of motion in CR3BP in the rotating frame with STM.
    This function is written for `scipy.integrate.solve=ivp()` and is compatible with njit.
    """
    # state derivative
    deriv = np.zeros(42)
    deriv[0:6] = rhs_cr3bp(t, state[0:6], mu)
    
    # derivative of STM
    A = np.zeros((6,6))
    A[0:3,3:6] = np.eye(3)
    A[3,4] = 2
    A[4,3] = -2
    A[3:6,0:3] = gravity_gradient_cr3bp(state[0:3], mu)
    deriv[6:42] = np.dot(A, state[6:].reshape(6,6)).reshape(36,)
    return deriv


def control_rhs_cr3bp(t, state, mu, u):
    """Equation of motion in CR3BP with continuous control in the rotating frame"""
    # derivative of state
    B = np.concatenate((np.zeros((3,3)), np.eye(3)))
    deriv = rhs_cr3bp(t, state[0:6], mu) + B @ u[0:3]
    return deriv


def control_rhs_cr3bp_stm(t, state, mu, u):
    """Equation of motion in CR3BP with continuous control in the rotating frame with STM"""
    # derivative of state
    B = np.concatenate((np.zeros((3,3)), np.eye(3)))
    deriv = np.zeros(60)    # 6 + 6*6 + 6*3
    deriv[0:6] = rhs_cr3bp(t, state[0:6], mu) + B @ u[0:3]
    
    # derivative of STM
    Phi_A = state[6:42].reshape(6,6)
    A = np.zeros((6,6))
    A[0:3,3:6] = np.eye(3)
    A[3,4] = 2
    A[4,3] = -2
    A[3:6,0:3] = gravity_gradient_cr3bp(state[0:3], mu)
    deriv[6:42] = np.dot(A, Phi_A).reshape(36,)

    # derivative of control sensitivity
    Phi_B = state[42:60].reshape(6,3)
    deriv[42:60] = (np.dot(A, Phi_B) + B).reshape(18,)
    return deriv


def control_rhs_cr3bp_logmass(t, state, parameters, u):
    """Equation of motion in CR3BP with continuous control in the rotating frame"""
    # unpack parameters
    mu, cex = parameters
    # derivative of state
    B = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, -1/cex],
    ])
    deriv = np.zeros(7)
    deriv[0:6] = rhs_cr3bp(t, state[0:6], mu)
    deriv += B @ u
    return deriv


def control_rhs_cr3bp_logmass_stm(t, state, parameters, u):
    """Equation of motion in CR3BP with continuous control in the rotating frame with STM"""
    # derivative of state
    mu, cex = parameters
    B = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, -1/cex],
    ])
    deriv = np.zeros(84)    # 7 + 7*7 + 7*4
    deriv[0:6] = rhs_cr3bp(t, state[0:6], mu)
    deriv[0:7] += B @ u[:]
    
    # derivative of STM
    Phi_A = state[7:56].reshape(7,7)
    A = np.zeros((7,7))
    A[0:3,3:6] = np.eye(3)
    A[3,4] = 2
    A[4,3] = -2
    A[3:6,0:3] = gravity_gradient_cr3bp(state[0:3], mu)
    deriv[7:56] = np.dot(A, Phi_A).reshape(49,)

    # derivative of control sensitivity
    Phi_B = state[56:84].reshape(7,4)
    deriv[56:84] = (np.dot(A, Phi_B) + B).reshape(28,)
    return deriv


def control_rhs_cr3bp_freetf(tau, state, mu, u):
    """Equation of motion in CR3BP with free final time
    state = [x,y,z,vx,vy,vz,t]
    u = [ax,ay,az,s] where s is the dilation factor
    """
    t = state[6]
    B = np.concatenate((np.zeros((3,3)), np.eye(3)))
    deriv = np.zeros(7,)
    deriv[0:6] = u[3] * (rhs_cr3bp(t, state[0:6], mu) + B @ u[0:3])
    deriv[6]   = u[3]
    return deriv


def control_rhs_cr3bp_freetf_stm(tau, state, mu, u):
    """Equation of motion in CR3BP with free final time
    state = [x,y,z,vx,vy,vz,t]
    u = [ax,ay,az,s] where s is the dilation factor
    """
    # derivative of state
    deriv = np.zeros(84)    # 7 + 7*7 + 7*4
    deriv[0:7] = control_rhs_cr3bp_freetf(tau, state[0:7], mu, u)
    
    # derivative of STM
    Phi_A = state[7:56].reshape(7,7)
    A = np.zeros((7,7))
    A[0:3,3:6] = np.eye(3)
    A[3,4] = 2
    A[4,3] = -2
    A[3:6,0:3] = gravity_gradient_cr3bp(state[0:3], mu)
    deriv[7:56] = np.dot(u[3] * A, Phi_A).reshape(49,)          # note: multiply A by time-dilation factor

    # derivative of control sensitivity
    f_eval = control_rhs_cr3bp_freetf(tau, state[0:7], mu, u)[0:6]/u[3]
    B_accel = np.concatenate((np.concatenate((np.zeros((3,3)), u[3]*np.eye(3))), f_eval.reshape(-1,1)), axis=1)
    B = np.concatenate((B_accel, np.array([0.0, 0.0, 0.0, 1.0]).reshape(1,-1)))
    Phi_B = state[56:84].reshape(7,4)
    deriv[56:84] = (np.dot(u[3] * A, Phi_B) + B).reshape(28,)
    return deriv


def control_rhs_cr3bp_logmass_freetf(tau, state, parameters, u):
    """Equation of motion in CR3BP with continuous control in the rotating frame
    state = [x,y,z,vx,vy,vz,log(mass),t]
    u = [ax, ay, az, s, Gamma] where s is the dilation factor, Gamma is the control magnitude (at convergence)
    """
    # unpack parameters
    mu, cex = parameters
    # derivative of state
    deriv = np.zeros(8,)
    B = np.concatenate((np.zeros((3,3)), np.eye(3)))
    deriv[0:6] = u[3] * (rhs_cr3bp(state[6], state[0:6], mu) + B @ u[0:3])
    deriv[6]   = u[3] * u[4] * (-1/cex)     # control on log(mass), multiplied by time-dilation factor
    deriv[7]   = u[3]                       # control on t(tau)
    return deriv


def control_rhs_cr3bp_logmass_freetf_stm(tau, state, parameters, u):
    """Equation of motion in CR3BP with continuous control in the rotating frame with STM
    state = [x,y,z,vx,vy,vz,log(mass),t] + STM.flatten()
    u = [ax, ay, az, s, Gamma] where s is the dilation factor, Gamma is the control magnitude (at convergence)
    """
    # derivative of state
    mu, cex = parameters
    deriv = np.zeros(112)    # 8 + 8*8 + 8*5
    deriv[0:8] = control_rhs_cr3bp_logmass_freetf(tau, state[0:8], parameters, u)
    
    # derivative of STM
    Phi_A = state[8:72].reshape(8,8)
    A = np.zeros((8,8))
    A[0:3,3:6] = np.eye(3)
    A[3,4] = 2
    A[4,3] = -2
    A[3:6,0:3] = gravity_gradient_cr3bp(state[0:3], mu)
    deriv[8:72] = np.dot(u[3] * A, Phi_A).reshape(64,)

    # derivative of control sensitivity
    B = np.zeros((8,5))
    B[3:6,0:3] = u[3] * np.eye(3)
    B[0:8,3] = control_rhs_cr3bp_logmass_freetf(tau, state[0:8], parameters, u)/u[3]
    B[6,4] = -u[3]/cex
    Phi_B = state[72:112].reshape(8,5)
    deriv[72:112] = (np.dot(u[3] * A, Phi_B) + B).reshape(40,)
    return deriv