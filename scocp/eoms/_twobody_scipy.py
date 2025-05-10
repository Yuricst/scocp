"""Two-body dynamics"""

from numba import njit, float64
import numpy as np


@njit(float64[:,:](float64[:], float64), cache=True)
def gravity_gradient_twobody(rvec, mu):
    """Compute gravity gradient matrix for R2BP in the rotating frame.

    Args:
        mu (float): R2BP parameter
        rvec (np.array): position vector of spacecraft

    Returns:
        np.array: 3-by-3 gravity gradient matrix
    """
    # unpack rvec
    x,y,z = rvec
    rnorm = np.sqrt(x**2 + y**2 + z**2)
    G = mu/rnorm**5 * (3*np.outer(rvec, rvec) - np.eye(3)*rnorm**2)
    return G


@njit(float64[:](float64, float64[:], float64), cache=True)
def rhs_twobody(t, state, mu):
    """Equation of motion in R2BP, formulated for scipy.integrate.solve=ivp(), compatible with njit

    Args:
        t (float): time
        state (np.array): 1D array of Cartesian state, length 6
        mu (float): R2BP parameter

    Returns:
        (np.array): 1D array of derivative of Cartesian state
    """
    deriv = np.zeros((6,))
    deriv[0:3] = state[3:6]
    deriv[3:6] = -mu * state[0:3] / np.linalg.norm(state[0:3])**3
    return deriv


@njit(float64[:](float64, float64[:], float64), cache=True)
def rhs_twobody_stm(t, state, mu):
    """Equation of motion in R2BP in the rotating frame with STM.
    This function is written for `scipy.integrate.solve=ivp()` and is compatible with njit.
    """
    # state derivative
    deriv = np.zeros(12)
    deriv[0:6] = rhs_twobody(t, state[0:6], mu)
    
    # derivative of STM
    A = np.zeros((6,6))
    A[0:3,3:6] = np.eye(3)
    A[3:6,0:3] = gravity_gradient_twobody(state[0:3], mu)
    # Create a contiguous copy of the STM part before reshaping
    stm = np.copy(state[6:])
    deriv[6:] = np.dot(A, stm.reshape(6,6)).reshape(36,)
    return deriv


def control_rhs_twobody_logmass_freetf(tau, state, parameters, u):
    """Equation of motion in R2BP with continuous control in the rotating frame
    state = [x,y,z,vx,vy,vz,log(mass),t]
    u = [ax, ay, az, s, Gamma] where s is the dilation factor, Gamma is the control magnitude (at convergence)
    """
    # unpack parameters
    mu, cex = parameters
    # derivative of state
    deriv = np.zeros(8,)
    B = np.concatenate((np.zeros((3,3)), np.eye(3)))
    deriv[0:6] = u[3] * (rhs_twobody(state[6], state[0:6], mu) + B @ u[0:3])
    deriv[6]   = u[3] * u[4] * (-1/cex)     # control on log(mass), multiplied by time-dilation factor
    deriv[7]   = u[3]                       # control on t(tau)
    return deriv


def control_rhs_twobody_logmass_freetf_stm(tau, state, parameters, u):
    """Equation of motion in R2BP with continuous control in the rotating frame with STM
    state = [x,y,z,vx,vy,vz,log(mass),t] + STM.flatten()
    u = [ax, ay, az, s, Gamma] where s is the dilation factor, Gamma is the control magnitude (at convergence)
    """
    # derivative of state
    mu, cex = parameters
    deriv = np.zeros(112)    # 8 + 8*8 + 8*5
    deriv[0:8] = control_rhs_twobody_logmass_freetf(tau, state[0:8], parameters, u)
    
    # derivative of STM
    Phi_A = state[8:72].reshape(8,8)
    A = np.zeros((8,8))
    A[0:3,3:6] = np.eye(3)
    A[3:6,0:3] = gravity_gradient_twobody(state[0:3], mu)
    deriv[8:72] = np.dot(u[3] * A, Phi_A).reshape(64,)

    # derivative of control sensitivity
    B = np.zeros((8,5))
    B[3:6,0:3] = u[3] * np.eye(3)
    B[0:8,3] = control_rhs_twobody_logmass_freetf(tau, state[0:8], parameters, u)/u[3]
    B[6,4] = -u[3]/cex
    Phi_B = state[72:112].reshape(8,5)
    deriv[72:112] = (np.dot(u[3] * A, Phi_B) + B).reshape(40,)
    return deriv


def control_rhs_twobody_mass_freetf(tau, state, parameters, u):
    """Equation of motion in R2BP with continuous control in the rotating frame
    
    - state = [x,y,z,vx,vy,vz,mass,t]
    - u = [ux, uy, uz, s, Gamma] where s is the dilation factor, Gamma is the control throttle magnitude (at convergence)
    - parameters = [mu, c1, c2]
    """
    # unpack parameters
    mu, c1, c2 = parameters
    # derivative of state
    deriv = np.zeros(8,)
    B = np.concatenate((np.zeros((3,3)), np.eye(3)))
    deriv[0:6] = u[3] * (rhs_twobody(state[6], state[0:6], mu) + B @ u[0:3] * (c1/state[6]))
    deriv[6]   = u[3] * u[4] * (-c2)     # control on log(mass), multiplied by time-dilation factor
    deriv[7]   = u[3]                       # control on t(tau)
    return deriv


def control_rhs_twobody_mass_freetf_stm(tau, state, parameters, u):
    """Equation of motion in R2BP with continuous control in the rotating frame with STM

    - state = [x,y,z,vx,vy,vz,mass,t] + STM.flatten()
    - u = [ux, uy, uz, s, Gamma] where s is the dilation factor, Gamma is the control throttle magnitude (at convergence)
    - parameters = [mu, c1, c2]
    """
    # derivative of state
    mu, c1, c2 = parameters
    deriv = np.zeros(112)    # 8 + 8*8 + 8*5
    deriv[0:8] = control_rhs_twobody_mass_freetf(tau, state[0:8], parameters, u)
    
    # derivative of STM
    Phi_A = state[8:72].reshape(8,8)
    A = np.zeros((8,8))
    A[0:3,3:6] = np.eye(3)
    A[3:6,0:3] = gravity_gradient_twobody(state[0:3], mu)
    A[3:6,6]   = -(c1/state[6]**2) * np.array(u[0:3])
    deriv[8:72] = np.dot(u[3] * A, Phi_A).reshape(64,)

    # derivative of control sensitivity
    B = np.zeros((8,5))
    B[3:6,0:3] = u[3] * c1/state[6] * np.eye(3)
    B[0:8,3] = control_rhs_twobody_mass_freetf(tau, state[0:8], parameters, u)/u[3]
    B[6,4] = -u[3] * c2
    Phi_B = state[72:112].reshape(8,5)
    deriv[72:112] = (np.dot(u[3] * A, Phi_B) + B).reshape(40,)
    return deriv