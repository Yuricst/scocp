"""Miscellaneous functions"""

from collections.abc import Callable
import cvxpy as cp
import numpy as np
from numba import njit

def zoh_control(times, us, t):
    """Zero-order hold control"""
    for i in range(len(times)-1):
        # Binary search to find interval containing t
        idx = np.searchsorted(times, t, side='right') - 1
        if idx >= 0 and idx < len(times)-1:
            return us[idx]
    return us[-1]  # Return last control if t > times[-1]


def zoh_controls(times, us, t_eval):
    """Zero-order hold control"""
    _,nu = us.shape
    us_zoh = np.zeros((len(t_eval),nu))
    for i,t in enumerate(t_eval):
        us_zoh[i,:] = zoh_control(times, us, t)
    return us_zoh


def get_augmented_lagrangian_penalty(weight, xi_dyn, lmb_dyn, xi=None, lmb_eq=None, zeta=None, lmb_ineq=None):
    """Evaluate augmented Lagrangian penalty function
    
    Args:
        weight (float): weight of the penalty function
        xi_dyn (cp.Variable): slack variable for dynamics
        lmb_dyn (cp.Parameter): multiplier for dynamics
        xi (cp.Variable, optional): slack variable for equality constraints
        lmb_eq (cp.Parameter, optional): multiplier for equality constraints
        zeta (cp.Variable, optional): slack variable for inequality constraints
        lmb_ineq (cp.Parameter, optional): multiplier for inequality constraints
    
    Returns:
        (cp.Expression): augmented Lagrangian penalty function
    """
    assert xi_dyn.shape == lmb_dyn.shape, f"xi_dyn.shape = {xi_dyn.shape} must match lmb_dyn.shape = {lmb_dyn.shape}"
    penalty = weight/2 * cp.sum_squares(xi_dyn)
    for i in range(lmb_dyn.shape[0]):
        penalty += lmb_dyn[i,:] @ xi_dyn[i,:]
    if xi is not None:
        penalty += weight/2 * cp.sum_squares(xi)
        for i in range(len(lmb_eq)):
            penalty += lmb_eq[i] * xi[i]
    if zeta is not None:
        penalty += weight/2 * cp.sum_squares(zeta)
        for i in range(len(lmb_ineq)):
            penalty += lmb_ineq[i] * zeta[i]
    return penalty


class MovingTarget:
    """Define moving target for rendezvous problem
    
    We assume the target state equality constraint is of the form

    g(r_N,v_N,t_N) = [ r_N - r_ref(t_N)
                       v_N - v_ref(t_N) ]

    where r_ref(t_N) and v_ref(t_N) are the position and velocity of the target at time t_N.

    Args:
        eval_state (function): function to evaluate target state
        eval_state_derivative (function): function to evaluate derivative of target state
    """
    def __init__(self, eval_state: Callable, eval_state_derivative: Callable):
        self.eval_state = eval_state
        self.eval_state_derivative = eval_state_derivative
        return
    
    def target_state(self, t: float) -> np.ndarray:
        """Get target state at time t"""
        return self.eval_state(t)
    
    def target_state_derivative(self, t: float) -> np.ndarray:
        """Get target state derivative w.r.t. time"""
        return self.eval_state_derivative(t)
    

# --------------------- Function to convert between Keplerian elements and Cartesian state --------------------- #
# See: Curtis pg.209
@njit
def _rotmat_ax1(phi):
    return np.array([ [1.0, 0.0,          0.0], 
                      [0.0,  np.cos(phi), np.sin(phi)], 
                      [0.0, -np.sin(phi), np.cos(phi)] ])

@njit
def _rotmat_ax2(phi):
    return np.array([ [np.cos(phi), 0.0, -np.sin(phi)], 
                      [0.0,         1.0,  0.0        ], 
                      [np.sin(phi), 0.0,  np.cos(phi)] ])

@njit
def _rotmat_ax3(phi):
    return np.array([ [ np.cos(phi), np.sin(phi), 0.0], 
                      [-np.sin(phi), np.cos(phi), 0.0], 
                      [0.0,          0.0,         1.0] ])


@njit
def kep2rv(elements, mu):
    """Get Cartesian state-vector to Keplerian elements
	
    Args:
        ecc (float): eccentricity, scalar
        sma (float): semi-major axis
        inc (float): inclination
        raan (float): right ascension of ascending node 
        aop (float): argument of periapsis
        ta (float): true anomaly
        mu (float): gravitational parameter

    Returns:
        (np.array): state-vector in Cartesian frame
    """
    # unpack elements
    sma, ecc, inc, raan, aop, ta = elements
    # angular momentum vector
    h   = np.sqrt(sma*mu*(1-ecc**2))
    # perifocal vector
    x   = (h**2/mu) * (1/(1 + ecc*np.cos(ta))) * np.cos(ta)
    y   = (h**2/mu) * (1/(1 + ecc*np.cos(ta))) * np.sin(ta)
    vx  = (mu/h)*(-np.sin(ta))
    vy  = (mu/h)*(ecc + np.cos(ta))
    rpf = np.array([ x, y, 0.0 ])
    vpf = np.array([ vx, vy, 0.0 ])

    # rotate by aop
    r_rot1 = np.dot(_rotmat_ax3(-aop), rpf)
    v_rot1 = np.dot(_rotmat_ax3(-aop), vpf)
    # rotate by inclination
    r_rot2 = np.dot(_rotmat_ax1(-inc), r_rot1)
    v_rot2 = np.dot(_rotmat_ax1(-inc), v_rot1)
    # rotate by raan
    r_rot3 = np.dot(_rotmat_ax3(-raan), r_rot2)
    v_rot3 = np.dot(_rotmat_ax3(-raan), v_rot2)

    # save inertial state
    state_inr = np.concatenate((r_rot3, v_rot3), axis=0) 
    return state_inr


@njit
def get_inclination(state):
    """Function computes inclination in radians from a two-body state vector, in inertially frozen frame
    
    Args:
        state (np.array): array of cartesian state in inertially frozen frame

    Returns:
        (float): inclination in radians
    """

    # decompose state to position and velocity vector
    r = np.array([state[0], state[1], state[2]])
    v = np.array([state[3], state[4], state[5]])
    # angular momentum
    h = np.cross(r,v)
    # inclination
    inc = np.arccos(h[2] / np.linalg.norm(h))
    return inc


@njit
def get_raan(state):
    """Function computes RAAN in radians from a two-body state vector, in inertially frozen frame
    
    Args:
        state (np.array): array of cartesian state in inertially frozen frame

    Returns:
        (float): RAAN in radians
    """
    
    # decompose state to position and velocity vector
    r = np.array([state[0], state[1], state[2]])
    v = np.array([state[3], state[4], state[5]])
    # angular momentum
    h = np.cross(r,v)
    # normal direction of xy-plane
    zdir = np.array([0, 0, 1])
    ndir = np.cross(zdir, h)
    # compute RAAN
    raan = np.arctan2(ndir[1], ndir[0])
    return raan
    
    
@njit
def get_eccentricity(state, mu):
    """Function computes eccentricity vector from a two-body state vector, in inertially frozen frame
    
    Args:
        state (np.array): array of cartesian state in inertially frozen frame
        mu (float): two-body mass parameter

    Returns:
        (np.arr): eccentricity vector
    """
    
    # decompose state to position and velocity vector
    r = np.array([state[0], state[1], state[2]])
    v = np.array([state[3], state[4], state[5]])
    # angular momentum
    h = np.cross(r,v)
    # normal direction of xy-plane
    #zdir = np.array([0, 0, 1])
    #ndir = np.cross(zdir, h)
    # compute eccentricity vector
    ecc = (1/mu) * np.cross(v,h) - r/np.linalg.norm(r)
    return ecc


@njit
def get_omega(state, mu):
    """Function computes argument of periapsis in radians from a two-body state vector, in inertially frozen frame.
    If eccentricity is 0, omega = RAAN
    
    Args:
        state (np.array): array of cartesian state in inertially frozen frame
        mu (float): two-body mass parameter

    Returns:
        (float): argument of periapsis in radians
    """
    
    # decompose state to position and velocity vector
    r = np.array([state[0], state[1], state[2]])
    v = np.array([state[3], state[4], state[5]])
    # angular momentum
    h = np.cross(r,v)
    # normal direction of xy-plane
    zdir = np.array([0, 0, 1])
    ndir = np.cross(zdir, h)
    # compute eccentricity vector
    ecc = (1/mu) * np.cross(v,h) - r/np.linalg.norm(r)
    if np.linalg.norm(ndir)*np.linalg.norm(ecc)!=0:
        # compute argument of periapsis
        omega = np.arccos( np.dot(ndir,ecc) / (np.linalg.norm(ndir)*np.linalg.norm(ecc)) )
        if ecc[2] < 0:
            omega = 2*np.pi - omega
    else:
        omega = get_raan(state)
    return omega


@njit
def get_trueanomaly(state, mu):
    """Function computes argument of periapsis in radians from a two-body state vector, in inertially frozen frame
    
    Args:
        state (np.array): array of cartesian state in inertially frozen frame
        mu (float): two-body mass parameter

    Returns:
        (float): true anomaly in radians
    """
    # decompose state to position and velocity vector
    r = np.array([state[0], state[1], state[2]])
    v = np.array([state[3], state[4], state[5]])
    # angular momentum
    h = np.linalg.norm( np.cross(r,v) )
    # radial velocity
    vr = np.dot(v,r)/np.linalg.norm(r)
    theta = np.arctan2(h*vr, h**2/np.linalg.norm(r) - mu)
    return theta


@njit
def get_semiMajorAxis(state, mu):
    """Function computes semi major axis of keplrian orbit
    
    Args:
        state (np.array): array of cartesian state in inertially frozen frame
        mu (float): two-body mass parameter

    Returns:
        (float): semi-major axis
    """
    # decompose state to position and velocity vector
    r = np.array([state[0], state[1], state[2]])
    v = np.array([state[3], state[4], state[5]])
    # angular momentum
    h = np.linalg.norm( np.cross(r,v) )
    # eccentricity
    e = np.linalg.norm( get_eccentricity(state, mu) )
    # semi-major axis
    a = h**2 / (mu*(1 - e**2))
    return a


@njit
def rv2kep(state, mu, fictious_vz=1.e-15):
    """Get Keplerian elements from Cartesian state-vector and gravitational parameter

    Args:
        state (np.array): state-vector
        mu (float): gravitational parameter
        fictious_vz (float): fictious vz to inject if state is purely planar

    Returns:
        (tuple): eccentricity, semi-major axis, inclination, raan, argument of periapsis, true anomaly
    """
    # if state is planar, inject fictious out-of-plane component
    if (state[2] == 0.) and (state[5] == 0.):
        state[5] = fictious_vz
    ecc  = get_eccentricity(state, mu)
    sma  = get_semiMajorAxis(state, mu)
    inc  = get_inclination(state)
    raan = get_raan(state)
    aop  = get_omega(state, mu)
    ta   = get_trueanomaly(state, mu)
    return [sma, ecc, inc, raan, aop, ta]