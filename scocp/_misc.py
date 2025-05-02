"""Miscellaneous functions"""

import numpy as np

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