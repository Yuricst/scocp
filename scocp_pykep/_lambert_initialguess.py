"""Construct initial guess from Lambert problem"""

import numpy as np
import pykep as pk

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scocp import (
    keplerder_nostm,
)

def get_lambert_gridsearch(mu, pl0, plf, t0, tf, dt):
    """Perform grid search with Lambert problem"""
    times = np.linspace(t0, tf, int(np.ceil((tf-t0)/dt)))
    N = len(times)-1
    costs = 1e15 * np.ones((N,N))
    l_sols = {}
    times_dict = {}
    for i, t0 in enumerate(times[:-1]):
        for j, tf in enumerate(times[i+1:]):
            # query planet positions
            rv0 = pl0.eph(t0)
            rvf = plf.eph(tf)

            # solve Lambert problem & compute cost
            if isinstance(pk.__version__, dict):
                lp = pk.lambert_problem(r1 = rv0[0], r2 = rvf[0], tof = (tf-t0)*86400.0)
                v10 = np.array(lp.get_v1()[0])
                v1f = np.array(lp.get_v2()[0])
            else:
                lp = pk.lambert_problem(rv0[0], rvf[0], (tf-t0)*86400.0, mu)
                v10 = np.array(lp.v0[0])
                v1f = np.array(lp.v1[0])
            DV_cost = np.linalg.norm(v10 - np.array(rv0[1])) + np.linalg.norm(v1f - np.array(rvf[1]))
            costs[i,j] = DV_cost
            l_sols[f"{i}_{j}"] = lp
            times_dict[f"{i}_{j}"] = (t0,tf)

    # departure & arrival times
    i_min,j_min = np.unravel_index(costs.argmin(), costs.shape)
    cost_min = costs[i_min,j_min]
    return cost_min, times_dict[f"{i_min}_{j_min}"][0], times_dict[f"{i_min}_{j_min}"][1], l_sols[f"{i_min}_{j_min}"]


def lambert_sol_to_guess(problem, tdep, tarr, pl0, v0):
    _x0 = np.concatenate((
        np.array(pl0.eph(tdep)[0])/problem.r_scaling, v0/problem.v_scaling
    ))
    xbar = np.zeros((problem.N,8))
    t_eval = np.linspace((tdep - problem.t0_min)/problem.TU2DAY, (tarr - problem.t0_min)/problem.TU2DAY, problem.N)
    for i, t in enumerate(t_eval):
        xbar[i,0:6] = keplerder_nostm(problem.mu, _x0, 0.0, t - t_eval[0])
    xbar[:,6] = np.linspace(1.0, 0.5, problem.N)   # mass
    xbar[:,7] = t_eval[:]

    sbar_initial = (tarr - tdep) * np.ones((problem.N-1,1))
    ubar = np.concatenate((np.zeros((problem.N-1,3)), sbar_initial), axis=1)
    vbar = np.sum(ubar[:,0:3], axis=1).reshape(-1,1)
    return xbar, ubar, vbar