"""Compare against pykep for direct transfers"""


import pykep as pk
import pygmo as pg
import numpy as np
from matplotlib import pyplot as plt
from pykep.examples import add_gradient, algo_factory

def pykep_direct_pl2pl():
    # 1 - Algorithm
    uda = pg.ipopt() # pylint: disable=no-member
    uda.set_integer_option("print_level", 5)
    uda.set_integer_option("acceptable_iter", 4)
    uda.set_integer_option("max_iter", 500)

    uda.set_numeric_option("tol", 1e-8)
    uda.set_numeric_option("dual_inf_tol", 1e-8)
    uda.set_numeric_option("constr_viol_tol", 1e-8)
    uda.set_numeric_option("compl_inf_tol", 1e-8)

    uda.set_numeric_option("acceptable_tol", 1e-3)
    uda.set_numeric_option("acceptable_dual_inf_tol", 1e-2)
    uda.set_numeric_option("acceptable_constr_viol_tol", 1e-6)
    uda.set_numeric_option("acceptable_compl_inf_tol", 1e-6)

    algo = pg.algorithm(uda)

    # import pygmo_plugins_nonfree as ppnf
    # uda = ppnf.snopt7(True,
    #                  "/Users/yuri/libsnopt7_cpp/libsnopt7_cpp.dylib",
    #                  7)
    # uda.set_integer_option("Major iterations limit", 2000)
    # uda.set_integer_option("Iterations limit", 200000)
    # uda.set_numeric_option("Major optimality tolerance", 1e-2)
    # uda.set_numeric_option("Major feasibility tolerance", 1e-9)
    # algo = pg.algorithm(uda)
    
    # 2 - Problem
    nseg = 40
    udp = add_gradient(
        pk.trajopt.direct_pl2pl(
            p0="earth",
            pf="mars",
            mass=800.0,
            thrust=0.2,
            isp=3000,
            vinf_arr=1e-6,
            vinf_dep=1e-6,
            hf=False,
            nseg=nseg,
            t0=[1100, 1400],
            tof=[100,800], #[200, 750]
        ),
        with_grad=True
    )

    prob = pg.problem(udp)
    prob.c_tol = [1e-5] * prob.get_nc()

    # 3 - Population
    pop = pg.population(prob, 1)

    # 4 - Solve the problem (evolve)
    pop = algo.evolve(pop)

    # 5 - Inspect the solution
    if prob.feasibility_x(pop.champion_x):
        print("Optimal Found!!")
    else:
        print("No solution found, try again :)")

    udp.udp_inner.pretty(pop.champion_x)

    fig = plt.figure(figsize=(12,6))
    ax_traj = fig.add_subplot(1,2,1,projection='3d')
    axis = udp.udp_inner.plot_traj(pop.champion_x, axes=ax_traj)
    plt.title("The trajectory in the heliocentric frame")

    ax_ctrl = fig.add_subplot(1,2,2)
    axis = udp.udp_inner.plot_control(pop.champion_x, axes=ax_ctrl)
    plt.title("The control profile (throttle)")
    return


def scocp_pl2pl():
    pl0 = pk.planet.jpl_lp('earth')
    plf = pk.planet.jpl_lp('mars')
    mass = 1000
    thrust = 0.8
    isp = 3000
    nseg = 40

    print(pl0)
    


if __name__ == "__main__":
    pykep_direct_pl2pl()
    plt.show()