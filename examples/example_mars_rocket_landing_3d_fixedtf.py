"""Fixed-time 3-DoF Mars rocket landing example."""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import scocp

PLOT_PATH = Path(__file__).resolve().parent / "plots" / "mars_rocket_landing_3d_fixedtf.png"


def run_example(plot_path: Path | None = None):
    """Solve, simulate, and optionally plot the default Mars landing benchmark."""

    problem = scocp.FixedTimeMars3DoFRocketLanding()
    solution = problem.solve()
    simulation = problem.simulate(solution)

    if plot_path is not None:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        problem.plot_summary(solution, sim=simulation, path=str(plot_path))
    return solution


if __name__ == "__main__":
    run_example(PLOT_PATH)
    plt.show()
