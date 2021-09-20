"""Just used to test out plotting functions."""
from pathlib import Path

import numpy as np

import adcp.exp
from adcp.exp.psearch2d import ParameterSearch2D

prob_params = {
    "t_scale": 1,
    "current_order": 3,
    "vehicle_order": 3,
    "rho_vs": np.logspace(-5, -6, 3),
    "rho_cs": np.logspace(-5, -6, 3),
}
sim_params = {
    "gps_points": "multi-first",
}


def __main__():
    experiment = ParameterSearch2D
    adcp.exp.run(
        experiment,
        prob_params=prob_params,
        sim_params=sim_params,
        trials_folder=Path(__file__).absolute().parent,
    )


if __name__ == "__main__":
    __main__()
