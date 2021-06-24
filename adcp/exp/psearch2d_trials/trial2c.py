from pathlib import Path

import numpy as np

import adcp.exp
from adcp.exp.psearch2d import ParameterSearch2D

prob_params = {
    "t_scale": 1,
    "rho_vs": np.logspace(-10, 0, 11),
    "rho_cs": np.logspace(-10, 0, 11),
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
