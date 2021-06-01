from pathlib import Path

import numpy as np

import adcp.exp
from adcp.exp.psearch2d import RigorousParameterSearch2D


def __main__():
    prob_params = {
        "t_scale": 1,
        "vehicle_order": 3,
        "current_order": 3,
        "rho_vs": np.logspace(-10, 0, 11),
        "rho_cs": np.logspace(-10, 0, 11),
    }
    sim_params = {
        "sims": 20,
    }
    experiment = RigorousParameterSearch2D
    adcp.exp.run(
        experiment,
        prob_params=prob_params,
        sim_params=sim_params,
        trials_folder=Path(__file__).absolute().parent,
        debug=True,
    )


if __name__ == "__main__":
    __main__()
