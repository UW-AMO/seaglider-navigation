from pathlib import Path
import warnings

import numpy as np

import adcp.exp
from adcp.exp.psearch2d import RigorousParameterSearch2D

warnings.warn(
    "Import trial 12a definitions from psearch2d.py, rather than from this"
    " module"
)


def __main__():
    experiment = RigorousParameterSearch2D
    prob_params = {
        "t_scale": 1,
        "current_order": 3,
        "vehicle_order": 3,
        "rho_vs": np.logspace(-10, 0, 11),
        "rho_cs": np.logspace(-10, 0, 11),
    }
    sim_params = {
        "sims": 20,
        "gps_points": "multi-first",
    }
    adcp.exp.run(
        experiment,
        prob_params=prob_params,
        sim_params=sim_params,
        trials_folder=Path(__file__).absolute().parent,
    )


if __name__ == "__main__":
    __main__()
