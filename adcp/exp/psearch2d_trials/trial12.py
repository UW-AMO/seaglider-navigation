from pathlib import Path
import warnings

import numpy as np

import adcp.exp
from adcp.exp.psearch2d import ParameterSearch2D

warnings.warn(
    "Import trial 12a definitions from psearch2d.py, rather than from this"
    " module"
)

prob_params = {
    "t_scale": 1,
    "current_order": 3,
    "vehicle_order": 3,
    "rho_vs": np.logspace(-10, 0, 11),
    "rho_cs": np.logspace(-10, 0, 11),
}
sim_params = {}


def __main__(debug=False):
    experiment = ParameterSearch2D
    adcp.exp.run(
        experiment,
        prob_params=prob_params,
        sim_params=sim_params,
        trials_folder=Path(__file__).absolute().parent,
        debug=debug,
    )


if __name__ == "__main__":
    __main__()
