import numpy as np

import adcp.exp
from adcp.exp.psearch2d import ParameterSearch2D


def __main__():
    experiment = ParameterSearch2D(
        variant="TTW Modeling",
        vehicle_vel="ttw",
        rho_vs=np.logspace(-10, -1, 11),
        rho_cs=np.logspace(-10, -1, 11),
    )
    adcp.exp.run(experiment)


if __name__ == "__main__":
    __main__()