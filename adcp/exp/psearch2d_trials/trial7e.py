import numpy as np

import adcp.exp
from adcp.exp.psearch2d import RigorousParameterSearch2D


def __main__():
    experiment = RigorousParameterSearch2D(
        variant="TTW Modeling, multi-first GPS point, mutli-simulation",
        sims=20,
        vehicle_vel="ttw",
        gps_points="multi-first",
        rho_vs=np.logspace(-10, -1, 11),
        rho_cs=np.logspace(-10, -1, 11),
    )
    adcp.exp.run(experiment)


if __name__ == "__main__":
    __main__()
