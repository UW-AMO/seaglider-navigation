import numpy as np

import adcp.exp
from adcp.exp.psearch2d import ParameterSearch2D


def __main__():
    experiment = ParameterSearch2D(
        variant=(
            "TTW Modeling, current order=3, vehicle order=3, "
            "multi-first GPS only"
        ),
        vehicle_vel="ttw",
        current_order=3,
        vehicle_order=3,
        gps_points="multi-first",
        rho_vs=np.logspace(-10, -1, 11),
        rho_cs=np.logspace(-10, -1, 11),
    )
    adcp.exp.run(experiment)


if __name__ == "__main__":
    __main__()
