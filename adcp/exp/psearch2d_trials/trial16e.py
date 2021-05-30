import numpy as np

import adcp.exp
from adcp.exp.psearch2d import RigorousParameterSearch2D


def __main__():
    experiment = RigorousParameterSearch2D(
        variant=(
            "No Time Scaling, vehicle_order=3, current_order=3, "
            "vehicle_vel=TTW, "
            "multi-first GPS only, multi-simulation"
        ),
        sims=20,
        t_scale=1,
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
