import importlib
from collections import namedtuple

import numpy as np

from adcp.exp.psearch2d import ParameterSearch2D, RigorousParameterSearch2D

this_module = importlib.import_module(__name__)


def __getattr__(name):
    import_name = "." + name
    try:
        imported = importlib.import_module(
            import_name, this_module.__spec__.parent
        )
    except ModuleNotFoundError:
        raise AttributeError(f"'{__name__}' has no attribute '{name}'")
    return imported


Trial = namedtuple("Trial", ["ex", "prob_params", "sim_params"])

trial18a = Trial(
    ParameterSearch2D,
    prob_params={
        "t_scale": 1,
        "vehicle_vel": "otg-cov",
        "rho_vs": np.logspace(-10, 0, 11),
        "rho_cs": np.logspace(-10, 0, 11),
    },
    sim_params={},
)
trial18c = Trial(
    ParameterSearch2D,
    prob_params={
        "t_scale": 1,
        "vehicle_vel": "otg-cov",
        "rho_vs": np.logspace(-10, 0, 11),
        "rho_cs": np.logspace(-10, 0, 11),
    },
    sim_params={"gps_points": "multi-first"},
)
trial18d = Trial(
    RigorousParameterSearch2D,
    prob_params={
        "t_scale": 1,
        "vehicle_vel": "otg-cov",
        "rho_vs": np.logspace(-10, 0, 11),
        "rho_cs": np.logspace(-10, 0, 11),
    },
    sim_params={
        "sims": 20,
    },
)
trial18e = Trial(
    RigorousParameterSearch2D,
    prob_params={
        "t_scale": 1,
        "vehicle_vel": "otg-cov",
        "rho_vs": np.logspace(-10, 0, 11),
        "rho_cs": np.logspace(-10, 0, 11),
    },
    sim_params={"sims": 20, "gps_points": "multi-first"},
)
trial19a = Trial(
    ParameterSearch2D,
    prob_params={
        "t_scale": 1,
        "vehicle_vel": "otg-cov",
        "current_order": 3,
        "rho_vs": np.logspace(-10, 0, 11),
        "rho_cs": np.logspace(-10, 0, 11),
    },
    sim_params={},
)
trial19c = Trial(
    ParameterSearch2D,
    prob_params={
        "t_scale": 1,
        "vehicle_vel": "otg-cov",
        "current_order": 3,
        "rho_vs": np.logspace(-10, 0, 11),
        "rho_cs": np.logspace(-10, 0, 11),
    },
    sim_params={"gps_points": "multi-first"},
)
trial19d = Trial(
    RigorousParameterSearch2D,
    prob_params={
        "t_scale": 1,
        "vehicle_vel": "otg-cov",
        "current_order": 3,
        "rho_vs": np.logspace(-10, 0, 11),
        "rho_cs": np.logspace(-10, 0, 11),
    },
    sim_params={
        "sims": 20,
    },
)
trial19e = Trial(
    RigorousParameterSearch2D,
    prob_params={
        "t_scale": 1,
        "vehicle_vel": "otg-cov",
        "current_order": 3,
        "rho_vs": np.logspace(-10, 0, 11),
        "rho_cs": np.logspace(-10, 0, 11),
    },
    sim_params={"sims": 20, "gps_points": "multi-first"},
)
trial20a = Trial(
    ParameterSearch2D,
    prob_params={
        "t_scale": 1,
        "vehicle_vel": "otg-cov",
        "vehicle_order": 3,
        "current_order": 3,
        "rho_vs": np.logspace(-10, 0, 11),
        "rho_cs": np.logspace(-10, 0, 11),
    },
    sim_params={},
)
trial20c = Trial(
    ParameterSearch2D,
    prob_params={
        "t_scale": 1,
        "vehicle_vel": "otg-cov",
        "vehicle_order": 3,
        "current_order": 3,
        "rho_vs": np.logspace(-10, 0, 11),
        "rho_cs": np.logspace(-10, 0, 11),
    },
    sim_params={"gps_points": "multi-first"},
)
trial20d = Trial(
    RigorousParameterSearch2D,
    prob_params={
        "t_scale": 1,
        "vehicle_vel": "otg-cov",
        "vehicle_order": 3,
        "current_order": 3,
        "rho_vs": np.logspace(-10, 0, 11),
        "rho_cs": np.logspace(-10, 0, 11),
    },
    sim_params={
        "sims": 20,
    },
)
trial20e = Trial(
    RigorousParameterSearch2D,
    prob_params={
        "t_scale": 1,
        "vehicle_vel": "otg-cov",
        "vehicle_order": 3,
        "current_order": 3,
        "rho_vs": np.logspace(-10, 0, 11),
        "rho_cs": np.logspace(-10, 0, 11),
    },
    sim_params={"sims": 20, "gps_points": "multi-first"},
)
