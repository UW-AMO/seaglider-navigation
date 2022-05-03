import importlib
from collections import namedtuple

import numpy as np

from adcp.exp.psearch2d import ParameterSearch2D, RigorousParameterSearch2D

this_module = importlib.import_module(__name__)


Trial = namedtuple("Trial", ["solve_params"])
std_search = np.logspace(-10, 0, 11)
trial1 = Trial({"rho_vs": std_search, "rho_cs": std_search})
trial2 = Trial({**trial1.solve_params, "t_scale": 1})
trial3 = Trial({**trial1.solve_params, "conditioner": None})
trial4 = Trial({**trial2.solve_params, "conditioner": None})
trial5 = Trial({**trial1.solve_params, "current_order": 3})
trial6 = Trial({**trial1.solve_params, "vehicle_order": 3})
trial7 = Trial({**trial1.solve_params, "vehicle_vel": "ttw"})
trial8 = Trial({**trial7.solve_params, "current_order": 3})
trial9 = Trial({**trial7.solve_params, "vehicle_order": 3})
trial10 = Trial({**trial8.solve_params, "vehicle_order": 3})
trial11 = Trial({**trial5.solve_params, "vehicle_order": 3})
trial12 = Trial({**trial11.solve_params, "t_scale": 1})
trial13 = Trial({**trial5.solve_params, "t_scale": 1})
trial14 = Trial({**trial6.solve_params, "t_scale": 1})
trial15 = Trial({**trial12.solve_params, "conditioner": None})
trial16 = Trial({**trial10.solve_params, "t_scale": 1})
trial17 = Trial({**trial2.solve_params, "vehicle_vel": "ttw"})
trial18 = Trial({**trial2.solve_params, "vehicle_vel": "otg-cov"})
trial19 = Trial({**trial18.solve_params, "current_order": 3})
trial20 = Trial({**trial19.solve_params, "vehicle_order": 3})
trial21 = Trial({**trial12.solve_params, "rho_cs": [1e-5], "rho_vs": [1e-5]})
trial22 = Trial(
    {
        **trial12.solve_params,
        "rho_cs": np.logspace(1e-5, 1e-6, 3),
        "rho_vs": np.logspace(1e-5, 1e-6, 3),
    }
)
trial23 = Trial({**trial18.solve_params, "conditioner": None})

Variant = namedtuple("Variant", ["ex", "sim_params"])
var_a = Variant(ParameterSearch2D, {})
var_b = Variant(ParameterSearch2D, {"gps_points": "first"})
var_c = Variant(ParameterSearch2D, {"gps_points": "multi-first"})
var_d = Variant(RigorousParameterSearch2D, {"sims": 20})
var_e = Variant(
    RigorousParameterSearch2D,
    {
        "sims": 20,
        "gps_points": "multi-first",
    },
)
