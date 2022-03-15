from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

import adcp
from adcp import viz
from adcp import matbuilder as mb
from adcp import optimization as op
from adcp import dataprep as dp
from adcp.exp import Experiment


class Cabage17(Experiment):
    def __init__(
        self,
        dive,
        rho_v=1e-3,
        rho_c=1e-3,
        rho_t=1e-2,
        rho_a=1e-2,
        rho_g=1e-0,
        rho_r=0,
        t_scale=1e0,
        conditioner="tanh",
        vehicle_vel="otg",
        current_order=2,
        vehicle_order=2,
        variant="Basic",
        last_gps=True,
    ):
        """Currently known dives:
        1980097, 1980099, 1960131, 1960132

        Arguments:
            last_gps: whether to remove the last GPS point from the data in
                order to simulate submerged navigation.
        """
        self.dive = dive
        self.name = "Real Dive Tracking"
        self.variant = variant
        self.config = adcp.ProblemConfig(
            t_scale, conditioner, vehicle_vel, current_order, vehicle_order
        )
        self.weights = adcp.Weights(rho_v, rho_c, rho_t, rho_a, rho_g, rho_r)
        self.last_gps = last_gps

    def run(self, visuals=True):
        data = adcp.ProblemData(
            dp.load_dive(self.dive), dp.load_adcp(self.dive)
        )
        last_gps_time = sorted(data.ddat["gps"].index)[-1]
        last_gps_posit = data.ddat["gps"].loc[last_gps_time]
        if not self.last_gps:
            data.ddat["gps"].drop(last_gps_time, inplace=True)
        prob = adcp.GliderProblem(data, self.config, self.weights)
        x_sol = op.backsolve(prob)

        c1, c2, c3, c4 = viz.check_condition(prob)
        viz.print_condition(c1, c2, c3, c4)

        # Calculate positional error to final GPS point
        gps_time_idx = np.argmin(
            abs(last_gps_time - pd.to_datetime(data.times))
        )
        Xs = prob.shape.Xs
        EV = prob.shape.EV
        NV = prob.shape.NV

        east_posit = (Xs @ EV @ x_sol)[gps_time_idx]
        north_posit = (Xs @ NV @ x_sol)[gps_time_idx]
        nav_error = (last_gps_posit["gps_nx_east"] - east_posit) ** 2 + (
            last_gps_posit["gps_ny_north"] - north_posit
        ) ** 2

        # identify which buoy data is relevant to trial
        mdat = dp.load_mooring("CBX16_T3_AUG2017.mat")
        first_time = data.ddat["depth"].index.min()
        first_time_buoy_idx = np.argmin(np.abs(first_time - mdat["time"]))
        first_buoy_currs_e = mdat["u"][first_time_buoy_idx]
        first_buoy_currs_n = mdat["v"][first_time_buoy_idx]
        first_buoy_depths = mdat["depth"][first_time_buoy_idx]
        first_buoy_bottom_depth = first_buoy_depths.max()
        first_buoy_top_depth = np.nan_to_num(
            first_buoy_depths, nan=np.inf
        ).min()
        last_time = data.ddat["depth"].index.max()
        last_time_buoy_idx = np.argmin(np.abs(last_time - mdat["time"]))
        last_buoy_currs_e = mdat["u"][last_time_buoy_idx]
        last_buoy_currs_n = mdat["v"][last_time_buoy_idx]
        last_buoy_depths = mdat["depth"][last_time_buoy_idx]
        last_buoy_bottom_depth = last_buoy_depths.max()
        last_buoy_top_depth = np.nan_to_num(last_buoy_depths, nan=np.inf).min()

        # identify which state current data is within buoy's water column
        first_state_depths_idx = np.argwhere(
            (data.depths > first_buoy_top_depth)
            & (data.depths <= first_buoy_bottom_depth)
        ).flatten()
        last_state_depths_idx = np.argwhere(
            (2 * data.deepest - data.depths > last_buoy_top_depth)
            & (2 * data.deepest - data.depths <= last_buoy_bottom_depth)
        ).flatten()
        first_state_depths = data.depths[first_state_depths_idx]
        last_state_depths = (
            2 * data.deepest - data.depths[last_state_depths_idx]
        )
        CV = prob.shape.CV
        EC = prob.shape.EC
        NC = prob.shape.NC
        first_state_currs_e = (CV @ EC @ x_sol)[first_state_depths_idx]
        first_state_currs_n = (CV @ NC @ x_sol)[first_state_depths_idx]
        last_state_currs_e = (CV @ EC @ x_sol)[last_state_depths_idx]
        last_state_currs_n = (CV @ NC @ x_sol)[last_state_depths_idx]

        # interpolate current from buoy and solution at all depths to get error
        first_depths = np.union1d(first_state_depths, first_buoy_depths)
        last_depths = np.union1d(last_state_depths, last_buoy_depths)
        interp_f_state_e = interp1d(
            first_state_depths, first_state_currs_e, fill_value="extrapolate"
        )
        interp_f_state_n = interp1d(
            first_state_depths, first_state_currs_n, fill_value="extrapolate"
        )
        interp_f_buoy_e = interp1d(
            first_buoy_depths, first_buoy_currs_e, fill_value="extrapolate"
        )
        interp_f_buoy_n = interp1d(
            first_buoy_depths, first_buoy_currs_n, fill_value="extrapolate"
        )
        descent_df = pd.DataFrame(
            {
                "state_e": interp_f_state_e(first_depths),
                "state_n": interp_f_state_n(first_depths),
                "buoy_e": interp_f_buoy_e(first_depths),
                "buoy_n": interp_f_buoy_n(first_depths),
            },
            index=first_depths,
        ).dropna()
        interp_f_state_e = interp1d(
            last_state_depths, last_state_currs_e, fill_value="extrapolate"
        )
        interp_f_state_n = interp1d(
            last_state_depths, last_state_currs_n, fill_value="extrapolate"
        )
        interp_f_buoy_e = interp1d(
            last_buoy_depths, last_buoy_currs_e, fill_value="extrapolate"
        )
        interp_f_buoy_n = interp1d(
            last_buoy_depths, last_buoy_currs_n, fill_value="extrapolate"
        )
        ascent_df = pd.DataFrame(
            {
                "state_e": interp_f_state_e(last_depths),
                "state_n": interp_f_state_n(last_depths),
                "buoy_e": interp_f_buoy_e(last_depths),
                "buoy_n": interp_f_buoy_n(last_depths),
            },
            index=last_depths,
        ).dropna()

        descent_df["delta_e"] = descent_df["state_e"] - descent_df["buoy_e"]
        descent_df["delta_n"] = descent_df["state_n"] - descent_df["buoy_n"]
        ascent_df["delta_e"] = ascent_df["state_e"] - ascent_df["buoy_e"]
        ascent_df["delta_n"] = ascent_df["state_n"] - ascent_df["buoy_n"]

        mean_squared_error = (
            trapezoid(descent_df["delta_e"] ** 2, descent_df.index)
            + trapezoid(descent_df["delta_n"] ** 2, descent_df.index)
            + trapezoid(ascent_df["delta_e"] ** 2, ascent_df.index)
            + trapezoid(ascent_df["delta_n"] ** 2, ascent_df.index)
        ) / (
            descent_df.index[-1]
            - descent_df.index[0]
            + ascent_df.index[-1]
            - ascent_df.index[0]
        )

        # Plotting.
        legacy = mb.legacy_select(
            prob.shape.m,
            prob.shape.n,
            prob.config.vehicle_order,
            prob.config.current_order,
            prob.config.vehicle_vel,
            prob=prob,
        )
        x_leg = legacy @ x_sol
        if visuals:
            viz.plot_bundle(
                x_leg,
                data.adat,
                data.ddat,
                data.times,
                data.depths,
                x=None,
                mdat=mdat,
                final_posit=last_gps_posit,
            )

        return {"metrics": [nav_error, mean_squared_error]}


Trial = namedtuple("Trial", ["ex", "solve_params"])
trial1 = Trial(Cabage17, {"rho_r": 1e-4})
trial2 = Trial(Cabage17, {"current_order": 3, "vehicle_order": 3})
trial3 = Trial(Cabage17, {"vehicle_vel": "otg-cov"})
trial4 = Trial(
    Cabage17,
    {
        "vehicle_vel": "otg-cov",
        "current_order": 3,
        "vehicle_order": 3,
    },
)
basic_solve_params = {
    "rho_t": 1e-2,
    "rho_a": 5e-2,
    "rho_g": 1e0,
    "rho_r": 0,
    "t_scale": 1e0,
}
trial5 = Trial(Cabage17, basic_solve_params)
trial6 = Trial(
    Cabage17,
    {
        "current_order": 3,
        "vehicle_order": 3,
        **basic_solve_params,
        "rho_v": 5e-5,
        "rho_c": 5e-3,
    },
)
trial7 = Trial(
    Cabage17,
    {
        "vehicle_vel": "otg-cov",
        **basic_solve_params,
        "rho_v": 1e-1,
        "rho_c": 1e-1,
    },
)
trial8 = Trial(
    Cabage17,
    {
        "vehicle_vel": "otg-cov",
        "current_order": 3,
        "vehicle_order": 3,
        **basic_solve_params,
        "rho_v": 1e-5,
        "rho_c": 1e-4,
    },
)
trial9 = Trial(
    Cabage17,
    {
        "current_order": 3,
        "vehicle_order": 3,
        **basic_solve_params,
        "rho_v": 5e-5,
        "rho_c": 5e-5,
        "rho_t": 1e-1,
        "rho_a": 1e-1,
    },
)

trial11 = Trial(
    Cabage17,
    {
        "vehicle_vel": "otg-cov",
        "current_order": 3,
        "vehicle_order": 3,
        **basic_solve_params,
        "rho_v": 1e-8,
        "rho_c": 1e-8,
        "rho_t": 1e-1,
        "rho_a": 1e-1,
    },
)

trial12 = Trial(
    Cabage17,
    {
        "current_order": 3,
        "vehicle_order": 3,
        **basic_solve_params,
        "rho_v": 1e-6,
        "rho_c": 1e-6,
        "rho_t": 1e-1,
        "rho_a": 1e-1,
    },
)

trial13 = Trial(
    Cabage17,
    {
        "current_order": 3,
        "vehicle_order": 3,
        **basic_solve_params,
        "rho_v": 1e-8,
        "rho_c": 1e-8,
        "rho_t": 1e-1,
        "rho_a": 1e-1,
    },
)
trial14 = Trial(
    Cabage17,
    {
        "current_order": 3,
        "vehicle_order": 3,
        **basic_solve_params,
        "rho_v": 1e-1,
        "rho_c": 1e-1,
        "rho_t": 1e-1,
        "rho_a": 1e-1,
    },
)

Variant = namedtuple(
    "Variant",
    [
        "data_params",
    ],
)
var_a = Variant({"dive": 1980097})
var_b = Variant({"dive": 1980099})
var_c = Variant({"dive": 1960131})
var_d = Variant({"dive": 1960132})
var_e = Variant({"dive": 1980097, "last_gps": False})
var_f = Variant({"dive": 1980099, "last_gps": False})
var_g = Variant({"dive": 1960131, "last_gps": False})
var_h = Variant({"dive": 1960132, "last_gps": False})
