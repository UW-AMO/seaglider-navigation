from itertools import product
from collections import namedtuple

from matplotlib import colors
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
        self.data = adcp.ProblemData(
            dp.load_dive(self.dive), dp.load_adcp(self.dive)
        )
        mdat = dp.load_mooring("CBX16_T3_AUG2017.mat")
        first_time = self.data.ddat["depth"].index.min()
        last_time = self.data.ddat["depth"].index.max()
        mdat = dp.interpolate_mooring(mdat, [first_time, last_time])
        self.mdat = mdat
        self.last_gps_time = sorted(self.data.ddat["gps"].index)[-1]
        last_gps_posit = self.data.ddat["gps"].loc[self.last_gps_time]
        self.last_gps_posit = last_gps_posit
        if not self.last_gps:
            self.data.ddat["gps"].drop(self.last_gps_time, inplace=True)

    def run(self, visuals=True):
        prob = adcp.GliderProblem(self.data, self.config, self.weights)
        x_sol = op.backsolve(prob)

        c1, c2, c3, c4 = viz.check_condition(prob)
        viz.print_condition(c1, c2, c3, c4)

        # Calculate positional error to final GPS point
        gps_time_idx = np.argmin(
            abs(self.last_gps_time - pd.to_datetime(self.data.times))
        )
        Xs = prob.shape.Xs
        EV = prob.shape.EV
        NV = prob.shape.NV
        EC = prob.shape.EC
        NC = prob.shape.NC

        east_posit = (Xs @ EV @ x_sol)[gps_time_idx]
        north_posit = (Xs @ NV @ x_sol)[gps_time_idx]
        nav_error = (self.last_gps_posit["gps_nx_east"] - east_posit) ** 2 + (
            self.last_gps_posit["gps_ny_north"] - north_posit
        ) ** 2

        # identify which buoy data is relevant to trial
        first_buoy_currs_e = self.mdat["u"][0]
        first_buoy_currs_n = self.mdat["v"][0]
        first_buoy_depths = self.mdat["depth"][0]
        first_buoy_bottom_depth = first_buoy_depths.max()
        first_buoy_top_depth = np.nan_to_num(
            first_buoy_depths, nan=np.inf
        ).min()
        last_buoy_currs_e = self.mdat["u"][1]
        last_buoy_currs_n = self.mdat["v"][1]
        last_buoy_depths = self.mdat["depth"][1]
        last_buoy_bottom_depth = last_buoy_depths.max()
        last_buoy_top_depth = np.nan_to_num(last_buoy_depths, nan=np.inf).min()

        # identify which state current data is within buoy's water column
        first_state_depths_idx = np.argwhere(
            (self.data.depths > first_buoy_top_depth)
            & (self.data.depths <= first_buoy_bottom_depth)
        ).flatten()
        last_state_depths_idx = np.argwhere(
            (2 * self.data.deepest - self.data.depths > last_buoy_top_depth)
            & (
                2 * self.data.deepest - self.data.depths
                <= last_buoy_bottom_depth
            )
        ).flatten()
        first_state_depths = self.data.depths[first_state_depths_idx]
        last_state_depths = (
            2 * self.data.deepest - self.data.depths[last_state_depths_idx]
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

        current_error = (
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

        # Measurement error metrics
        z_ttw_n = mb.get_zttw(prob.data.ddat, "north", prob.config.t_scale)
        z_ttw_e = mb.get_zttw(prob.data.ddat, "east", prob.config.t_scale)
        A, B = mb.uv_select(
            prob.data.times,
            prob.data.depths,
            prob.data.ddat,
            prob.data.adat,
            vehicle_vel=prob.config.vehicle_vel,
        )
        ttw_sol_n = (A @ prob.shape.Vs @ NV - B @ prob.shape.CV @ NC) @ x_sol
        ttw_sol_e = (A @ prob.shape.Vs @ EV - B @ prob.shape.CV @ EC) @ x_sol
        ttw_mse = np.linalg.norm(z_ttw_n - ttw_sol_n) ** 2 / (2 * len(z_ttw_n))
        ttw_mse += np.linalg.norm(z_ttw_e - ttw_sol_e) ** 2 / (
            2 * len(z_ttw_e)
        )
        A, B = mb.adcp_select(
            prob.data.times,
            prob.data.depths,
            prob.data.ddat,
            prob.data.adat,
            vehicle_vel=prob.config.vehicle_vel,
        )
        z_adcp_n = mb.get_zadcp(prob.data.adat, "north", prob.config.t_scale)
        z_adcp_e = mb.get_zadcp(prob.data.adat, "east", prob.config.t_scale)
        adcp_sol_n = (A @ prob.shape.Vs @ NV - B @ prob.shape.CV @ NC) @ x_sol
        adcp_sol_e = (A @ prob.shape.Vs @ EV - B @ prob.shape.CV @ EC) @ x_sol
        adcp_mse = np.linalg.norm(z_adcp_n - adcp_sol_n) ** 2 / (
            2 * len(z_adcp_n)
        )
        adcp_mse += np.linalg.norm(z_adcp_e - adcp_sol_e) ** 2 / (
            2 * len(z_adcp_e)
        )

        print("ttw_mse: ", ttw_mse)
        print("adcp_mse: ", adcp_mse)
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
                self.data.adat,
                self.data.ddat,
                self.data.times,
                self.data.depths,
                x=None,
                mdat=self.mdat,
                final_posit=self.last_gps_posit,
            )

        return {"path": x_leg, "metrics": [nav_error, current_error]}


class ParameterSearch2D(Cabage17):
    def __init__(
        self,
        rho_vs=np.logspace(-10, -1, 10),
        rho_cs=np.logspace(-10, -1, 10),
        **kwargs,
    ):
        self.sub_problem_kwargs = kwargs
        self.rho_vs = rho_vs
        self.rho_cs = rho_cs
        self.errmap = np.zeros((2, len(self.rho_vs), len(self.rho_cs)))
        self.paths = []
        super().__init__(**kwargs)

    def run(self, visuals=True):
        for (i, rv), (j, rc) in product(
            enumerate(self.rho_vs), enumerate(self.rho_cs)
        ):
            experiment = Cabage17(
                **self.sub_problem_kwargs, rho_v=rv, rho_c=rc
            )
            result = experiment.run(visuals=False)
            self.errmap[:, i, j] = result["metrics"]
            self.paths.append(result["path"])

        i1, j1, i2, j2 = self.best_parameters()
        if visuals:
            self.display_errmaps(i1, j1, i2, j2)
            print("Best Navigation Solution")
            self.display_solutions(i1, j1)
            if i1 != i2 or j1 != j2:
                print("Best Current Solution")
                self.display_solutions(i2, j2)
            else:
                print("... and it's also the best current solution")
        return {"output": self.errmap, "metrics": self.errmap.min(axis=(1, 2))}

    def display_solutions(self, i, j):
        best_x = self.paths[i * len(self.rho_vs) + j]
        weights = adcp.Weights(
            self.rho_vs[i],
            self.rho_cs[j],
            self.weights.rho_t,
            self.weights.rho_a,
            self.weights.rho_g,
            self.weights.rho_r,
        )
        prob = adcp.GliderProblem(self.data, self.config, weights)
        try:
            c1, c2, c3, c4 = viz.check_condition(prob)
            viz.print_condition(c1, c2, c3, c4)
        except ValueError:
            print("small matrices")

        viz.plot_bundle(
            best_x,
            self.data.adat,
            self.data.ddat,
            self.data.times,
            self.data.depths,
            x=None,
            mdat=self.mdat,
            final_posit=self.last_gps_posit,
        )

    def best_parameters(self):
        i2, j2 = np.unravel_index(
            self.errmap[1, :, :].argmin(), (len(self.rho_vs), len(self.rho_cs))
        )
        i1, j1 = np.unravel_index(
            self.errmap[0, :, :].argmin(), (len(self.rho_vs), len(self.rho_cs))
        )
        return i1, j1, i2, j2

    def display_errmaps(self, i1, j1, i2, j2):
        print("Navigation Error Map")
        viz.show_errmap(
            self.errmap,
            0,
            self.rho_vs,
            self.rho_cs,
            norm=colors.LogNorm(vmin=1e0, vmax=1e6),
        )
        print("Current Error Map")
        viz.show_errmap(
            self.errmap,
            1,
            self.rho_vs,
            self.rho_cs,
            norm=colors.LogNorm(vmin=1e-1, vmax=1e0),
        )
        # %%
        print(
            f"Best Nav Solution: rho_v={self.rho_vs[i1]:.2E},"
            f" rho_c={self.rho_cs[j1]:.2E}"
        )
        print(
            f"Creates path error: {self.errmap[0, i1, j1]:.2E} and current "
            f"error: {self.errmap[1, i1, j1]:.2E}"
        )
        if (i1 != i2) or (j1 != j2):
            print(
                f"Best Current Solution: rho_v={self.rho_vs[i2]:.2E}, "
                f"rho_c={self.rho_cs[j2]:.2E}"
            )
            print(
                f"Creates path error: {self.errmap[0, i2, j2]:.2E} and current"
                f" error: {self.errmap[1, i2, j2]:.2E}"
            )


Trial = namedtuple("Trial", ["ex", "solve_params"])
trial1 = Trial(Cabage17, {})
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
trial15 = Trial(
    Cabage17,
    {
        "current_order": 3,
        "vehicle_order": 3,
        **basic_solve_params,
        "rho_v": 5e-5,
        "rho_c": 1e-3,
        "rho_t": 1e3,
        "rho_a": 1e-1,
    },
)
trial16 = Trial(
    Cabage17,
    {
        "current_order": 3,
        "vehicle_order": 3,
        **basic_solve_params,
        "rho_v": 5e-5,
        "rho_c": 1e-3,
        "rho_t": 1e5,
        "rho_a": 1e-1,
    },
)
trial17 = Trial(
    Cabage17,
    {
        "current_order": 3,
        "vehicle_order": 3,
        **basic_solve_params,
        "rho_v": 5e-5,
        "rho_c": 1e-5,
        "rho_t": 1e5,
        "rho_a": 1e-1,
    },
)
trial18 = Trial(
    Cabage17,
    {
        "current_order": 3,
        "vehicle_order": 3,
        **basic_solve_params,
        "rho_v": 5e-5,
        "rho_c": 1e-3,
        "rho_t": 1e-1,
        "rho_a": 1e5,
    },
)
trial19 = Trial(
    Cabage17,
    {
        "current_order": 3,
        "vehicle_order": 3,
        **basic_solve_params,
        "rho_v": 5e-5,
        "rho_c": 1e-3,
        "rho_t": 1e5,
        "rho_a": 1e-4,
    },
)
trial20 = Trial(
    Cabage17,
    {
        "t_scale": 1e-3,
        "current_order": 3,
        "vehicle_order": 3,
        **basic_solve_params,
        "rho_v": 5e-5,
        "rho_c": 1e-3,
        "rho_t": 1e-1,
        "rho_a": 1e5,
    },
)
trial21 = Trial(
    ParameterSearch2D,
    {
        "vehicle_vel": "otg",
        "current_order": 2,
        "vehicle_order": 2,
        "rho_vs": np.logspace(-5, -6, 2),
        "rho_cs": np.logspace(-5, -6, 2),
    },
)
trial22 = Trial(
    ParameterSearch2D,
    {
        "vehicle_vel": "otg",
        **basic_solve_params,
        "current_order": 2,
        "vehicle_order": 2,
    },
)
trial23 = Trial(
    ParameterSearch2D,
    {
        "vehicle_vel": "otg-cov",
        **basic_solve_params,
        "current_order": 2,
        "vehicle_order": 2,
    },
)
trial24 = Trial(
    ParameterSearch2D,
    {
        "vehicle_vel": "otg",
        **basic_solve_params,
        "current_order": 3,
        "vehicle_order": 3,
    },
)
trial25 = Trial(
    ParameterSearch2D,
    {
        "vehicle_vel": "otg-cov",
        **basic_solve_params,
        "current_order": 3,
        "vehicle_order": 3,
    },
)
trial26 = Trial(
    ParameterSearch2D,
    {
        "vehicle_vel": "otg",
        **basic_solve_params,
        "current_order": 2,
        "vehicle_order": 2,
        "conditioner": None,
    },
)
trial27 = Trial(
    ParameterSearch2D,
    {
        "vehicle_vel": "otg",
        **basic_solve_params,
        "current_order": 2,
        "vehicle_order": 2,
        "t_scale": 1e3,
    },
)
trial28 = Trial(
    ParameterSearch2D,
    {
        "vehicle_vel": "otg",
        **basic_solve_params,
        "current_order": 2,
        "vehicle_order": 2,
        "rho_t": 1e4,
    },
)


Variant = namedtuple("Variant", ["data_params"])
var_a = Variant({"dive": 1980097})
var_b = Variant({"dive": 1980099})
var_c = Variant({"dive": 1960131})
var_d = Variant({"dive": 1960132})
var_e = Variant({"dive": 1980097, "last_gps": False})
var_f = Variant({"dive": 1980099, "last_gps": False})
var_g = Variant({"dive": 1960131, "last_gps": False})
var_h = Variant({"dive": 1960132, "last_gps": False})
