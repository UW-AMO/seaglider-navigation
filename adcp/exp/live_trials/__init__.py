from collections import namedtuple

import numpy as np

import adcp
from adcp import viz
from adcp import matbuilder as mb
from adcp import optimization as op
from adcp import dataprep as dp
from adcp.exp import Experiment


class Cabbage17(Experiment):
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
    ):
        """Currently known dives:
        1980097, 1980099, 1960131, 1960132
        """
        self.dive = dive
        self.name = "Real Dive Tracking"
        self.variant = variant
        self.config = adcp.ProblemConfig(
            t_scale, conditioner, vehicle_vel, current_order, vehicle_order
        )
        self.weights = adcp.Weights(rho_v, rho_c, rho_t, rho_a, rho_g, rho_r)

    def run(self, visuals=True):
        data = adcp.ProblemData(
            dp.load_dive(self.dive), dp.load_adcp(self.dive)
        )
        prob = adcp.GliderProblem(data, self.config, self.weights)
        x_sol = op.backsolve(prob)

        mdat = dp.load_mooring("CBX16_T3_AUG2017.mat")

        first_time = data.ddat["depth"].index.min()
        first_time_buoy_idx = np.argmin(np.abs(first_time - mdat["time"]))
        first_buoy_depths = mdat["depth"][
            first_time_buoy_idx,
        ]
        first_buoy_bottom_depth = first_buoy_depths.max()
        first_buoy_top_depth = np.nan_to_num(
            first_buoy_depths, nan=np.inf
        ).min()
        last_time = data.ddat["depth"].index.max()
        last_time_buoy_idx = np.argmin(np.abs(last_time - mdat["time"]))
        last_buoy_depths = mdat["depth"][
            last_time_buoy_idx,
        ]
        last_buoy_bottom_depth = last_buoy_depths.max()
        last_buoy_top_depth = np.nan_to_num(last_buoy_depths, nan=np.inf).min()
        first_state_depths_idx = (  # noqa
            data.depths > first_buoy_top_depth
        ) & (data.depths <= first_buoy_bottom_depth)
        last_state_depths_idx = (  # noqa
            data.depths.max() - data.depths > last_buoy_top_depth
        ) & (data.depths.max() - data.depths <= last_buoy_bottom_depth)

        # east_eval_currents = (VC @ EC x_sol)[first_state_depths_idx
        # + last_stape_depths_idx]
        # north_eval_currents = (VC @ NC x_sol)[first_state_depths_idx
        # + last_stape_depths_idx]

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
                dac=True,
                mdat=mdat,
            )
        return {"metrics": [0]}


Trial = namedtuple("Trial", ["ex", "prob_params", "sim_params"])
var_a = {"ex": Cabbage17, "sim_params": {"dive": 1980097}}
var_b = {"ex": Cabbage17, "sim_params": {"dive": 1980099}}
var_c = {"ex": Cabbage17, "sim_params": {"dive": 1960131}}
var_d = {"ex": Cabbage17, "sim_params": {"dive": 1960132}}
trial1 = {"prob_params": {}}
trial2 = {"prob_params": {"current_order": 3, "vehicle_order": 3}}
trial3 = {"prob_params": {"vehicle_vel": "otg-cov"}}
trial4 = {
    "prob_params": {
        "vehicle_vel": "otg-cov",
        "current_order": 3,
        "vehicle_order": 3,
    }
}
trial1a = Trial(**trial1, **var_a)
trial2a = Trial(**trial2, **var_a)
trial3a = Trial(**trial3, **var_a)
trial4a = Trial(**trial4, **var_a)
