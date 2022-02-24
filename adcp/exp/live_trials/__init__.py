from collections import namedtuple

import adcp
from adcp import viz
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
        t_scale=1e1,
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

        moor_data = dp.load_mooring("CBX16_T3_AUG2017.mat")
        if visuals:
            viz.plot_bundle(
                x_sol,
                data.adat,
                data.ddat,
                data.times,
                data.depths,
                x=None,
                dac=True,
                mdat=moor_data,
            )


Trial = namedtuple("Trial", ["ex", "prob_params", "sim_params"])
var_a = {"ex": Cabbage17, "sim_params": {"dive": 1980097}}
trial1 = {"prob_params": {}}
trial1a = Trial(**trial1, **var_a)
