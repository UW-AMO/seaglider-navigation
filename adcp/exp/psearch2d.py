# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 11:04:36 2020

@author: 600301
"""
from itertools import product, repeat

import numpy as np
from matplotlib import colors

from adcp import simulation as sim
from adcp import matbuilder as mb
from adcp import optimization as op
from adcp import viz
from adcp.exp import Experiment


class ParameterSearch2D(Experiment):
    default_prob = op.GliderProblem(
        rho_g=1e-0,
        rho_t=1e-2,
        rho_a=1e-2,
        rho_r=0,
        t_scale=1e3,
        conditioner="tanh",
        vehicle_vel="otg",
        current_order=2,
        vehicle_order=2,
    )
    default_sp = sim.SimParams(
        rho_t=1e-2,
        rho_a=1e-2,
        rho_g=1e-0,
        rho_v=0,
        rho_c=0,
        sigma_t=0.4,
        sigma_c=0.3,
        n_timepoints=2000,
        measure_points={"gps": "endpoints", "ttw": 0.5, "range": 0.05},
        vehicle_method="curved",
        curr_method="curved",
    )

    def __init__(
        self,
        rho_vs=np.logspace(-10, -1, 6),
        rho_cs=np.logspace(-10, -1, 6),
        t_scale=1e3,
        conditioner="tanh",
        vehicle_vel="otg",
        current_order=2,
        vehicle_order=2,
        gps_points="endpoints",
        seed=3453,
        sp=default_sp,
        prob=default_prob,
        variant="Basic",
    ):
        self.name = "2D Parameter Search"
        self.variant = variant
        # unexpected parameters
        self.overrides = {}
        if sp != self.default_sp:
            self.overrides["SimParams":sp]
        if prob != self.default_prob:
            self.overrides["GliderProblem":prob]
        self.sp = sp
        self.prob = prob
        # expected parameters dominate
        self.prob = op.GliderProblem(
            copyobj=self.prob,
            t_scale=t_scale,
            conditioner=conditioner,
            vehicle_vel=vehicle_vel,
            current_order=current_order,
            vehicle_order=vehicle_order,
        )
        self.sp.measure_points = {**self.sp.measure_points, "gps": gps_points}
        self.rho_vs = rho_vs
        self.rho_cs = rho_cs
        self.errmap = np.zeros((2, len(rho_vs), len(rho_cs)))
        self.paths = list(repeat(list(repeat(None, len(rho_cs))), len(rho_vs)))
        self.output = None
        self.seed = seed

    def gen_data(self):
        ddat, adat, x, curr_df, v_df = sim.simulate(self.sp)
        self.ddat = ddat
        self.adat = adat
        self.x = x
        self.curr_df = curr_df
        self.v_df = v_df
        self.prob = op.GliderProblem(copyobj=self.prob, ddat=ddat, adat=adat)

    def run(self, visuals=True):
        self.gen_data()
        for ((i, rv), (j, rc)) in product(
            enumerate(self.rho_vs), enumerate(self.rho_cs)
        ):
            prob = op.GliderProblem(copyobj=self.prob, rho_v=rv, rho_c=rc)

            x_sol = op.backsolve(prob)

            legacy = mb.legacy_select(
                prob.m,
                prob.n,
                prob.vehicle_order,
                prob.current_order,
                prob.vehicle_vel,
                prob=prob,
            )

            x_leg = legacy @ x_sol

            leg_Xs = mb.x_select(prob.m, 2)
            leg_Vs = mb.v_select(prob.m, 2)
            leg_NV = mb.nv_select(
                len(prob.times), len(prob.depths), 2, 2, "otg"
            )
            leg_EV = mb.ev_select(
                len(prob.times), len(prob.depths), 2, 2, "otg"
            )
            leg_NC = mb.nc_select(
                len(prob.times), len(prob.depths), 2, 2, "otg"
            )
            leg_EC = mb.ec_select(
                len(prob.times), len(prob.depths), 2, 2, "otg"
            )
            self.prob.EV = leg_EV
            self.prob.NV = leg_NV
            self.prob.EC = leg_EC
            self.prob.NC = leg_NC
            self.prob.Xs = leg_Xs
            self.prob.Vs = leg_Vs

            err = x_leg - self.x
            path_error = (
                np.linalg.norm(leg_Xs @ leg_NV @ err) ** 2
                + np.linalg.norm(leg_Xs @ leg_EV @ err) ** 2
            )
            current_error = (
                np.linalg.norm(leg_EC @ err) ** 2
                + np.linalg.norm(leg_NC @ err) ** 2
            )
            self.errmap[0, i, j] = path_error
            self.errmap[1, i, j] = current_error
            self.paths[i][j] = x_leg
        if visuals:
            i1, j1, i2, j2 = self.best_parameters()
            self.display_errmaps(i1, j1, i2, j2)
            self.display_solutions(i1, j1, i2, j2)
        return {
            "errmap": self.errmap,
            "paths": self.paths,
            "x": self.x,
            "curr_df": self.curr_df,
            "v_df": self.v_df,
            "metrics": self.errmap.min(axis=(1, 2)),
        }

    def best_parameters(self):
        i2, j2 = np.unravel_index(
            self.errmap[1, :, :].argmin(), (len(self.rho_vs), len(self.rho_cs))
        )
        i1, j1 = np.unravel_index(
            self.errmap[0, :, :].argmin(), (len(self.rho_vs), len(self.rho_cs))
        )
        return i1, j1, i2, j2

    def display_errmaps(self, i1, j1, i2, j2):
        viz.show_errmap(
            self.errmap,
            0,
            self.rho_vs,
            self.rho_cs,
            norm=colors.LogNorm(vmin=1e7, vmax=2e9),
        )
        viz.show_errmap(
            self.errmap,
            1,
            self.rho_vs,
            self.rho_cs,
            norm=colors.LogNorm(vmin=1e0, vmax=1e3),
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

    def display_solutions(self, i1, j1, i2, j2):
        nav_x = self.paths[i1][j1]
        prob = op.GliderProblem(
            copyobj=self.prob,
            rho_v=self.rho_vs[i1],
            rho_c=self.rho_cs[j1],
        )
        try:
            c1, c2, c3, c4 = viz.check_condition(prob)
            viz.print_condition(c1, c2, c3, c4)
        except ValueError:
            print("small matrices")

        viz.plot_bundle(
            nav_x, self.prob, self.prob.times, self.prob.depths, self.x
        )
        if i1 != i2 or j1 != j2:
            curr_x = self.paths[i2][j2]
            prob = op.GliderProblem(
                copyobj=self.prob,
                rho_v=self.rho_vs[i2],
                rho_c=self.rho_cs[j2],
            )
            viz.plot_bundle(
                curr_x, self.prob, self.prob.times, self.prob.depths, self.x
            )
            try:
                c1, c2, c3, c4 = viz.check_condition(prob)
                viz.print_condition(c1, c2, c3, c4)
            except ValueError:
                print("small matrices")
        else:
            print("... and it's also the best current solution")


class RigorousParameterSearch2D(ParameterSearch2D):
    def __init__(self, sims=20, **kwargs):
        self.sub_problem_kwargs = kwargs
        self.sims = sims
        super().__init__(**kwargs)

    def run(self):
        results = []
        for n_sim in range(self.sims):
            curr_search = ParameterSearch2D(**self.sub_problem_kwargs)
            results.append(curr_search.run(visuals=False))

        self.errmap = np.zeros((2, len(self.rho_vs), len(self.rho_cs)))
        for res in results:
            self.errmap += res["errmap"] / self.sims

        i1, j1, i2, j2 = self.best_parameters()
        self.display_errmaps(i1, j1, i2, j2)
        return {"output": self.errmap, "metrics": self.errmap.min(axis=(1, 2))}
