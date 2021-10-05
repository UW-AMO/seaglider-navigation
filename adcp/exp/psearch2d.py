# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 11:04:36 2020

@author: 600301
"""
from itertools import product

import numpy as np
from matplotlib import colors

from adcp import simulation as sim
from adcp import matbuilder as mb
from adcp import optimization as op
from adcp import viz
from adcp.exp import Experiment
import adcp


class ParameterSearch2D(Experiment):
    rho_g = 1e-0
    rho_t = 1e-2
    rho_a = 1e-2
    rho_r = 0
    t_scale = 1e3
    conditioner = "tanh"
    vehicle_vel = "otg"
    current_order = 2
    vehicle_order = 2

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
        variant="Basic",
    ):
        self.name = "2D Parameter Search"
        self.variant = variant

        # self.prob = prob
        # # expected parameters dominate
        # self.prob = op.GliderProblem(
        #     copyobj=self.prob,
        #     t_scale=t_scale,
        #     conditioner=conditioner,
        #     vehicle_vel=vehicle_vel,
        #     current_order=current_order,
        #     vehicle_order=vehicle_order,
        # )
        self.sp = sp
        self.sp.measure_points["gps"] = gps_points
        self.rho_vs = rho_vs
        self.rho_cs = rho_cs
        self.errmap = np.zeros((2, len(rho_vs), len(rho_cs)))
        self.paths = []
        self.seed = seed
        self.config = adcp.ProblemConfig(
            t_scale, conditioner, vehicle_vel, current_order, vehicle_order
        )

    def gen_data(self):
        ddat, adat, x, curr_df, v_df = sim.simulate(self.sp)
        self.x = x
        self.curr_df = curr_df
        self.v_df = v_df
        return ddat, adat

    def run(self, visuals=True):
        self.data = adcp.ProblemData(*self.gen_data())
        self.shape = adcp.StateVectorShape(self.data, self.config)

        legacy_size_shape = adcp.create_legacy_shape(self.shape)
        for ((i, rv), (j, rc)) in product(
            enumerate(self.rho_vs), enumerate(self.rho_cs)
        ):
            rho_t = type(self).rho_t
            rho_a = type(self).rho_a
            rho_g = type(self).rho_g
            rho_r = type(self).rho_r
            weights = adcp.Weights(rv, rc, rho_t, rho_a, rho_g, rho_r)
            prob = adcp.GliderProblem(self.data, self.config, weights)

            x_sol = op.backsolve(prob)
            if np.isnan(x_sol.max()):  # too ill conditioned, couldn't solve
                self.errmap[:, i, j] = [np.inf, np.inf]
                self.paths.append(None)
                continue
            legacy = mb.legacy_select(
                prob.shape.m,
                prob.shape.n,
                prob.config.vehicle_order,
                prob.config.current_order,
                prob.config.vehicle_vel,
                prob=prob,
            )
            x_leg = legacy @ x_sol

            err = x_leg - self.x
            path_error = (
                sum((legacy_size_shape.Xs @ legacy_size_shape.EV @ err) ** 2)
                / legacy_size_shape.Xs.shape[0]
                + sum((legacy_size_shape.Xs @ legacy_size_shape.NV @ err) ** 2)
                / legacy_size_shape.Xs.shape[0]
            )
            path_error = np.sqrt(path_error)
            current_error = (
                sum((legacy_size_shape.EC @ err) ** 2)
                / legacy_size_shape.EC.shape[0]
                + sum((legacy_size_shape.NC @ err) ** 2)
                / legacy_size_shape.NC.shape[0]
            )
            current_error = np.sqrt(current_error)
            self.errmap[0, i, j] = path_error
            self.errmap[1, i, j] = current_error
            self.paths.append(x_leg)
        if visuals:
            i1, j1, i2, j2 = self.best_parameters()
            self.display_errmaps(i1, j1, i2, j2)
            self.display_solutions(i1, j1)
            if i1 != i2 or j1 != j2:
                self.display_solutions(i2, j2)
            else:
                print("... and it's also the best current solution")
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
            norm=colors.LogNorm(vmin=3e1, vmax=3e3),
        )
        viz.show_errmap(
            self.errmap,
            1,
            self.rho_vs,
            self.rho_cs,
            norm=colors.LogNorm(vmin=1e-2, vmax=1e0),
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

    def display_solutions(self, i, j):
        best_x = self.paths[i * len(self.rho_vs) + j]
        weights = adcp.Weights(
            self.rho_vs[i],
            self.rho_cs[j],
            self.rho_t,
            self.rho_a,
            self.rho_g,
            self.rho_r,
        )
        prob = adcp.GliderProblem(self.data, self.config, weights)
        try:
            c1, c2, c3, c4 = viz.check_condition(prob)
            viz.print_condition(c1, c2, c3, c4)
        except ValueError:
            print("small matrices")

        viz.plot_bundle(
            best_x,
            prob.data.adat,
            prob.data.ddat,
            prob.data.times,
            prob.data.depths,
            self.x,
        )
        weights = adcp.Weights(
            self.rho_vs[i],
            self.rho_cs[j],
            self.rho_t,
            self.rho_a,
            self.rho_g,
            self.rho_r,
        )
        prob = adcp.GliderProblem(self.data, self.config, weights)
        n_ttw_e = len(mb.get_zttw(prob.data.ddat, "east", prob.config.t_scale))
        n_ttw_n = len(
            mb.get_zttw(prob.data.ddat, "north", prob.config.t_scale)
        )
        n_adcp_e = len(
            mb.get_zadcp(prob.data.adat, "east", prob.config.t_scale)
        )
        n_adcp_n = len(
            mb.get_zadcp(prob.data.adat, "north", prob.config.t_scale)
        )
        n_gps_e = len(mb.get_zgps(prob.data.ddat, "east"))
        n_gps_n = len(mb.get_zgps(prob.data.ddat, "north"))

        AtAinv_vel = prob.AtAinv
        AtAinv_pos, v_pos_cols, c_pos_cols = op.solution_variance_estimator(
            prob.AtA,
            prob.shape.m,
            prob.shape.n,
            prob.config.current_order,
            prob.config.vehicle_order,
            prob.config.vehicle_vel,
            1,
        )
        viz.display_uncertainty(
            AtAinv_vel,
            prob.A,
            prob.AtAinv_v_points,
            prob.AtAinv_c_points,
            (n_ttw_e, n_ttw_n, n_adcp_e, n_adcp_n, n_gps_e, n_gps_n),
        )
        viz.display_uncertainty(
            AtAinv_pos,
            prob.A,
            v_pos_cols,
            c_pos_cols,
            (n_ttw_e, n_ttw_n, n_adcp_e, n_adcp_n, n_gps_e, n_gps_n),
        )


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
