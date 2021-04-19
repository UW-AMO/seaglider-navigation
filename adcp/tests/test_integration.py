# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 11:04:36 2020

@author: 600301
"""
from typing import Tuple
import random

import numpy as np

from adcp import simulation as sim
from adcp import matbuilder as mb
from adcp import optimization as op
from adcp import viz


def test_integration():
    errors = standard_sim()
    assert errors["path_error"] < 2e6 and errors["current_error"] < 1e0


# %% ...or simulate new data
def standard_sim(
    t_scale=1e3,
    conditioner="tanh",
    vehicle_vel="otg",
    current_order=2,
    vehicle_order=2,
    rho_v=1e-8,
    rho_c=1e-8,
):
    rho_t = 1e-3
    rho_a = 1e-3
    rho_g = 1e-1

    sim_rho_v = 0
    sim_rho_c = 0
    sp = sim.SimParams(
        rho_t=rho_t,
        rho_a=rho_a,
        rho_g=rho_g,
        rho_v=sim_rho_v,
        rho_c=sim_rho_c,
        sigma_t=0.4,
        sigma_c=0.3,
        n_timepoints=2000,
        measure_points={"gps": "endpoints", "ttw": 0.5, "range": 0.05},
        vehicle_method="curved",
        curr_method="curved",
    )
    ddat, adat, x, curr_df, v_df = sim.simulate(sp, verbose=True)

    # %% No Range
    rho_v = 1e-8
    rho_c = 1e-8
    rho_g = rho_g
    rho_t = rho_t
    rho_a = rho_a
    rho_r = 0
    print(
        f"""Solution method covariances:
      vehicle process: {rho_v}
      current process:{rho_c}
      GPS measurement: {rho_g}
      TTW measurement: {rho_t}
      ADCP meawsurement: {rho_a}"""
    )

    prob = op.GliderProblem(
        ddat=ddat,
        adat=adat,
        rho_v=rho_v,
        rho_c=rho_c,
        rho_g=rho_g,
        rho_t=rho_t,
        rho_a=rho_a,
        rho_r=rho_r,
        t_scale=t_scale,
        conditioner=conditioner,
        vehicle_vel=vehicle_vel,
        current_order=current_order,
        vehicle_order=vehicle_order,
    )

    # %%  Solve problem
    x_sol = op.backsolve(prob)

    legacy = mb.legacy_select(
        prob.m,
        prob.n,
        prob.vehicle_order,
        prob.current_order,
        prob.vehicle_vel,
        prob=prob,
    )
    leg_Xs = mb.x_select(len(prob.times), 2)
    leg_NV = mb.nv_select(len(prob.times), len(prob.depths), 2, 2, "otg")
    leg_EV = mb.ev_select(len(prob.times), len(prob.depths), 2, 2, "otg")
    leg_NC = mb.nc_select(len(prob.times), len(prob.depths), 2, 2, "otg")
    leg_EC = mb.ec_select(len(prob.times), len(prob.depths), 2, 2, "otg")

    err = legacy @ x_sol - x
    path_error = (
        np.linalg.norm(leg_Xs @ leg_NV @ err) ** 2
        + np.linalg.norm(leg_Xs @ leg_EV @ err) ** 2
    )
    current_error = (
        np.linalg.norm(leg_EC @ err) ** 2 + np.linalg.norm(leg_NC @ err) ** 2
    )
    return {
        "path_error": path_error,
        "current_error": current_error,
        "x_true": x,
        "x_sol": x_sol,
        "prob": prob,
    }


# %%
def main():
    """Print summary detail and plots of solution"""
    res_dict = standard_sim()
    prob = res_dict["prob"]

    legacy = mb.legacy_select(
        prob.m,
        prob.n,
        prob.vehicle_order,
        prob.current_order,
        prob.vehicle_vel,
    )

    x_plot = legacy @ res_dict["x_sol"]
    x_true = legacy @ res_dict["x_true"]
    perror = res_dict["path_error"]
    cerror = res_dict["current_error"]

    print("Navigation position error:", perror)
    print("Current error:", cerror)
    plot_bundle(x_plot, x_true, prob)
    c1, c2, c3, c4 = check_condition(prob)
    print("100x100 sample of kalman matrix has condition number ", c1)
    print("100x100 sample of backsolve matrix has condition number ", c2)
    print("1000x1000 sample of kalman matrix has condition number ", c3)
    print("1000x1000 sample of backsolve matrix has condition number ", c4)


def plot_bundle(sol_x, x_true, prob):
    viz.vehicle_speed_plot(
        sol_x,
        prob.ddat,
        prob.times,
        prob.depths,
        direction="both",
        x_true=x_true,
        ttw=False,
    )
    viz.inferred_ttw_error_plot(
        sol_x, prob.adat, prob.ddat, direction="both", x_true=x_true
    )
    viz.current_depth_plot(
        sol_x, prob.adat, prob.ddat, direction="both", x_true=x_true
    )
    viz.inferred_adcp_error_plot(
        sol_x, prob.adat, prob.ddat, direction="both", x_true=x_true
    )
    viz.vehicle_posit_plot(
        sol_x,
        prob.ddat,
        prob.times,
        prob.depths,
        x_true=x_true,
        dead_reckon=True,
    )


def check_condition(prob: op.GliderProblem) -> Tuple:
    """Checks condition on matrices for glider problem"""
    m = len(prob.times)
    n = len(prob.depths)
    kalman_mat = op.gen_kalman_mat(prob)
    A, _ = op.solve_mats(prob)

    r100 = np.array(random.sample(range(0, 4 * m + 2 * n), 100))
    r1000 = np.array(random.sample(range(0, 4 * m + 2 * n), 1000))
    c1 = np.linalg.cond(kalman_mat.todense()[r100[:, None], r100])
    c2 = np.linalg.cond(A.todense()[r100[:, None], r100])
    c3 = np.linalg.cond(kalman_mat.todense()[r1000[:, None], r1000])
    c4 = np.linalg.cond(A.todense()[r1000[:, None], r1000])

    return c1, c2, c3, c4


if __name__ == "__main__":
    main()
