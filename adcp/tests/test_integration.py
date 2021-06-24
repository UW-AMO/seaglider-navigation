# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 11:04:36 2020

@author: 600301
"""
import numpy as np

from adcp import simulation as sim
from adcp import matbuilder as mb
from adcp import optimization as op
from adcp import viz


def test_integration():
    errors = standard_sim()
    assert errors["path_error"] < 1e7 and errors["current_error"] < 1e1


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
    rho_t = 1e-2
    rho_a = 1e-2
    rho_g = 1e-0

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
    legacy_size_prob = prob.legacy_size_prob()

    err = legacy @ x_sol - x
    path_error = (
        np.linalg.norm(legacy_size_prob.Xs @ legacy_size_prob.NV @ err) ** 2
        + np.linalg.norm(legacy_size_prob.Xs @ legacy_size_prob.NV @ err) ** 2
    )
    current_error = (
        np.linalg.norm(legacy_size_prob.EC @ err) ** 2
        + np.linalg.norm(legacy_size_prob.NC @ err) ** 2
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
        prob=prob,
    )

    legacy_size_prob = prob.legacy_size_prob()
    x_plot = legacy @ res_dict["x_sol"]
    x_true = legacy @ res_dict["x_true"]
    perror = res_dict["path_error"]
    cerror = res_dict["current_error"]

    print("Navigation position error:", perror)
    print("Current error:", cerror)
    viz.plot_bundle(
        x_plot,
        legacy_size_prob.adat,
        legacy_size_prob.ddat,
        legacy_size_prob.times,
        legacy_size_prob.depths,
        x_true,
    )
    c1, c2, c3, c4 = viz.check_condition(prob)
    viz.print_condition(c1, c2, c3, c4)


if __name__ == "__main__":
    main()
