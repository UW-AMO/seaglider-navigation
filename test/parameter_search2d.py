# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 11:04:36 2020

@author: 600301
"""
from itertools import product, repeat

import numpy as np
from matplotlib import colors

from adcp import dataprep as dp
from adcp import simulation as sim
from adcp import matbuilder as mb
from adcp import optimization as op
from adcp.psearch import check_condition, plot_bundle, show_errmap

# %% ...or simulate new data
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
depths = dp.depthpoints(adat, ddat)
times = dp.timepoints(adat, ddat)

# %% No Range
rho_c = 1e0
rho_v = 1e0
rho_vs = rho_v * np.logspace(-10, -1, 11)
rho_cs = rho_c * np.logspace(-10, -1, 11)
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

# Channel 1: position error or current error
errmap = np.zeros((2, len(rho_vs), len(rho_cs)))
paths = list(repeat(list(repeat(None, len(rho_cs))), len(rho_vs)))

for ((i, rv), (j, rc)) in product(enumerate(rho_vs), enumerate(rho_cs)):
    prob = op.GliderProblem(
        ddat=ddat,
        adat=adat,
        rho_v=rv,
        rho_c=rc,
        rho_g=rho_g,
        rho_t=rho_t,
        rho_a=rho_a,
        rho_r=rho_r,
        t_scale=1e3,
        conditioner="tanh",
        vehicle_vel="otg",
        current_order=2,
        vehicle_order=2,
    )

    print(i, " ", j)
    # %%  Solve problem
    seed = 3453
    x_sol = op.backsolve(prob)

    f = op.f(prob, verbose=True)
    g = op.g(prob)
    h = op.h(prob)

    legacy = mb.legacy_select(
        prob.m,
        prob.n,
        prob.vehicle_order,
        prob.current_order,
        prob.vehicle_vel,
        prob=prob,
    )
    Xs = prob.Xs
    EV = prob.EV
    NV = prob.NV
    EC = prob.EC
    NC = prob.NC
    CV = prob.CV

    x_leg = legacy @ x_sol

    leg_Xs = mb.x_select(len(prob.times), 2)
    leg_NV = mb.nv_select(len(prob.times), len(prob.depths), 2, 2, "otg")
    leg_EV = mb.ev_select(len(prob.times), len(prob.depths), 2, 2, "otg")
    leg_NC = mb.nc_select(len(prob.times), len(prob.depths), 2, 2, "otg")
    leg_EC = mb.ec_select(len(prob.times), len(prob.depths), 2, 2, "otg")

    err = x_leg - x
    path_error = (
        np.linalg.norm(leg_Xs @ leg_NV @ err) ** 2
        + np.linalg.norm(leg_Xs @ leg_EV @ err) ** 2
    )
    current_error = (
        np.linalg.norm(leg_EC @ err) ** 2 + np.linalg.norm(leg_NC @ err) ** 2
    )
    errmap[0, i, j] = path_error
    errmap[1, i, j] = current_error
    paths[i][j] = x_leg
    # print(f"Scenario: rho_v = {rv} and rho_c = {rc}")
    # print(f"\tPath error: {path_error}")
    # print(f"\tCurrent error: {current_error}\n")


show_errmap(errmap, 0, rho_vs, rho_cs, norm=colors.LogNorm(vmin=1e7, vmax=2e9))
show_errmap(errmap, 1, rho_vs, rho_cs, norm=colors.LogNorm(vmin=1e0, vmax=1e3))
# %%
i1, j1 = np.unravel_index(errmap[0, :, :].argmin(), (len(rho_vs), len(rho_cs)))
nav_x = paths[i1][j1]
i2, j2 = np.unravel_index(errmap[1, :, :].argmin(), (len(rho_vs), len(rho_cs)))
curr_x = paths[i2][j2]


print(f"Best Navigational Solution: rho_v={rho_vs[i1]}, rho_c={rho_cs[j1]}")
print(
    f"""For measurement errors:
    vehicle process: {sim_rho_v}
    current process:{sim_rho_c}
    GPS measurement: {rho_g}
    TTW measurement: {rho_t}
    ADCP meawsurement: {rho_a}
Creates path error: {errmap[0, i1, j1]}
and current error: {errmap[1, i1, j1]}"""
)
prob = op.GliderProblem(
    ddat=ddat,
    adat=adat,
    rho_v=rho_vs[i2],
    rho_c=rho_cs[j2],
    rho_g=rho_g,
    rho_t=rho_t,
    rho_a=rho_a,
    rho_r=rho_r,
)
try:
    c1, c2, c3, c4 = check_condition(prob)
    print("100x100 sample of kalman matrix has condition number ", c1)
    print("100x100 sample of backsolve matrix has condition number ", c2)
    print("1000x1000 sample of kalman matrix has condition number ", c3)
    print("1000x1000 sample of backsolve matrix has condition number ", c4)
except ValueError:
    print("small matrices")

plot_bundle(nav_x, prob, times, depths, x)
if (i1 != i2) or (j1 != j2):
    print(f"Best Current Solution: rho_v={rho_vs[i2]}, rho_c={rho_cs[j2]}")
    plot_bundle(curr_x, prob, times, depths, x)
    prob = op.GliderProblem(
        ddat=ddat,
        adat=adat,
        rho_v=rho_vs[i2],
        rho_c=rho_cs[j2],
        rho_g=rho_g,
        rho_t=rho_t,
        rho_a=rho_a,
        rho_r=rho_r,
    )
    try:
        c1, c2, c3, c4 = check_condition(prob)
        print("100x100 sample of kalman matrix has condition number ", c1)
        print("100x100 sample of backsolve matrix has condition number ", c2)
        print("1000x1000 sample of kalman matrix has condition number ", c3)
        print("1000x1000 sample of backsolve matrix has condition number ", c4)
    except ValueError:
        print("small matrices")

else:
    print("... and it's also the best current solution")
