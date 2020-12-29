# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 11:04:36 2020

@author: 600301
"""
# %% Imports
import math
from itertools import product, repeat
from typing import Tuple, List
import random

import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt

from adcp import dataprep as dp
from adcp import simulation as sim
from adcp import matbuilder as mb
from adcp import optimization as op
from adcp import viz

# %% Simulate new data
rho_t = 1e-1
rho_a = 1e-1
rho_g = 1e1

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
    measure_points=dict(gps="endpoints", ttw=0.5, range=0.05),
    vehicle_method="curved",
    curr_method="curved",
)
ddat, adat, x, curr_df, v_df = sim.simulate(sp, verbose=True)
depths = dp.depthpoints(adat, ddat)
times = dp.timepoints(adat, ddat)

# %% No Range
rho_c = 1e-9
rho_v = 1e4
n_steps = 25
v_factors = np.logspace(-11, 0, n_steps)
c_factors = np.logspace(0, 0, n_steps)
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
errmap = np.zeros((2, len(c_factors)))
paths = []
plt.figure()
for i, (v_factor, c_factor) in enumerate(zip(v_factors, c_factors)):
    rc = c_factor * rho_c
    rv = v_factor * rho_v
    prob = op.GliderProblem(
        ddat,
        adat,
        rho_v=rv,
        rho_c=rc,
        rho_g=rho_g,
        rho_t=rho_t,
        rho_a=rho_a,
        rho_r=rho_r,
    )
    # %%  Solve problem
    print(f"solving for rho_v={rv} and rho_c={rc}.")
    seed = 3453
    x_sol, (NV, EV, NC, EC, Xs, Vs) = op.backsolve(prob)

    f = op.f(prob, verbose=True)
    g = op.g(prob)
    h = op.h(prob)
    # %% Plotting
    err = x_sol - x
    path_error = (
        np.linalg.norm(Xs @ NV @ err) ** 2 + np.linalg.norm(Xs @ EV @ err) ** 2
    )
    current_error = (
        np.linalg.norm(EC @ err) ** 2 + np.linalg.norm(NC @ err) ** 2
    )
    errmap[0, i] = path_error
    errmap[1, i] = current_error
    paths.append(x_sol)

# %%
def show_errmap(rho_vs: float, rho_cs: float) -> None:
    fig = plt.figure()
    ax1 = fig.gca()
    n = len(rho_cs)
    labels = list(
        map(lambda tup: f"{tup[0]:.1e}\n{tup[1]:.1e}", zip(rho_vs, rho_cs))
    )
    ln1 = ax1.semilogy(labels, errmap[0, :], c="blue", label="Position Error")
    ax2 = ax1.twinx()
    ln2 = ax2.semilogy(labels, errmap[1, :], c="red", label="Current Error")
    ax1.set_title(f"Error for 1_D rho_v & rho_c search")
    ax1.set_ylabel("Position Error")
    ax2.set_ylabel("Current Error")
    ax1.set_xlabel("rho_v\n rho_c")
    ax1.set_xticks(ax1.get_xticks()[::4])
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)


show_errmap(rho_v * v_factors, rho_c * c_factors)
# %%
f_nav, f_curr = tuple(errmap.argmin(axis=1))
nav_x = paths[f_nav]
curr_x = paths[f_curr]


def plot_bundle(sol_x):
    ax1 = viz.vehicle_speed_plot(
        sol_x, ddat, times, depths, direction="both", x_true=x, ttw=False
    )
    ax2 = viz.inferred_ttw_error_plot(
        sol_x, adat, ddat, direction="both", x_true=x
    )
    ax3 = viz.current_depth_plot(sol_x, adat, ddat, direction="both", x_true=x)
    ax4 = viz.inferred_adcp_error_plot(
        sol_x, adat, ddat, direction="both", x_true=x
    )
    ax5 = viz.vehicle_posit_plot(
        sol_x, ddat, times, depths, x_true=x, dead_reckon=True
    )


print(
    f"Best Navigational Solution: rho_v={rho_v*v_factors[f_nav]},"
    f" rho_c={rho_c*c_factors[f_nav]}"
)
print(
    f"""For measurement errors:
    vehicle process: {sim_rho_v}
    current process:{sim_rho_c}
    GPS measurement: {rho_g}
    TTW measurement: {rho_t}
    ADCP meawsurement: {rho_a}
Path error: {errmap[0, f_nav]}
Current error: {errmap[1, f_nav]}"""
)

plot_bundle(nav_x)
if f_nav != f_curr:
    print(
        f"Best Current Solution: rho_v={rho_v*v_factors[f_curr]},"
        f" rho_c={rho_c*c_factors[f_curr]}"
    )
    print(
        f"Path error: {errmap[0, f_curr]}\n"
        f"Current error: {errmap[1, f_curr]}"
    )
    plot_bundle(curr_x)
else:
    print("... and it's also the best current solution")
# %%
