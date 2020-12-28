# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 11:04:36 2020

@author: 600301
"""
import math
from itertools import product, repeat
from typing import Tuple
import random

import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt

from adcp import dataprep as dp
from adcp import simulation as sim
from adcp import matbuilder as mb
from adcp import optimization as op
from adcp import viz

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
    measure_points=dict(gps="endpoints", ttw=0.5, range=0.05),
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

# reduce to old-size x

for ((i, rv), (j, rc)) in product(enumerate(rho_vs), enumerate(rho_cs)):
    prob = op.GliderProblem(
        ddat,
        adat,
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
    )
    Xs = prob.Xs
    EV = prob.EV
    NV = prob.NV
    EC = prob.EC
    NC = prob.NC
    CV = prob.CV

    x_sol = legacy @ x_sol

    err = x_sol - x
    path_error = (
        np.linalg.norm(Xs @ NV @ err) ** 2 + np.linalg.norm(Xs @ EV @ err) ** 2
    )
    current_error = (
        np.linalg.norm(EC @ err) ** 2 + np.linalg.norm(NC @ err) ** 2
    )
    errmap[0, i, j] = path_error
    errmap[1, i, j] = current_error
    paths[i][j] = x_sol
    # print(f"Scenario: rho_v = {rv} and rho_c = {rc}")
    # print(f"\tPath error: {path_error}")
    # print(f"\tCurrent error: {current_error}\n")

# %%
def show_errmap(index: int = 0, rho_vs: list = [], rho_cs: list = []) -> None:
    fig = plt.figure()
    ax = fig.gca()  # add_axes([.05,.05,.85,.9])
    im = ax.imshow(errmap[index, :, :], norm=colors.LogNorm())
    if index == 0:
        ax.set_title("Position Error")
    else:
        ax.set_title("Current Error")
    ax.set_yticks(range(len(rho_vs)))
    ax.set_xticks(range(len(rho_cs)))
    ylabels = np.round(np.log(rho_vs) / math.log(10)).astype(int)
    xlabels = np.round(np.log(rho_cs) / math.log(10)).astype(int)
    ax.set_yticklabels(ylabels)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("rho_v (log scale)")
    ax.set_xlabel("rho_c (log scale)")
    cax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
    ax.invert_yaxis()
    fig.colorbar(im, cax=cax)
    # plt.close()
    # return fig


show_errmap(0, rho_vs, rho_cs)
show_errmap(1, rho_vs, rho_cs)
# %%
i1, j1 = np.unravel_index(errmap[0, :, :].argmin(), (len(rho_vs), len(rho_cs)))
nav_x = paths[i1][j1]
i2, j2 = np.unravel_index(errmap[1, :, :].argmin(), (len(rho_vs), len(rho_cs)))
curr_x = paths[i2][j2]


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
    ddat,
    adat,
    rho_v=rho_vs[i2],
    rho_c=rho_cs[j2],
    rho_g=rho_g,
    rho_t=rho_t,
    rho_a=rho_a,
    rho_r=rho_r,
)
c1, c2, c3, c4 = check_condition(prob)
print("100x100 sample of kalman matrix has condition number ", c1)
print("100x100 sample of backsolve matrix has condition number ", c2)
print("1000x1000 sample of kalman matrix has condition number ", c3)
print("1000x1000 sample of backsolve matrix has condition number ", c4)


plot_bundle(nav_x)
if (i1 != i2) or (j1 != j2):
    print(f"Best Current Solution: rho_v={rho_vs[i2]}, rho_c={rho_cs[j2]}")
    plot_bundle(curr_x)
    prob = op.GliderProblem(
        ddat,
        adat,
        rho_v=rho_vs[i2],
        rho_c=rho_cs[j2],
        rho_g=rho_g,
        rho_t=rho_t,
        rho_a=rho_a,
        rho_r=rho_r,
    )
    c1, c2, c3, c4 = check_condition(prob)
    print("100x100 sample of kalman matrix has condition number ", c1)
    print("100x100 sample of backsolve matrix has condition number ", c2)
    print("1000x1000 sample of kalman matrix has condition number ", c3)
    print("1000x1000 sample of backsolve matrix has condition number ", c4)

else:
    print("... and it's also the best current solution")
# %%
