import math
from typing import Tuple
import random

import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt

from adcp import optimization as op
from adcp import viz


def show_errmap(
    errmap, index: int = 0, rho_vs: list = [], rho_cs: list = [], norm=None
) -> None:
    fig = plt.figure()
    ax = fig.gca()  # add_axes([.05,.05,.85,.9])
    if norm is None:
        norm = colors.LogNorm()
    im = ax.imshow(errmap[index, :, :], norm=norm)
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


def plot_bundle(sol_x, prob, times, depths, x):
    viz.vehicle_speed_plot(
        sol_x, prob.ddat, times, depths, direction="both", x_true=x, ttw=False
    )
    viz.inferred_ttw_error_plot(sol_x, prob, direction="both", x_true=x)
    viz.current_depth_plot(
        sol_x,
        prob.adat,
        prob.ddat,
        direction="both",
        x_true=x,
        prob=prob,
        adcp=True,
    )
    viz.inferred_adcp_error_plot(
        sol_x, prob.adat, prob.ddat, direction="both", x_true=x
    )
    viz.vehicle_posit_plot(
        sol_x, prob.ddat, times, depths, x_true=x, dead_reckon=True
    )


def check_condition(prob: op.GliderProblem) -> Tuple:
    """Checks condition on matrices for glider problem"""
    m = len(prob.times)
    n = len(prob.depths)
    kalman_mat = op.gen_kalman_mat(prob)
    A, _ = op.solve_mats(prob)

    r100 = np.array(random.sample(range(0, 4 * m + 2 * n), 100))
    r1000 = np.array(random.sample(range(0, 4 * m + 2 * n), 1000))
    c1 = np.linalg.cond(kalman_mat[r100[:, None], r100].todense())
    c2 = np.linalg.cond(A[r100[:, None], r100].todense())
    c3 = np.linalg.cond(kalman_mat[r1000[:, None], r1000].todense())
    c4 = np.linalg.cond(A[r1000[:, None], r1000].todense())

    return c1, c2, c3, c4
