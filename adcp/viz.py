# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 18:44:43 2020

@author: 600301
"""
from typing import Tuple
import math
import random

from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import scipy

from adcp import dataprep as dp
from adcp import matbuilder as mb
from adcp import optimization as op
import adcp

cmap = plt.get_cmap("tab10")


# %%
def inferred_adcp_error_plot(
    solx, adat, ddat, direction="north", x_true=None, x_sol=None
):
    if direction.lower() == "both":
        plt.figure(figsize=[12, 6])
        plt.subplot(1, 2, 1)
        ax1 = inferred_adcp_error_plot(
            solx, adat, ddat, direction="north", x_true=x_true, x_sol=x_sol
        )
        plt.subplot(1, 2, 2)
        ax2 = inferred_adcp_error_plot(
            solx, adat, ddat, direction="east", x_true=x_true, x_sol=x_sol
        )
        return ax1, ax2

    ax = plt.gca()
    ax.set_title(direction.title() + " Shear velocities")
    ax.set_xlabel("meters/second")
    ax.set_ylabel("depth")
    times = dp.timepoints(adat, ddat)
    depths = dp.depthpoints(adat, ddat)
    m = len(times)
    n = len(depths)
    Vs = mb.v_select(m)
    if direction.lower() in {"north", "south"}:
        zadcp = mb.get_zadcp(adat, "north") / mb.t_scale
        XV = mb.nv_select(m, n)
        XC = mb.nc_select(m, n)
    elif direction.lower() in {"east", "west"}:
        zadcp = mb.get_zadcp(adat, "east") / mb.t_scale
        XV = mb.ev_select(m, n)
        XC = mb.ec_select(m, n)
    else:
        raise ValueError

    depth_df = dp._depth_interpolator(times, ddat)
    turnaround = depth_df.ascending.idxmax()
    deepest = depth_df.loc[turnaround, "depth"]

    A, B = mb.adcp_select(times, depths, ddat, adat)
    adcp_order = (B @ range(B.shape[1])).astype(int)
    adcp_depths = depths[adcp_order]
    sinking_inds = [(adcp_depths < deepest) & (adcp_depths > 0)]
    rising_inds = [(adcp_depths < deepest * 2) & (adcp_depths > deepest)]
    sinking_meas = zadcp[tuple(sinking_inds)]
    sinking_depths = adcp_depths[tuple(sinking_inds)]

    rising_meas = zadcp[tuple(rising_inds)]
    rising_depths = adcp_depths[tuple(rising_inds)]
    ln0 = ax.plot(
        sinking_meas,
        sinking_depths,
        ":",
        color="gold",
        label="Descending-measured",
    )
    ln1 = ax.plot(
        rising_meas,
        2 * deepest - rising_depths,
        ":",
        color="purple",
        label="Ascending-measured",
    )
    lines = [ln0, ln1]

    adcp_lbfgs = (B @ XC - A @ Vs @ XV) @ solx
    sinking_lbfgs = adcp_lbfgs[tuple(sinking_inds)]
    rising_lbfgs = adcp_lbfgs[tuple(rising_inds)]
    ln2 = ax.plot(
        sinking_lbfgs,
        sinking_depths,
        "--",
        color="deeppink",
        label="Descending-LBFGS",
    )
    ln3 = ax.plot(
        rising_lbfgs,
        2 * deepest - rising_depths,
        "--",
        color="chartreuse",
        label="Ascending-LBFGS",
    )

    lines = [*lines, ln2, ln3]

    if x_true is not None:
        adcp_true = (B @ XC - A @ Vs @ XV) @ x_true
        sinking_true = adcp_true[tuple(sinking_inds)]
        rising_true = adcp_true[tuple(rising_inds)]
        ln4 = ax.plot(
            sinking_true,
            sinking_depths,
            "-",
            color="maroon",
            label="Descending-true",
        )
        ln5 = ax.plot(
            rising_true,
            2 * deepest - rising_depths,
            "-",
            color="teal",
            label="Ascending-true",
        )
        lines = [*lines, ln4, ln5]

    if x_sol is not None:
        adcp_back = (B @ XC - A @ Vs @ XV) @ x_sol
        sinking_back = adcp_back[tuple(sinking_inds)]
        rising_back = adcp_back[tuple(rising_inds)]
        ln6 = ax.plot(
            sinking_back,
            sinking_depths,
            "--",
            color="xkcd:pumpkin",
            label="Descending-baksolve",
        )
        ln7 = ax.plot(
            rising_back,
            2 * deepest - rising_depths,
            "--",
            color="blue",
            label="Ascending-backsolve",
        )
        lines = [*lines, ln6, ln7]

    ax.legend()
    return ax


def inferred_ttw_error_plot(
    solx, adat, ddat, direction="north", x_true=None, x_sol=None
):

    if direction.lower() == "both":
        plt.figure(figsize=[12, 6])
        plt.subplot(1, 2, 1)
        ax1 = inferred_ttw_error_plot(
            solx, adat, ddat, direction="north", x_true=x_true, x_sol=x_sol
        )
        plt.subplot(1, 2, 2)
        ax2 = inferred_ttw_error_plot(
            solx, adat, ddat, direction="east", x_true=x_true, x_sol=x_sol
        )
        return ax1, ax2

    ax = plt.gca()
    ax.set_title(direction.title() + " TTW velocities")
    ax.set_xlabel("Time")
    ax.set_ylabel("meters/second")
    times = dp.timepoints(adat, ddat)
    depths = dp.depthpoints(adat, ddat)
    m = len(times)
    n = len(depths)
    Vs = mb.v_select(m)
    if direction.lower() in {"north", "south"}:
        zttw = mb.get_zttw(ddat, "north", t_scale=1)
        XV = mb.nv_select(m, n)
        XC = mb.nc_select(m, n)
    elif direction.lower() in {"east", "west"}:
        zttw = mb.get_zttw(ddat, "east", t_scale=1)
        XV = mb.ev_select(m, n)
        XC = mb.ec_select(m, n)
    else:
        raise ValueError

    A, B = mb.uv_select(times, depths, ddat)

    ln0 = ax.plot(
        zttw.index, zttw.values, ":", color=cmap(3), label="TTW measured"
    )

    ttw_lbfgs = (A @ Vs @ XV - B @ XC) @ solx
    ln1 = ax.plot(zttw.index, ttw_lbfgs, "--", color=cmap(0), label="LBFGS")

    lines = [ln0, ln1]

    if x_true is not None:
        ttw_true = (A @ Vs @ XV - B @ XC) @ x_true
        ln2 = ax.plot(zttw.index, ttw_true, "-", color=cmap(1), label="True")
        lines = [*lines, ln2]

    if x_sol is not None:
        ttw_back = (A @ Vs @ XV - B @ XC) @ x_sol
        ln3 = ax.plot(
            zttw.index, ttw_back, "--", color=cmap(2), label="Backsolve"
        )
        lines = [*lines, ln3]

    ax.legend()
    return ax


# %%
def current_depth_plot(
    solx,
    adat,
    ddat,
    direction="north",
    x_true=None,
    x_sol=None,
    mdat=None,
    adcp=False,
    prob=None,
):
    """Produce a current by depth plot, showing the current during
    descending and ascending measurements.  Various other options.

    Parameters:
        solx (numpy.array): LBFGS solution for state vector
        adat (dict): ADCP data. See dataprep.load_adcp() or
            simulation.construct_load_dicts()
        ddat (dict): dive data. See dataprep.load_dive() or
            simulation.construct_load_dicts()
        direction (str): either 'north' or 'south'
        x_true (numpy.array): true state vector
        x_sol (numpy.array): backsolve solution for state vector
        mdat (dict): Moored ADCP measurements.  See
            dataprep.load_mooring()
        adcp (bool): Whether to include ADCP measurement data or not
        prob (GliderProblem): A problem set that bundles adat, ddat, and more
    """

    if direction.lower() == "both":
        plt.figure(figsize=[12, 6])
        plt.subplot(1, 2, 1)
        ax1 = current_depth_plot(
            solx,
            adat,
            ddat,
            direction="north",
            x_true=x_true,
            x_sol=x_sol,
            mdat=mdat,
            adcp=adcp,
            prob=prob,
        )
        plt.subplot(1, 2, 2)
        ax2 = current_depth_plot(
            solx,
            adat,
            ddat,
            direction="east",
            x_true=x_true,
            x_sol=x_sol,
            mdat=mdat,
            adcp=adcp,
            prob=prob,
        )
        return ax1, ax2
    ax = plt.gca()
    times = dp.timepoints(adat, ddat)
    depths = dp.depthpoints(adat, ddat)
    m = len(times)
    n = len(depths)
    if direction.lower() in {"north", "south"}:
        currs = mb.nc_select(m, n) @ solx
    elif direction.lower() in {"east", "west"}:
        currs = mb.ec_select(m, n) @ solx
    depth_df = dp._depth_interpolator(times, ddat)
    turnaround = depth_df.ascending.idxmax()
    deepest = depth_df.loc[turnaround, "depth"]
    sinking = currs[(depths < deepest) & (depths > 0)]
    sinking_depths = depths[(depths < deepest) & (depths > 0)]
    rising = currs[(depths > deepest) & (depths < deepest * 2)]
    rising_depths = depths[(depths > deepest) & (depths < deepest * 2)]

    lines = []

    # ADCP traces
    if adcp:
        if direction.lower() in {"north", "south"}:
            zadcp = mb.get_zadcp(adat, "north") / mb.t_scale
            V = mb.nv_select(m, n)
        elif direction.lower() in {"east", "west"}:
            zadcp = mb.get_zadcp(adat, "east") / mb.t_scale
            V = mb.ev_select(m, n)
        A_adcp, B_adcp = mb.adcp_select(times, depths, ddat, adat)
        t_inds_with_adcp_obs = np.argwhere(
            np.asarray(A_adcp.sum(axis=0)).squeeze()
        )
        Vs = mb.v_select(m)
        A_adcp = scipy.sparse.csc_matrix(A_adcp)
        shifted_adcp_trace = A_adcp @ Vs @ V @ solx + zadcp
        # what about ADCP traces above the water surface (<0 or >deepest)
        for i in t_inds_with_adcp_obs:
            rows_in_trace = np.argwhere(A_adcp[:, i])[:, 0]
            adcp_vals = shifted_adcp_trace[rows_in_trace]
            depth_ind_in_trace = np.argwhere(B_adcp[rows_in_trace, :])[:, 1]
            depth_vals = depths[depth_ind_in_trace]
            depth_vals[depth_vals > deepest] = (
                2 * deepest - depth_vals[depth_vals > deepest]
            )
            ln_ = ax.plot(
                adcp_vals,
                depth_vals,
                "--",
                color="purple",
            )
            lines.append(ln_)

    ln0 = ax.plot(
        sinking,
        sinking_depths,
        "--",
        color="deeppink",
        label="Descending-LBFGS",
    )
    ln1 = ax.plot(
        rising,
        2 * deepest - rising_depths,
        "--",
        color="chartreuse",
        label="Ascending-LBFGS",
    )
    lines.append(ln0)
    lines.append(ln1)

    # Add in true simulated profiles, if available
    if x_true is not None:
        if direction.lower() in {"north", "south"}:
            true_currs = mb.nc_select(m, n) @ x_true
        elif direction.lower() in {"east", "west"}:
            true_currs = mb.ec_select(m, n) @ x_true
        sinking_true = true_currs[(depths < deepest) & (depths > 0)]
        rising_true = true_currs[(depths > deepest) & (depths < deepest * 2)]
        ln2 = ax.plot(
            sinking_true,
            sinking_depths,
            "-",
            color="maroon",
            label="Descending-True",
        )
        ln3 = ax.plot(
            rising_true,
            2 * deepest - rising_depths,
            "-",
            color="teal",
            label="Ascending-True",
        )
        lines = [*lines, ln2, ln3]
    # Add in backsolve solution, if available
    if x_sol is not None:
        if direction.lower() in {"north", "south"}:
            true_currs = mb.nc_select(m, n) @ x_sol
        elif direction.lower() in {"east", "west"}:
            true_currs = mb.ec_select(m, n) @ x_sol
        sinking_true = true_currs[(depths < deepest) & (depths > 0)]
        rising_true = true_currs[(depths > deepest) & (depths < deepest * 2)]
        ln4 = ax.plot(
            sinking_true,
            sinking_depths,
            "--",
            color="xkcd:pumpkin",
            label="Descending-Backsolve",
        )
        ln5 = ax.plot(
            rising_true,
            2 * deepest - rising_depths,
            "--",
            color="blue",
            label="Ascending-Backsolve",
        )
        lines = [*lines, ln4, ln5]

    # Preprocess to get mooring data, if necessary:
    if mdat is not None:
        first_time = ddat["depth"].index.min()
        last_time = ddat["depth"].index.max()
        first_truth_idx = np.argmin(np.abs(first_time - mdat["time"]))
        last_truth_idx = np.argmin(np.abs(last_time - mdat["time"]))
        true_currs = mdat["u"] if direction.lower() == "north" else mdat["v"]
        ln6 = ax.plot(
            true_currs[first_truth_idx, :],
            mdat["depth"][first_truth_idx, :],
            "-",
            color="coral",
            label="Descending-Mooring",
        )
        ln7 = ax.plot(
            true_currs[last_truth_idx, :],
            mdat["depth"][last_truth_idx, :],
            "-",
            color="cyan",
            label="Ascending-Mooring",
        )
        #        ax.legend(loc='lower left')
        lines = [*lines, ln6, ln7]

    ax.legend()
    ax.invert_yaxis()
    font_dict = {"size": "medium"}
    ax.set_title(direction.title() + "erly Current", **font_dict)
    ax.set_xlabel("Current (meters/sec)".title(), **font_dict)
    ax.set_ylabel("Depth (m); Descending=warm, Ascending=cool", **font_dict)
    plt.tight_layout()

    # Adjust ylim if just plotting surface
    if mdat is not None:
        max_depth = max(
            *mdat["depth"][first_truth_idx, :],
            *mdat["depth"][last_truth_idx, :],
        )
        ax.set_ylim(max_depth, 0)

    return lines


# %%
def vehicle_speed_plot(
    solx,
    ddat,
    times,
    depths,
    direction="north",
    x_sol=None,
    x_true=None,
    x0=None,
    ttw=False,
):
    """Plots the vehicle's solved speed, optionally with different
    comparison solutions.

    Parameters:
        solx (numpy.array): LBFGS solution for state vector
        ddat (dict): dive data. See dataprep.load_dive() or
            simulation.construct_load_dicts()
        times (numpy.array): times at which a measurement occured
        depths (numpy.array): depths at which a measurement occured
        direction (str): either 'north' or 'south'
        x_sol (numpy.array): backsolve solution for state vector
        x_true (numpy.array): true state vector
        x0 (numpy.array): starting state vector for LBFGS
        ttw (bool): Whether to include measured TTW values
    """
    if direction.lower() == "both":
        plt.figure(figsize=[12, 6])
        plt.subplot(1, 2, 1)
        ax1 = vehicle_speed_plot(
            solx,
            ddat,
            times,
            depths,
            direction="north",
            x_true=x_true,
            x_sol=x_sol,
            x0=x0,
        )
        plt.subplot(1, 2, 2)
        ax2 = vehicle_speed_plot(
            solx,
            ddat,
            times,
            depths,
            direction="east",
            x_true=x_true,
            x_sol=x_sol,
            x0=x0,
        )
        return ax1, ax2

    ax = plt.gca()
    m = len(times)
    n = len(depths)
    dirV = (
        mb.nv_select(m, n)
        if direction.lower() == "north"
        else mb.ev_select(m, n)
    )
    Vs = mb.v_select(m)
    cmap = plt.get_cmap("tab10")
    font_dict = {"size": "medium"}
    ax.set_title(f"{direction}ward Vehicle Velocity".title(), **font_dict)
    ln1 = ax.plot(
        times, Vs @ dirV @ solx, "--", color=cmap(0), label="LBFGS Votg"
    )
    lns = [ln1[0]]
    if x_sol is not None:
        ln2 = ax.plot(
            times,
            Vs @ dirV @ x_sol,
            "--",
            color=cmap(2),
            label="backsolve Votg",
        )
        lns.append(ln2[0])
    if x0 is not None:
        ln3 = ax.plot(
            times,
            Vs @ dirV @ x0,
            ":",
            color=cmap(4),
            label="x0 Votg",
        )
        lns.append(ln3[0])
    if x_true is not None:
        ln4 = ax.plot(
            times,
            Vs @ dirV @ x_true,
            "-",
            color=cmap(1),
            label="Votg_true",
        )
        lns.append(ln4[0])
    if ttw:
        ln5 = ax.plot(
            mb.get_zttw(ddat).index,
            mb.get_zttw(ddat).values / 1e3,
            ":",
            color=cmap(3),
            label="TTW measured",
        )
        lns.append(ln5[0])
    ax.set_ylabel("meters/second", **font_dict)
    ax.set_xlabel("time", **font_dict)
    labs = [line.get_label() for line in lns]
    ax.legend(lns, labs)
    return ax


def current_plot(solx, x_sol, adat, times, depths, direction="north"):
    """Deprecated"""
    fig = plt.figure()
    m = len(times)
    n = len(depths)
    dirC = (
        mb.nc_select(m, n)
        if direction.lower() == "north"
        else mb.ec_select(m, n)
    )
    ax = fig.gca()
    cmap = ax.get_cmap("tab10")
    ax.set_title(f"{direction}ward Current".title())
    ax.plot(dirC @ x_sol, color=cmap(0), label="backsolve")
    ax.plot(dirC @ solx, color=cmap(1), label="LBFGS")
    ax.legend()
    ax.twiny()
    ax.plot(mb.get_zadcp(adat) / 1e3, color=cmap(3), label="z_ttw")
    return ax


def vehicle_posit_plot(
    x,
    ddat,
    times,
    depths,
    x0=None,
    backsolve=None,
    x_true=None,
    dead_reckon=True,
):
    """Plots the vehicle's position in x-y coordinates, optionally
    with different comparison solutions.

         Parameters:
         x (numpy.array): LBFGS solution for state vector
         ddat (dict): dive data. See dataprep.load_dive() or
             simulation.construct_load_dicts()
         times (numpy.array): times at which a measurement occured
         depths (numpy.array): depths at which a measurement occured
         x0 (numpy.array): starting state vector for LBFGS
         backsolve (numpy.array): backsolve solution for state vector
         x_true (numpy.array): true state vector
         dead_reckon (bool): Whether to include dead reckoning solution,
             treating TTW measurements as over-the-ground truth.
    """
    m = len(times)
    n = len(depths)
    NV = mb.nv_select(m, n)
    EV = mb.ev_select(m, n)
    Xs = mb.x_select(m)
    plt.figure()
    ax = plt.gca()
    ax.set_title("Vehicle Position")
    ln1 = ax.plot(
        Xs @ EV @ x / 1000,
        Xs @ NV @ x / 1000,
        "--",
        color=cmap(0),
        label="Optimization Solution",
    )
    lns = [ln1[0]]
    if backsolve is not None:
        ln2 = ax.plot(
            Xs @ EV @ backsolve / 1000,
            Xs @ NV @ backsolve / 1000,
            "--",
            color=cmap(2),
            label="backsolve",
        )
        lns.append(ln2[0])
    if x0 is not None:
        ln3 = ax.plot(
            Xs @ EV @ x0 / 1000,
            Xs @ NV @ x0 / 1000,
            ":",
            color=cmap(3),
            label="x0",
        )
        lns.append(ln3[0])
    if x_true is not None:
        ln4 = ax.plot(
            Xs @ EV @ x_true / 1000,
            Xs @ NV @ x_true / 1000,
            "-",
            color=cmap(1),
            label="X_true",
        )
        lns.append(ln4[0])
    if dead_reckon:
        df = dp.dead_reckon(ddat)
        ln5 = ax.plot(
            df.x_dead / 1000,
            df.y_dead / 1000,
            ":",
            color=cmap(4),
            label="Dead Reckoning",
        )
        ln6 = ax.plot(
            df.x_corr / 1000,
            df.y_corr / 1000,
            "--",
            color=cmap(5),
            label="Reckoning, Corrected",
        )
        lns.append(ln5[0])
        lns.append(ln6[0])
    ax.legend()
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    return ax


def plot_bundle(sol_x, adat, ddat, times, depths, x):
    vehicle_speed_plot(
        sol_x, ddat, times, depths, direction="both", x_true=x, ttw=False
    )
    inferred_ttw_error_plot(sol_x, adat, ddat, direction="both", x_true=x)
    current_depth_plot(
        sol_x,
        adat,
        ddat,
        direction="both",
        x_true=x,
        adcp=True,
    )
    inferred_adcp_error_plot(sol_x, adat, ddat, direction="both", x_true=x)
    vehicle_posit_plot(sol_x, ddat, times, depths, x_true=x, dead_reckon=True)


def display_uncertainty(AtAinv, A, rows):
    solution_variance_plot(AtAinv, rows)
    influence_plot(AtAinv, A, rows)


def solution_variance_plot(AtAinv, rows):
    plt.figure()
    ax = plt.gca()
    ax.matshow(AtAinv.todense()[rows, rows])
    # plt.figure()
    # ax = plt.gca()
    # ax.plot(np.diagonal(AtAinv.todense())[rows])


def influence_plot(AtA, A, rows):
    pass


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


def check_condition(prob: adcp.GliderProblem) -> Tuple:
    """Checks condition on matrices for glider problem"""
    m = len(prob.times)
    n = len(prob.depths)
    kalman_mat = op.gen_kalman_mat(
        prob.data, prob.config, prob.shape, prob.weights
    )
    A, _ = op.solve_mats(prob)

    r100 = np.array(random.sample(range(0, 4 * m + 2 * n), 100))
    r1000 = np.array(random.sample(range(0, 4 * m + 2 * n), 1000))
    c1 = np.linalg.cond(kalman_mat[r100[:, None], r100].todense())
    c2 = np.linalg.cond(A[r100[:, None], r100].todense())
    c3 = np.linalg.cond(kalman_mat[r1000[:, None], r1000].todense())
    c4 = np.linalg.cond(A[r1000[:, None], r1000].todense())

    return c1, c2, c3, c4


def print_condition(c1: int, c2: int, c3: int, c4: int):
    print(f"100x100 sample of kalman matrix has condition {c1:.2E}")
    print(f"100x100 sample of A has condition {c2:.2E}")
    print(f"1000x1000 sample of kalman matrix has condition {c3:.2E}")
    print(f"1000x1000 sample of A has condition {c4:.2E}")
