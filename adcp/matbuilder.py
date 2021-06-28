# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 21:25:07 2019

@author: 600301
"""
import warnings
from itertools import repeat
from typing import Iterable

import scipy.sparse
import scipy.interpolate
import numpy as np
import pandas as pd

from . import dataprep as dp

t_scale = 1
"""timescale, 1=seconds, 1e-3=milliseconds, 1e3=kiloseconds.  Used to
control condition number of problem.
"""
conditioner = "tanh"


def vehicle_select(times, depths, ddat):
    """Creates the matrix that will select the appropriate current values
    for depths where the vehicle is present.  It is the matrix equivalent
    of dp._depth_interpolator()
    """
    vehicle_depths = dp._depth_interpolator(times, ddat)["depth"]
    idxdepth = np.array(
        [np.argwhere(depths == d) for d in vehicle_depths]
    ).flatten()
    mat_shape = (len(times), len(depths))
    B = scipy.sparse.csr_matrix(
        (np.ones(len(idxdepth)), (range(len(idxdepth)), idxdepth)),
        shape=mat_shape,
    )
    return B


# %% Vector item selection
def uv_select(times, depths, ddat, adat=None, vehicle_vel="otg"):
    """Creates the matrices that will select the appropriate V_otg
    and current values for comparing with hydrodynamic model
    uv measurements.

    Parameters:
        times ([numpy.datetime64,]) : all of the sample times to predict
            V_otg for.  returned by dataprep.timepoints()
        depths ([numpy.datetime64,]) : all of the sample depths to predict
            current for.  returned by dataprep.depthpoints()
        ddat (dict): the recorded dive data returned by load_dive()
        adat (dict): the recorded ADCP data returned by load_adcp()
        vehicle_vel (str): whether modeling "otg" or "ttw" vehicle velocity

    Returns:
        tuple of nupmy arrays.  The first multiplies the V_otg vector,
        the second multiplies the current vector
    """
    uv_times = ddat["uv"].index.unique().to_numpy()
    idxuv = [k for k, t in enumerate(times) if t in uv_times]
    mat_shape = (len(idxuv), len(times))
    A = scipy.sparse.coo_matrix(
        (np.ones(len(idxuv)), (range(len(idxuv)), idxuv)), shape=mat_shape
    )

    uv_depths = dp._depth_interpolator(times, ddat).loc[uv_times, "depth"]
    uv_depths = uv_depths.to_numpy().flatten()
    d_list = list(depths)
    idxdepth = [d_list.index(d) for d in uv_depths]
    mat_shape = (len(idxdepth), len(depths))
    if vehicle_vel[:3] == "otg":
        B = scipy.sparse.coo_matrix(
            (np.ones(len(idxdepth)), (range(len(idxdepth)), idxdepth)),
            shape=mat_shape,
        )
    elif vehicle_vel == "ttw":
        B = scipy.sparse.coo_matrix(mat_shape)
    return A, B


def adcp_select(times, depths, ddat, adat, vehicle_vel="otg"):
    """Creates the matrices that will select the appropriate V_otg
    and current values for comparing with ADCP measurements.

    Parameters:
        times ([numpy.datetime64,]) : all of the sample times to predict
            V_otg for.  returned by dataprep.timepoints()
        depths ([numpy.datetime64,]) : all of the sample depths to predict
            current for.  returned by dataprep.depthpoints()
        ddat (dict): the recorded dive data returned by load_dive()
        adat (dict): the recorded ADCP data returned by load_adcp()
        vehicle_vel (str): whether modeling "otg" or "ttw" vehicle velocity

    Returns:
        tuple of nupmy arrays.  The first multiplies the vehicle
        velocity vector, the second multiplies the current vector
    """
    adcp_times = np.unique(adat["time"])
    idxadcp = [k for k, t in enumerate(times) if t in adcp_times]
    valid_obs = np.isfinite(adat["UV"])
    obs_per_t = valid_obs.sum(axis=0)
    col_idx = [np.repeat(idx, n_obs) for idx, n_obs in zip(idxadcp, obs_per_t)]
    col_idx = [el for arr in col_idx for el in arr]  # unpack list o'lists
    mat_shape = (len(col_idx), len(times))
    A = scipy.sparse.coo_matrix(
        (np.ones(len(col_idx)), (range(len(col_idx)), col_idx)),
        shape=mat_shape,
    )

    depth_df = dp._depth_interpolator(times, ddat)
    turnaround = depth_df.ascending.idxmax()  # first true index
    deepest = depth_df.loc[turnaround, "depth"]
    rising_times = pd.to_datetime(adat["time"]) > turnaround
    adcp_depths = adat["Z"].copy()
    adcp_depths[:, rising_times] = 2 * deepest - adat["Z"][:, rising_times]

    d_list = list(depths)
    idxdepth = [d_list.index(d) for d in adcp_depths.T[valid_obs.T]]
    mat_shape = (len(idxdepth), len(depths))
    B1 = scipy.sparse.coo_matrix(
        (np.ones(len(idxdepth)), (range(len(idxdepth)), idxdepth)),
        shape=mat_shape,
    )

    if vehicle_vel == "ttw":
        adcp_vehicle_depths = depth_df.loc[adcp_times, "depth"].values
        adcp_vehicle_depths = np.repeat(
            adcp_vehicle_depths[:, np.newaxis], valid_obs.shape[0], axis=1
        ).T
        idx_v_depth = [
            d_list.index(d) for d in adcp_vehicle_depths.T[valid_obs.T]
        ]
        B2 = scipy.sparse.coo_matrix(
            (-np.ones(len(idxdepth)), (range(len(idxdepth)), idx_v_depth)),
            shape=mat_shape,
        )
    else:
        B2 = scipy.sparse.coo_matrix(mat_shape)
    return A, B1 + B2


def gps_select(times, depths, ddat, adat, vehicle_vel="otg"):
    """Creates the matrix that will select the appropriate position
    for comparing with GPS measurements

    Parameters:
        times ([numpy.datetime64,]) : all of the sample times to predict
            V_otg for.  returned by dataprep.timepoints()
        depths ([numpy.datetime64,]) : all of the sample depths to predict
            current for.  returned by dataprep.depthpoints()
        ddat (dict): the recorded dive data returned by load_dive()
        adat (dict): the recorded ADCP data returned by load_adcp()
        vehicle_method (str): Whether the vehicle modeling describes
            over-the-ground (world referenced) or through-the-water
            (relative to water) metrics

    Returns:
        Tuple: sparse array to multiply the vehicle posit vector and one
        to multiply the current posit vector, if modeling ttw velocity
    """
    gps_times = ddat["gps"].index.unique().to_numpy()
    idxgps = [k for k, t in enumerate(times) if t in gps_times]
    mat_shape = (len(idxgps), len(times))
    A = scipy.sparse.coo_matrix(
        (np.ones(len(idxgps)), (range(len(idxgps)), idxgps)), shape=mat_shape
    )

    if vehicle_vel == "ttw":
        mat_shape = (len(idxgps), len(depths))
        d_list = list(depths)
        depth_df = dp._depth_interpolator(times, ddat)
        gps_depth = depth_df.loc[gps_times, "depth"].values
        idxdepth = [d_list.index(d) for d in gps_depth]

        B = scipy.sparse.coo_matrix(
            (np.ones(len(idxdepth)), (range(len(idxdepth)), idxdepth)),
            shape=mat_shape,
        )
    else:
        B = None

    return A, B


def range_select(times, ddat):
    """Creates the matrix that will select the appropriate position
    for comparing with range measurements

    Parameters:
        times ([numpy.datetime64,]) : all of the sample times to predict
            V_otg for.  returned by dataprep.timepoints()
        ddat (dict): the recorded dive data returned by load_dive()

    Returns:
        Nupmys arrays to multiply the X_otg vectors.
    """
    range_times = ddat["range"].index.unique().to_numpy()
    idxrange = [k for k, t in enumerate(times) if t in range_times]
    mat_shape = (len(idxrange), len(times))
    A = scipy.sparse.coo_matrix(
        (np.ones(len(idxrange)), (range(len(idxrange)), idxrange)),
        shape=mat_shape,
    )
    return A


# %% Kalman Process Matrices
class ConditionWarning(Warning):
    def __init__(self, cond):
        self.msg = f"The condition number of covariance matrix is high: {cond}"


def q_cond(dts, dim=2):
    """Calculates the condition number of a block diagonal covariance matrix
    with intervals dts

    Returns:
        float: Condition number of Q
    """
    min_dt = min(dts)
    max_dt = max(dts)
    if dim == 3:

        def arr_func(dt: Iterable) -> np.ndarray:
            return np.array(
                [
                    [dt, dt ** 2 / 2, dt ** 3 / 6],
                    [dt ** 2 / 2, dt ** 3 / 3, dt ** 4 / 8],
                    [dt ** 3 / 6, dt ** 4 / 8, dt ** 5 / 20],
                ]
            )

        max_arr = arr_func(max_dt)
        min_arr = arr_func(min_dt)
        max_eig = max(abs(np.linalg.eigvals(max_arr)))
        min_eig = min(abs(np.linalg.eigvals(min_arr)))
        return max_eig / min_eig
    elif dim == 2:
        max_disc = 1 + max_dt ** 2 / 3 + max_dt ** 4 / 9
        max_eig = (
            1 / 2 * (max_dt + max_dt ** 3 / 3 + max_dt * np.sqrt(max_disc))
        )
        min_disc = 1 + min_dt ** 2 / 3 + min_dt ** 4 / 9
        min_eig = (
            1 / 2 * (min_dt + min_dt ** 3 / 3 - min_dt * np.sqrt(min_disc))
        )
        return max_eig / min_eig
    elif dim == 1:
        return max_dt / min_dt
    else:
        raise ValueError(
            "Can only calculate condition number of 1, 2, or 3"
            "dimensional kalman smoothing covariance."
        )


def reduce_condition(deltas, method=None, minmax_ratio=0.2):
    avg = deltas.mean()
    if method is None or method.lower() == "none":
        return deltas
    elif method.lower() == "avg":
        return avg * np.ones(len(deltas))
    elif method.lower() == "tanh":
        factor = minmax_ratio * avg
        max_deviation = max(*abs(deltas - avg), 1e-3)
        # tanh most active in [-3,3]
        return factor * np.tanh(3 * (deltas - avg) / max_deviation) + avg
    elif method.lower() == "sux":
        return np.ones(deltas)
    else:
        raise ValueError("reduce_condition received bad method parameter")


def vehicle_Qblocks(
    times, rho=1, order=2, conditioner=conditioner, t_scale=t_scale
):
    """Create the diagonal blocks of the kalman matrix for smoothing
    vehicle motion"""

    delta_times = times[1:] - times[:-1]
    dts = delta_times.astype(float) / 1e9 / t_scale
    dts = reduce_condition(dts, method=conditioner)
    cond = q_cond(dts, dim=order)
    if cond > 1e3:
        warnings.warn(ConditionWarning(cond))
    elif cond < 1:
        raise RuntimeError("Calculated invalid condition number for Q")
    if order == 2:
        Qs = [
            t_scale ** 3
            * rho
            * np.array([[dt, dt ** 2 / 2], [dt ** 2 / 2, dt ** 3 / 3]])
            for dt in dts
        ]
    elif order == 3:
        Qs = [
            t_scale ** 5
            * rho
            * np.array(
                [
                    [dt, dt ** 2 / 2, dt ** 3 / 6],
                    [dt ** 2 / 2, dt ** 3 / 3, dt ** 4 / 8],
                    [dt ** 3 / 6, dt ** 4 / 8, dt ** 5 / 20],
                ]
            )
            for dt in dts
        ]
    else:
        raise ValueError
    return Qs


def vehicle_Qinv(
    times, rho=1, order=2, conditioner=conditioner, t_scale=t_scale
):
    """Creates the precision matrix for smoothing the vehicle with velocity
    covariance rho.
    """

    Qs = vehicle_Qblocks(times, rho, order, conditioner, t_scale)
    Qinvs = [np.linalg.inv(Q) for Q in Qs]
    return scipy.sparse.block_diag(Qinvs)


def vehicle_Q(times, rho=1, order=2, conditioner=conditioner, t_scale=t_scale):
    """Creates the covariance matrix for smoothing the vehicle with velocity
    covariance rho.
    """

    Qs = vehicle_Qblocks(times, rho, order, conditioner, t_scale)
    return scipy.sparse.block_diag(Qs)


def vehicle_G(times, order=2, conditioner=conditioner, t_scale=t_scale):
    """Creates the update matrix for smoothing the vehicle"""
    delta_times = times[1:] - times[:-1]
    dts = delta_times.astype(float) / 1e9 / t_scale  # raw dts in nanoseconds
    dts = reduce_condition(dts, method=conditioner)
    if order == 2:
        negGs = [np.array([[-1, 0], [-dt, -1]]) for dt in dts]
    elif order == 3:
        negGs = [
            np.array([[-1, 0, 0], [-dt, -1, 0], [-(dt ** 2) / 2, -dt, -1]])
            for dt in dts
        ]
    else:
        raise ValueError
    negG = scipy.sparse.block_diag(negGs)
    m = len(delta_times) * order
    posG = scipy.sparse.eye(m)
    append_me = scipy.sparse.coo_matrix(([], ([], [])), (m, order))
    return scipy.sparse.hstack((negG, append_me)) + scipy.sparse.hstack(
        (append_me, posG)
    )


def depth_Qblocks(
    depths,
    rho=1,
    order=2,
    depth_rate=None,
    conditioner=conditioner,
    t_scale=t_scale,
    vehicle_vel="otg",
):
    """Create the diagonal blocks of the kalman matrix for smoothing
    current"""

    delta_depths = depths[1:] - depths[:-1]
    if vehicle_vel[:3] == "otg":
        order -= 1
        depth_rate = np.ones(len(delta_depths))
    elif order == 1:
        raise ValueError(
            "If including depth_rate/modeling vttw, minimum order is 2."
        )
    dds = reduce_condition(delta_depths, method=conditioner)
    cond = q_cond(dds, dim=1)

    dr_neg1 = depth_rate ** -1
    dr_neg2 = depth_rate ** -2
    if cond > 1e3:
        warnings.warn(ConditionWarning(cond))
    elif cond < 1:
        raise RuntimeError("Calculated invalid condition number for Q")
    if order == 1:
        Qs = [t_scale ** 2 * rho * np.array([[dd]]) for dd in dds]
    elif order == 2:
        Qs = [
            t_scale ** 2
            * rho
            * np.array(
                [
                    [dd, dd ** 2 / 2 * dr1],
                    [dd ** 2 / 2 * dr1, dd ** 3 / 3 * dr2],
                ]
            )
            for dd, dr1, dr2 in zip(dds, dr_neg1, dr_neg2)
        ]
    elif order == 3:
        Qs = [
            t_scale ** 2
            * rho
            * np.array(
                [
                    [dd, dd ** 2 / 2, dd ** 3 / 6 * dr1],
                    [dd ** 2 / 2, dd ** 3 / 3, dd ** 4 / 8 * dr1],
                    [dd ** 3 / 6 * dr1, dd ** 4 / 8 * dr1, dd ** 5 / 20 * dr2],
                ]
            )
            for dd, dr1, dr2 in zip(dds, dr_neg1, dr_neg2)
        ]
    else:
        raise ValueError
    return Qs


def depth_Qinv(
    depths,
    rho=1,
    order=2,
    depth_rate=None,
    conditioner=conditioner,
    t_scale=t_scale,
    vehicle_vel="otg",
):
    """Creates the precision matrix for smoothing the currint with depth
    covariance rho.
    """
    Qs = depth_Qblocks(
        depths,
        rho,
        order,
        depth_rate,
        conditioner,
        t_scale,
        vehicle_vel="otg",
    )
    Qinvs = [np.linalg.inv(Q) for Q in Qs]
    return scipy.sparse.block_diag(Qinvs)


def depth_Q(
    depths,
    rho=1,
    order=2,
    depth_rate=None,
    conditioner=conditioner,
    t_scale=t_scale,
    vehicle_vel="otg",
):
    """Creates the covariance matrix for smoothing the current with depth
    covariance rho.
    """
    Qs = depth_Qblocks(
        depths,
        rho,
        order,
        depth_rate,
        conditioner,
        t_scale,
        vehicle_vel="otg",
    )
    return scipy.sparse.block_diag(Qs, dtype=float)


def depth_G(
    depths,
    order=2,
    depth_rate=None,
    conditioner=conditioner,
    vehicle_vel="otg",
):
    """Creates the update matrix for smoothing the current"""
    delta_depths = depths[1:] - depths[:-1]
    if vehicle_vel[:3] == "otg":
        order -= 1
        depth_rate = np.ones(len(delta_depths))
    elif order == 1:
        raise ValueError(
            "If including depth_rate/modeling vttw, minimum order is 2."
        )
    dds = reduce_condition(delta_depths, method=conditioner)

    dr_neg1 = depth_rate ** -1
    if order == 1:
        negGs = [np.array([[-1]]) for dd in dds]
    elif order == 2:
        negGs = [
            np.array([[-1, 0], [-dd * dr1, -1]])
            for dd, dr1 in zip(dds, dr_neg1)
        ]
    elif order == 3:
        negGs = [
            np.array(
                [
                    [-1, 0, 0],
                    [-dd, -1, 0],
                    [-(dd ** 2) / 2 * dr1, -dd * dr1, -1],
                ]
            )
            for dd, dr1 in zip(dds, dr_neg1)
        ]
    else:
        raise ValueError
    negG = scipy.sparse.block_diag(negGs)
    m = len(delta_depths) * (order)
    posG = scipy.sparse.eye(m)
    append_me = scipy.sparse.coo_matrix(([], ([], [])), (m, order))
    return scipy.sparse.hstack((negG, append_me)) + scipy.sparse.hstack(
        (append_me, posG)
    )


# %% Data selection
def get_zttw(ddat, direction="north", t_scale=t_scale):
    """Select the hydrodynamic measurements vector in the specified
    direction.
    """
    if direction in ["u", "north", "u_north"]:
        return ddat["uv"].u_north * t_scale
    elif direction in ["v", "east", "v_east"]:
        return ddat["uv"].v_east * t_scale
    else:
        return None


def get_zadcp(adat, direction="north", t_scale=t_scale):
    """Select the adcp measurements vector in the specified
    direction.
    """
    valid_obs = np.isfinite(adat["UV"])
    if direction in ["u", "north", "u_north"]:
        return adat["UV"].T[valid_obs.T].imag * t_scale
    elif direction in ["v", "east", "v_east"]:
        return adat["UV"].T[valid_obs.T].real * t_scale
    else:
        return None


def get_zgps(ddat, direction="north"):
    """Select the GPS measurements vector in the specified
    direction.
    """
    if direction in ["u", "north", "u_north", "gps_ny_north"]:
        return ddat["gps"].gps_ny_north.to_numpy()
    elif direction in ["v", "east", "v_east", "gps_nx_east"]:
        return ddat["gps"].gps_nx_east.to_numpy()
    else:
        return None


def get_zrange(ddat):
    """Select the range measurements vector as well as the range
    offset vectors

    Returns:
        Tuple of Numpy arrays. The first is the range measurement
        vector.  The second is the source position east of the origin.
        The third is the source position north of the origin.  All
        units in meters.
    """
    r = ddat["range"].range.to_numpy()
    x = ddat["range"].src_pos_e.to_numpy()
    y = ddat["range"].src_pos_n.to_numpy()
    return r, x, y


def size_of_x(
    m: int, n: int, vehicle_order: int = 2, current_order: int = 1
) -> int:
    """Calculates the size of the state vector

    Parameters:
        m (int) : number of timepoints
        n (int) : number of depthpoints
        vehicle_order (int) : order of vehicle smoothing in matrix Q
        current_order (int) : order of current smoothing in matrix Q
    """
    return 2 * (m * vehicle_order + n * current_order)


def a_select(m, order=3):
    """Creates the matrix that selects the acceleration entries of the
    state vector for the vehicle in one direction,
    e.g. [a1, v1, x1, a2, v2, x2, v3, ...].

    Parameters:
        m (int) : number of timesteps
        order (int) : order of vehicle smoothing. 2=velocity, 3=accel
            Must be greater than or equal to 3
    """
    mat_shape = (m, order * m)
    if order < 3:
        return None
    return scipy.sparse.coo_matrix(
        (np.ones(m), (range(m), range(order - 3, order * m, order))),
        shape=mat_shape,
    )


def v_select(m, order=2):
    """Creates the matrix that selects the velocity entries of the
    state vector for the vehicle in one direction,
    e.g. [v1, x1, v2, x2, v3, ...].

    Parameters:
        m (int) : number of timesteps
        order (int) : order of vehicle smoothing. 2=velocity, 3=accel
    """
    mat_shape = (m, order * m)
    return scipy.sparse.coo_matrix(
        (np.ones(m), (range(m), range(order - 2, order * m, order))),
        shape=mat_shape,
    )


def x_select(m, order=2):
    """Creates the matrix that selects the position entries of the
    state vector for the vehicle in one direction,
    e.g. [v1, x1, v2, x2, v3, ...].

    Parameters:
        m (int) : number of timepoints
        order (int) : order of vehicle smoothing. 2=velocity, 3=accel
    """
    mat_shape = (m, order * m)
    return scipy.sparse.coo_matrix(
        (np.ones(m), (range(m), range(order - 1, order * m, order))),
        shape=mat_shape,
    )


def ca_select(n, order=3, vehicle_vel="otg"):
    """Creates the matrix that selects the current velocity entries of
    the state vector for the vehicle in one direction,
    e.g. [v1, x1, v2, x2, v3, ...].

    Parameters:
        n (int) : number of depthpoints
        order (int) : order of current smoothing. 2=velocity, 3=accel
        vehicle_vel (str) : whether vehicle velocity models through-the
            -water velocity or over-the-ground
    """

    if order < 3:
        return None
    if vehicle_vel[:3] == "otg":
        order = order - 1
        cols = range(order - 2, order * n, order)
    elif vehicle_vel == "ttw":
        cols = range(order - 3, order * n, order)
    else:
        raise ValueError
    mat_shape = (n, order * n)
    return scipy.sparse.coo_matrix(
        (np.ones(n), (range(n), cols)), shape=mat_shape
    )


def cv_select(n, order=2, vehicle_vel="otg"):
    """Creates the matrix that selects the current velocity entries of
    the state vector for the vehicle in one direction,
    e.g. [v1, x1, v2, x2, v3, ...].

    Parameters:
        n (int) : number of depthpoints
        order (int) : order of current smoothing. 2=velocity, 3=accel
        vehicle_vel (str) : whether vehicle velocity models through-the
            -water velocity or over-the-ground
    """

    if vehicle_vel[:3] == "otg":
        order = order - 1
        cols = range(order - 1, order * n, order)
    elif vehicle_vel == "ttw":
        cols = range(order - 2, order * n, order)
    else:
        raise ValueError
    mat_shape = (n, order * n)
    return scipy.sparse.coo_matrix(
        (np.ones(n), (range(n), cols)), shape=mat_shape
    )


def cx_select(n, order=2, vehicle_vel="otg"):
    """Creates the matrix that selects the current velocity entries of
    the state vector for the vehicle in one direction,
    e.g. [v1, x1, v2, x2, v3, ...].

    Parameters:
        n (int) : number of depthpoints
        order (int) : order of current smoothing. 2=velocity, 3=accel
        vehicle_vel (str) : whether vehicle velocity models through-the
            -water velocity or over-the-ground
    """

    if vehicle_vel[:3] == "otg":
        return None
        # Current X position not modeled when vehicle velocity is measured over ground. # noqa
    elif vehicle_vel == "ttw":
        cols = range(order - 1, order * n, order)
    else:
        raise ValueError
    mat_shape = (n, order * n)
    return scipy.sparse.coo_matrix(
        (np.ones(n), (range(n), cols)), shape=mat_shape
    )


def e_select(m, n, vehicle_order=2, current_order=2, vehicle_vel="otg"):
    """Creates a selection matrix for choosing indexes of X
    related to easterly variables.

    Parameters:
        m (int) : number of timepoints
        n (int) : number of depthpoints
        vehicle_order (int) : order of vehicle smoothing in matrix Q
        current_order (int) : order of current smoothing in matrix Q
        vehicle_vel (str) : if "otg", then current skips modeling 1st order
    """

    current_order -= vehicle_vel[:3] == "otg"
    EV = ev_select(m, n, vehicle_order, current_order)
    EC = ec_select(m, n, vehicle_order, current_order)
    return scipy.sparse.vstack((EV, EC))


def n_select(m, n, vehicle_order=2, current_order=2, vehicle_vel="otg"):
    """Creates a selection matrix for choosing indexes of X
    related to easterly vehicle kinematics.

    Parameters:
        m (int) : number of timepoints
        n (int) : number of depthpoints
        vehicle_order (int) : order of vehicle smoothing in matrix Q
        current_order (int) : order of current smoothing in matrix Q
        vehicle_vel (str) : if "otg", then current skips modeling 1st order
    """

    current_order -= vehicle_vel[:3] == "otg"
    NV = nv_select(m, n, vehicle_order, current_order)
    NC = nc_select(m, n, vehicle_order, current_order)
    return scipy.sparse.vstack((NV, NC))


def ev_select(m, n, vehicle_order=2, current_order=2, vehicle_vel="otg"):
    """Creates a selection matrix for choosing indexes of X
    related to easterly vehicle kinematics.

    Parameters:
        m (int) : number of timepoints
        n (int) : number of depthpoints
        vehicle_order (int) : order of vehicle smoothing in matrix Q
        current_order (int) : order of current smoothing in matrix Q
        vehicle_vel (str) : if "otg", then current skips modeling 1st order
    """

    current_order -= vehicle_vel[:3] == "otg"
    n_rows = vehicle_order * m
    n_cols = size_of_x(m, n, vehicle_order, current_order)
    return scipy.sparse.eye(n_rows, n_cols)


def nv_select(m, n, vehicle_order=2, current_order=2, vehicle_vel="otg"):
    """Creates a selection matrix for choosing indexes of X
    related to northerly vehicle kinematics.

    Parameters:
        m (int) : number of timepoints
        n (int) : number of depthpoints
        vehicle_order (int) : order of vehicle smoothing in matrix Q
        current_order (int) : order of current smoothing in matrix Q
        vehicle_vel (str) : if "otg", then current skips modeling 1st order
    """

    current_order -= vehicle_vel[:3] == "otg"
    n_rows = vehicle_order * m
    n_cols = size_of_x(m, n, vehicle_order, current_order)
    diag = vehicle_order * m

    return scipy.sparse.eye(n_rows, n_cols, diag)


def ec_select(m, n, vehicle_order=2, current_order=2, vehicle_vel="otg"):
    """Creates a selection matrix for choosing indexes of X
    related to easterly current.

    Parameters:
        m (int) : number of timepoints
        n (int) : number of depthpoints
        vehicle_order (int) : order of vehicle smoothing in matrix Q
        current_order (int) : order of current smoothing in matrix Q
        vehicle_vel (str) : if "otg", then current skips modeling 1st order
    """

    current_order -= vehicle_vel[:3] == "otg"
    n_rows = current_order * n
    n_cols = size_of_x(m, n, vehicle_order, current_order)
    diag = 2 * vehicle_order * m

    return scipy.sparse.eye(n_rows, n_cols, diag)


def nc_select(m, n, vehicle_order=2, current_order=2, vehicle_vel="otg"):
    """Creates a selection matrix for choosing indexes of X
    related to northerly current.

    Parameters:
        m (int) : number of timepoints
        n (int) : number of depthpoints
        vehicle_order (int) : order of vehicle smoothing in matrix Q
        current_order (int) : order of current smoothing in matrix Q
        vehicle_vel (str) : if "otg", then current skips modeling 1st order
    """

    current_order -= vehicle_vel[:3] == "otg"
    n_rows = current_order * n
    n_cols = size_of_x(m, n, vehicle_order, current_order)
    diag = 2 * vehicle_order * m + current_order * n

    return scipy.sparse.eye(n_rows, n_cols, diag)


def legacy_select(
    m, n, vehicle_order=2, current_order=2, vehicle_vel="otg", prob=None
):
    """Creates a selection matrix for choosing indexes of X
    that align with the original modeling variable (smoothing order=2,
    vehicle_vel="otg").

    Parameters:
        m (int) : number of timepoints
        n (int) : number of depthpoints
        vehicle_order (int) : order of vehicle smoothing in matrix Q
        current_order (int) : order of current smoothing in matrix Q
        vehicle_vel (str) : if "otg", then current skips modeling 1st order
        prob (GliderProblem) : The problem that is redundant with all that info
    """

    EC = ec_select(m, n, vehicle_order, current_order, vehicle_vel)
    NC = nc_select(m, n, vehicle_order, current_order, vehicle_vel)
    CV = cv_select(n, current_order, vehicle_vel)

    current_order -= vehicle_vel[:3] == "otg"
    vehicle_block = scipy.sparse.diags(
        np.ones(2), vehicle_order - 2, (2, vehicle_order)
    )
    vehicle_mat = scipy.sparse.block_diag(list(repeat(vehicle_block, m)))

    if vehicle_vel[:3] == "otg":
        vehicle_c_mat = scipy.sparse.csr_matrix((2 * m, current_order * n))
    else:
        vehicle_depths = dp._depth_interpolator(prob.times, prob.ddat).loc[
            prob.times, "depth"
        ]
        vehicle_depths = vehicle_depths.to_numpy().flatten()
        d_list = list(prob.depths)
        idxdepth = [d_list.index(d) for d in vehicle_depths]

        vehicle_c_block = scipy.sparse.diags(
            np.ones(2), current_order - 2, (2, current_order)
        )
        idx_depth_shift = [-1] + idxdepth[:-1]
        blocks = []
        for i, (l, r) in enumerate(zip(idx_depth_shift, idxdepth)):
            block = scipy.sparse.csr_matrix((2 * m, current_order * (r - l)))
            block[2 * i : 2 * (i + 1), -current_order:] = vehicle_c_block
            blocks.append(block)
        vehicle_c_mat = scipy.sparse.hstack(blocks)
        w = vehicle_c_mat.shape[1]
        blank_mat = scipy.sparse.csr_matrix((2 * m, current_order * n - w))
        vehicle_c_mat = scipy.sparse.hstack((vehicle_c_mat, blank_mat))

    vehicle_mat_e = scipy.sparse.hstack(
        (
            vehicle_mat,
            scipy.sparse.csr_matrix((2 * m, vehicle_order * m)),
            vehicle_c_mat,
            scipy.sparse.csr_matrix((2 * m, current_order * n)),
        )
    )
    vehicle_mat_n = scipy.sparse.hstack(
        (
            scipy.sparse.csr_matrix((2 * m, vehicle_order * m)),
            vehicle_mat,
            scipy.sparse.csr_matrix((2 * m, current_order * n)),
            vehicle_c_mat,
        )
    )
    current_mat_n = CV @ NC
    current_mat_e = CV @ EC

    return scipy.sparse.vstack(
        (vehicle_mat_e, vehicle_mat_n, current_mat_e, current_mat_n)
    )


# %%
