# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 13:00:31 2020

@author: 600301
"""
import random
import copy

import numpy as np
import pandas as pd

import adcp.matbuilder as mb
import adcp.dataprep as dp


# %%
class SimParams:
    """Statically holds glider dive simulation parameters."""

    defaults = {
        "duration": pd.Timedelta("3 hours"),
        "max_depth": 750,
        "n_dives": 1,
        "n_timepoints": 1001,
        "rho_v": 0.1,
        "rho_c": 0.1,
        "rho_t": 1,
        "rho_a": 1,
        "rho_g": 1,
        "rho_r": 1,
        "adcp_bins": 4,
        "seed": 124,
        "sigma_t": 1,
        "sigma_c": 1,
        "curr_method": "constant",
        "vehicle_method": "constant",
        "measure_points": {
            "gps": "endpoints",
            "ttw": 0.5,
            "range": 0.05,
        },
    }

    def __init__(self, copyobj=None, **kwargs):
        """Creates a SimParams object

        Parameters:
            duration (pandas Tiemdelta): duration of the dive profile
            max_depth (int, float): depth at apogee.
            n_dives (int): number of total dives.  Only tested with n=1.
            n_timepoints (int): number of timepoints to sample
            rho_v (float): process variance of vehicle velocity.  Exact
                use depends upon vehicle_method
            rho_c (float): process variance of current profile.  Exact
                use depends upon curr_method
            rho_t (float): variance of error in TTW hydrodynamic model
                measurement
            rho_a (float): variance of error in ADCP measurement
            rho_g (float): variance of error in gps measurement
            rho_r (float): variance of error in range measurement
            adcp_bins (int): number of ADCP bins measured.
            seed (int): random seed
            sigma_t (float): variance of max vehicle velocity generator.
                Exact use depends on vehicle_method
            sigma_c (float): variance of max current generator.  Exact
                use depends on curr_method
            curr_method (str): method to simulate current profile
            vehicle_method (str): method to simulate vehicle TTW
                trajectory
            measure_points (dict): dictionary with keys 'gps', 'ttw' and
                'range' to describe how measurement points in time are
                assigned to different measurement devices.
        """
        if copyobj is not None:
            self.__dict__ = copy.copy(copyobj.__dict__)
        else:
            self.__dict__ = copy.copy(self.defaults)
        for k, v in kwargs.items():
            try:
                self.defaults[k]
                setattr(self, k, v)
            except KeyError:
                raise AttributeError(
                    f"{k} not a argument for Problem" " constructor"
                )


# %%
"""A simulation parameter object.  It holds the parameters that get
    passed around by all of the subordinate methods to create a
    simulation.
"""


def gen_dive(sim_params):
    """Simulate the times and depths of measurement for a simulation

    Arguments:
        sim_params (SimParams): Object of parameters, see definition
            in this module

    Returns:
        pandas DataFrame of depth (meters) indexed by time
    """
    start_time = pd.Timestamp("2020-01-01")
    depths_down = np.arange(
        0,
        sim_params.max_depth,
        sim_params.max_depth / (sim_params.n_timepoints // 2),
    )
    depths_up = sim_params.max_depth + np.arange(
        0.1,
        sim_params.max_depth,
        sim_params.max_depth / (sim_params.n_timepoints // 2),
    )
    timepoints = np.linspace(
        start_time.value,
        (start_time + sim_params.duration).value,
        sim_params.n_timepoints,
    )
    timepoints = pd.to_datetime(timepoints)

    if sim_params.n_timepoints % 2 == 1:
        depths = np.concatenate(
            (depths_down.tolist(), [sim_params.max_depth], depths_up.tolist())
        )
    else:
        depths = np.concatenate((depths_down.tolist(), depths_up.tolist()))

    depth_df = pd.DataFrame(
        depths, index=pd.Index(timepoints, name="time"), columns=["depth"]
    )
    return depth_df


def gen_adcp_depths(depth_df, sim_params):
    """Simulate the depths measured by the ADCP

    Arguments:
        depth_df (pandas DataFrame): depth indexed by time
        sim_params (SimParams): namedtuple of parameters, see definition
            in this module

    Returns:
        adcp_df (pandas Series): depths to be measured by ADCP (meters)
            indexed by time of observation
    """
    np.random.seed(sim_params.seed)
    random.seed(sim_params.seed)

    max_adcp_depth = 12  # max height above segalider ADCP can measure

    midpoint = depth_df.index[sim_params.n_timepoints // 2 + 1]

    down_adcp_times = depth_df.index[depth_df.index <= midpoint]
    down_adcp_points = (
        depth_df.loc[down_adcp_times].depth.to_numpy().reshape((-1, 1))
    )
    down_adcp_depths = max_adcp_depth * np.random.random_sample(
        (len(down_adcp_points), sim_params.adcp_bins)
    )
    down_adcp_depths = down_adcp_points - np.sort(down_adcp_depths, axis=1)

    up_adcp_times = depth_df.index[depth_df.index > midpoint]
    up_adcp_points = (
        depth_df.loc[up_adcp_times].depth.to_numpy().reshape((-1, 1))
    )
    up_adcp_depths = max_adcp_depth * np.random.random_sample(
        (len(up_adcp_points), sim_params.adcp_bins)
    )
    up_adcp_depths = up_adcp_points + np.sort(up_adcp_depths, axis=1)
    adcp_depths = np.vstack((down_adcp_depths, up_adcp_depths))
    return pd.DataFrame(adcp_depths, index=depth_df.index)


def sim_current_profile(depths, sim_params, method="cos"):
    """Simulate the current profile at all relevant depths

    Arguments:
        depths (iterable): depths to assign a current to.
        sim_params (SimParams): namedtuple of parameters, see definition
            in this module
        method (string)

    Returns:
        pandas DataFrame of current (meters/sec) indexed by depth (meters)
    """
    all_depths = np.array(depths)
    if method.lower() in ["cos", "curved"]:
        x_scale = np.pi / (sim_params.max_depth)
        e_scale_down = sim_params.sigma_c * np.random.normal()
        e_scale_up = sim_params.sigma_c * np.random.normal()
        n_scale_down = sim_params.sigma_c * np.random.normal()
        n_scale_up = sim_params.sigma_c * np.random.normal()
        midpoint_idx = np.argmin(abs(all_depths - sim_params.max_depth))
        e_curr_down = e_scale_down + e_scale_down * np.cos(
            all_depths[:midpoint_idx] * x_scale
        )
        e_curr_up = e_scale_up + e_scale_up * np.cos(
            all_depths[midpoint_idx:] * x_scale
        )
        n_curr_down = n_scale_down + n_scale_down * np.cos(
            all_depths[:midpoint_idx] * x_scale
        )
        n_curr_up = n_scale_up + n_scale_up * np.cos(
            all_depths[midpoint_idx:] * x_scale
        )
        e_curr = np.hstack((e_curr_down, e_curr_up))
        n_curr = np.hstack((n_curr_down, n_curr_up))
    elif method.lower() == "linear":
        first_n = sim_params.sigma_c * np.random.normal(size=1)
        first_e = sim_params.sigma_c * np.random.normal(size=1)
        last_n = sim_params.sigma_c * np.random.normal(size=1)
        last_e = sim_params.sigma_c * np.random.normal(size=1)
        n_curr = np.linspace(first_n[0], last_n[0], len(depths))
        e_curr = np.linspace(first_e[0], last_e[0], len(depths))
        n_curr = n_curr + np.random.normal(
            scale=sim_params.rho_c, size=len(n_curr)
        )
        e_curr = e_curr + np.random.normal(
            scale=sim_params.rho_c, size=len(e_curr)
        )
    elif method.lower() == "constant":
        first_n = sim_params.sigma_c * np.random.normal(size=1)
        first_e = sim_params.sigma_c * np.random.normal(size=1)
        last_n = first_n
        last_e = first_e
        n_curr = np.linspace(first_n[0], last_n[0], len(depths))
        e_curr = np.linspace(first_e[0], last_e[0], len(depths))
        n_curr = n_curr + np.random.normal(
            scale=sim_params.rho_c, size=len(n_curr)
        )
        e_curr = e_curr + np.random.normal(
            scale=sim_params.rho_c, size=len(e_curr)
        )
    else:
        raise ValueError(f"current simulation method {method} not defined")
    #   Truly principled method:
    #    Qc = mb.depth_Q(all_depths, rho_c)
    #    delta_C_n = np.random.multivariate_normal(
    #                        mean=np.zeros(len(all_depths)-1),
    #                        cov = Qc.todense())
    #    delta_C_e = np.random.multivariate_normal(
    #                        mean=np.zeros(len(all_depths)-1),
    #                        cov = Qc.todense())
    #    cum_diff_n = np.cumsum(delta_C_n)
    #    cum_diff_e = np.cumsum(delta_C_e)
    #    current_n = np.concatenate((first_n, first_n+cum_diff_n))
    #    current_e = np.concatenate((first_e, first_n+cum_diff_e))

    curr_df = pd.DataFrame(
        data=np.hstack([e_curr.reshape((-1, 1)), n_curr.reshape((-1, 1))]),
        index=pd.Index(depths, name="depth"),
        columns=["curr_e", "curr_n"],
    )
    return curr_df


def sim_vehicle_path(depth_df, curr_df, sim_params, method="sin"):
    """Simulate the vehicle kinematics at all relevant times

    Arguments:
        depth_df (pandas DataFrame): depth indexed by time
        curr_df (pandas DataFrame): current indexed by depth
        sim_params (SimParams): namedtuple of parameters, see definition
            in this module
        method (string)

    Returns:
        pandas DataFrame of position (meters), TTW velocity, and
            OTG velocity (meters/second) indexed by time.
    """
    total_time = float((depth_df.index[-1] - depth_df.index[0]).value)
    delta_t = (depth_df.index - depth_df.index[0]).values
    delta_t = np.array([float(dt) for dt in delta_t])
    if method.lower() in ["sin", "curved"]:
        t_to_x = 2 * np.pi / total_time
        e_scale_down = sim_params.sigma_t * np.random.normal()
        e_scale_up = sim_params.sigma_t * np.random.normal()
        n_scale_down = sim_params.sigma_t * np.random.normal()
        n_scale_up = sim_params.sigma_t * np.random.normal()
        midpoint_idx = len(delta_t) // 2
        ttw_e_down = e_scale_down * np.sin(delta_t[:midpoint_idx] * t_to_x)
        ttw_e_up = e_scale_up * np.sin(delta_t[midpoint_idx:] * t_to_x)
        ttw_n_down = n_scale_down * np.sin(delta_t[:midpoint_idx] * t_to_x)
        ttw_n_up = n_scale_up * np.sin(delta_t[midpoint_idx:] * t_to_x)
        ttw_e = np.hstack((ttw_e_down, ttw_e_up))
        ttw_n = np.hstack((ttw_n_down, ttw_n_up))
    elif method.lower() == "linear":
        first_n = sim_params.sigma_t * np.random.normal(size=1)
        first_e = sim_params.sigma_t * np.random.normal(size=1)
        last_n = sim_params.sigma_t * np.random.normal(size=1)
        last_e = sim_params.sigma_t * np.random.normal(size=1)
        ttw_n = np.linspace(first_n[0], last_n[0], len(delta_t))
        ttw_e = np.linspace(first_e[0], last_e[0], len(delta_t))
    elif method.lower() == "constant":
        e_scale = sim_params.sigma_t * np.random.normal()
        n_scale = sim_params.sigma_t * np.random.normal()
        ttw_e = np.repeat(e_scale, len(delta_t))
        ttw_n = np.repeat(n_scale, len(delta_t))
    else:
        raise ValueError(f"vehicle simulation method {method} not defined")
    ttw_n = ttw_n + np.random.normal(scale=sim_params.rho_v, size=len(ttw_n))
    ttw_e = ttw_e + np.random.normal(scale=sim_params.rho_v, size=len(ttw_e))

    v_df = curr_df.loc[pd.Index(depth_df.depth)]
    v_df.index = depth_df.index
    v_df["ttw_e"] = ttw_e
    v_df["ttw_n"] = ttw_n
    v_df["v_otg_n"] = v_df["curr_n"] + v_df["ttw_n"]
    v_df["v_otg_e"] = v_df["curr_e"] + v_df["ttw_e"]
    delta_t = (v_df.index[1:] - v_df.index[:-1]).total_seconds()

    delta_x_otg_n = v_df.v_otg_n.iloc[1:] * delta_t
    delta_x_otg_e = v_df.v_otg_e.iloc[1:] * delta_t
    v_df["x"] = np.hstack(([0], delta_x_otg_e.cumsum()))
    v_df["y"] = np.hstack(([0], delta_x_otg_n.cumsum()))

    return v_df


def select_times(depth_df, sim_params):
    """Choose which time points get measured by which device.

    Arguments:
        depth_df (pandas DataFrame): depth indexed by time
        sim_params (SimParams): namedtuple of parameters.  In particular
            the measure_points field is expected to be a dictionary
            containing either strings or floats in [0,1) to represent
            the portion of observations measured by the key (device).

    Returns:
        dictionary of timepoints indexed by the sensor observing during
        that time.
    """
    timepoints = depth_df.index
    n_timepoints = len(timepoints)
    if n_timepoints < 3:
        raise ValueError("need at least 3 timepoints to assign")
    if sim_params.measure_points["gps"] == "first":
        gps_times = timepoints[0:1]
    elif sim_params.measure_points["gps"] == "last":
        gps_times = timepoints[-2:-1]
    elif sim_params.measure_points["gps"] == "endpoints":
        gps_times = timepoints[[0, -1]]
    elif (
        sim_params.measure_points["range"]
        + sim_params.measure_points["gps"]
        + sim_params.measure_points["ttw"]
    ) >= 1:
        raise ValueError(
            "Can only sample 100% of timepoints,"
            "check sim_params.measure_points"
        )
    if (
        sim_params.measure_points["range"] + sim_params.measure_points["ttw"]
    ) >= 1:
        raise ValueError(
            "Can only sample 100% of timepoints,"
            "check sim_params.measure_points"
        )
    unallocated_times = set(timepoints) - set(gps_times)
    num_ttw_times = int(n_timepoints * sim_params.measure_points["ttw"])
    ttw_times = pd.to_datetime(
        sorted(random.sample(unallocated_times, num_ttw_times))
    )

    unallocated_times -= set(ttw_times)
    num_range_times = int(n_timepoints * sim_params.measure_points["range"])
    range_times = pd.to_datetime(
        sorted(random.sample(unallocated_times, num_range_times))
    )

    unallocated_times -= set(range_times)
    if isinstance(sim_params.measure_points["gps"], float):
        num_gps_times = int(n_timepoints * sim_params.measure_points["gps"])
        gps_times = pd.to_datetime(
            sorted(random.sample(unallocated_times, num_gps_times))
        )
        unallocated_times -= set(gps_times)

    adcp_times = pd.to_datetime(sorted(unallocated_times))

    return {
        "gps": gps_times,
        "ttw": ttw_times,
        "range": range_times,
        "adcp": adcp_times,
    }


# %%
def sim_measurement_noise(
    depth_df,
    adcp_df,
    curr_df,
    v_df,
    ttw_times,
    adcp_times,
    gps_times,
    range_times,
    sim_params,
):
    """Generate measurement noise around the simulated vehicle path
        and current profile

    Arguments:
        depth_df (pandas DataFrame): depth indexed by time
        curr_df (pandas DataFrame): current indexed by depth
        v_df (pandas DataFrame): kinematics indexed by time
        ttw_times (iterable of np.datetime64): times the hydrodynamic
            model measures
        adcp_times (iterable of np.datetime64): times the ADCP measures
        gps_times (iterable of np.datetime64): times the gps measures
        range_times (iterable of np.datetime64): times the range measures
        sim_params (SimParams): namedtuple of parameters, see definition
            in this module

    Returns:
        Dict of dataframes of northward TTW velocity, eastward TTW
            velocity, n/e adcp, n/e gps, and range, followed by range
            posits, the positions of the range measurement beacons
    """
    # Calculate & simulate z_ttw, z_adcp, and z_gps
    z_ttw_n = np.random.normal(
        loc=v_df.loc[ttw_times, "ttw_n"], scale=sim_params.rho_t
    )
    z_ttw_e = np.random.normal(
        loc=v_df.loc[ttw_times, "ttw_e"], scale=sim_params.rho_t
    )

    def n_curr_selector(depth):
        return np.nan if np.isnan(depth) else curr_df.loc[depth, "curr_n"]

    def e_curr_selector(depth):
        return np.nan if np.isnan(depth) else curr_df.loc[depth, "curr_e"]

    cse = np.vectorize(e_curr_selector)
    csn = np.vectorize(n_curr_selector)

    n_adcp_mean = csn(adcp_df.loc[adcp_times]) - v_df.loc[
        adcp_times, "v_otg_n"
    ].values.reshape((-1, 1))
    e_adcp_mean = cse(adcp_df.loc[adcp_times]) - v_df.loc[
        adcp_times, "v_otg_e"
    ].values.reshape((-1, 1))
    z_adcp_n = np.random.normal(loc=n_adcp_mean, scale=sim_params.rho_a)
    z_adcp_e = np.random.normal(loc=e_adcp_mean, scale=sim_params.rho_a)

    z_gps_n = np.random.normal(
        loc=v_df.loc[gps_times, "y"], scale=sim_params.rho_g
    )
    z_gps_e = np.random.normal(
        loc=v_df.loc[gps_times, "x"], scale=sim_params.rho_g
    )

    # Simulate range
    # Choose a covariance based upon maximum position deviation from origin
    range_scale = np.abs((v_df.y.min(), v_df.x.min())).max() / 10
    range_posit_randomness = (
        np.random.random_sample((len(range_times), 2)) - 0.5
    )
    range_posits = range_posit_randomness * range_scale
    ranges = np.sqrt(
        (range_posits[:, 0] - v_df.loc[range_times, "x"]) ** 2
        + (range_posits[:, 1] - v_df.loc[range_times, "y"]) ** 2
    )
    z_range = np.random.normal(loc=ranges, scale=sim_params.rho_r)

    return {
        "z_ttw_n": z_ttw_n,
        "z_ttw_e": z_ttw_e,
        "z_adcp_n": z_adcp_n,
        "z_adcp_e": z_adcp_e,
        "z_gps_n": z_gps_n,
        "z_gps_e": z_gps_e,
        "z_range": z_range,
        "range_posits": range_posits,
    }


# %%
def construct_load_dicts(
    depth_df,
    adcp_df,
    measurements,
    ttw_times,
    adcp_times,
    gps_times,
    range_times,
    sim_params,
):
    """Create the data structures from simulated data that mock
    the result of dataprep.load_dive() and dataprep.load_adcp()

    Arguments:
        depth_df (pandas DataFrame): depth indexed by time
        measurements (dict of numpy arrays): simulated measurements, a
            result from sim_measurement_noise
        ttw_times (iterable of np.datetime64): times the hydrodynamic
            model measures
        adcp_times (iterable of np.datetime64): times the ADCP measures
        gps_times (iterable of np.datetime64): times the gps measures
        range_times (iterable of np.datetime64): times the range measures
        sim_params (SimParams): namedtuple of parameters, see definition
            in this module

    Returns:
        Tuple of (dive data, adcp data)
    """
    z_ttw_n = measurements["z_ttw_n"]
    z_ttw_e = measurements["z_ttw_e"]
    z_adcp_n = measurements["z_adcp_n"]
    z_adcp_e = measurements["z_adcp_e"]
    z_gps_n = measurements["z_gps_n"]
    z_gps_e = measurements["z_gps_e"]
    z_range = measurements["z_range"]
    range_posits = measurements["range_posits"]

    gps_df = pd.DataFrame(
        {"gps_nx_east": z_gps_e, "gps_ny_north": z_gps_n},
        index=pd.Index(gps_times, name="time"),
    )
    range_data = np.hstack((range_posits, z_range.reshape((-1, 1))))
    range_df = pd.DataFrame(
        range_data,
        index=pd.Index(range_times, name="time"),
        columns=["src_pos_e", "src_pos_n", "range"],
    )
    ttw_df = pd.DataFrame(
        np.vstack([z_ttw_n, z_ttw_e]).T,
        index=pd.Index(ttw_times, name="time"),
        columns=["u_north", "v_east"],
    )

    # Note: Have to reflect depths that go past max_depth
    ddf = depth_df.loc[
        pd.Index(
            gps_times.union(range_times).union(ttw_times).sort_values(),
            name="time",
        )
    ]
    reflect = ddf.depth > sim_params.max_depth
    ddf.loc[reflect, "depth"] = (
        2 * sim_params.max_depth - ddf.loc[reflect, "depth"]
    )
    adf = adcp_df.loc[adcp_times]
    reflect = (adf.loc[adcp_times] > sim_params.max_depth).all(axis=1)
    adf.loc[reflect] = 2 * sim_params.max_depth - adf.loc[reflect]
    # Construct dataframes & dictionaries for ADCP data
    ddat = {"gps": gps_df, "depth": ddf, "uv": ttw_df, "range": range_df}
    adat = {
        "time": adcp_times.to_numpy(),
        "Z": adf.values.T,
        "UV": (z_adcp_e + z_adcp_n * 1j).T,
    }

    return ddat, adat


# %%
def true_solution(curr_df, v_df, final_depths, sim_params):
    """Constructs the true solution in the format created by the
    methods in optimization.py

    Arguments:
        curr_df (pandas DataFrame): current indexed by depth
        v_df (pandas DataFrame): kinematics indexed by time
        final_depths (numpy Array): all the depths used in the final
            simulated data.
        sim_params (SimParams): namedtuple of parameters, see definition
            in this module

    Returns:
        1-D numpy array
    """
    # Construct ideal optimization result
    m = sim_params.n_timepoints
    n = len(final_depths)
    EV = mb.ev_select(m, n)
    NV = mb.nv_select(m, n)
    EC = mb.ec_select(m, n)
    NC = mb.nc_select(m, n)
    Vs = mb.v_select(m)
    Xs = mb.x_select(m)

    x = np.zeros(4 * m + 2 * n)
    x += EV.T @ Xs.T @ v_df.x
    x += EV.T @ Vs.T @ v_df.v_otg_e
    x += NV.T @ Xs.T @ v_df.y
    x += NV.T @ Vs.T @ v_df.v_otg_n

    # Inference just uses depth in the dive to guess turnaround not
    # max depth from sim_params.  Thus need to interpolate current to
    # the calculated ascending depths
    cdf = curr_df.append(
        pd.DataFrame(index=final_depths, columns=["curr_e", "curr_n"])
    )
    cdf = cdf.sort_index()
    cdf = cdf.interpolate(method="index")
    subset_df = cdf.loc[~cdf.index.duplicated()].reindex(final_depths)
    x += EC.T @ subset_df.curr_e
    x += NC.T @ subset_df.curr_n

    return x


# %%
def simulate(sim_params, verbose=False):
    """Simulates the data recorded during a dive.

    Returns:
        tuple of 2 dictionaries and a numpy vector.  The first element of
        the tuple is the dive data, as would be loaded from
        adcp.dataprep.load_dive(dive_num). The second element of the
        tuple is the ADCP data, as would be loaded from
        adcp.dataprep.load_adcp(dive_num).  The numpy vector is the true
        simulated current and vehicle kinematics as it should be
        constructed by a solver
            [Vehicle, Current] -> [East, North] -> [v1 x1, v2, x2, ...]
    """
    if verbose:
        print(
            f"""Simulation parameters:
              Random seed: {sim_params.seed}
              Current maximum variance {sim_params.sigma_c}
              Current profile method {sim_params.curr_method}
              Current profile variance {sim_params.rho_c}
              TTW maximum variance {sim_params.sigma_t}
              TTW path method {sim_params.vehicle_method}
              TTW profile variance {sim_params.rho_v}
              ADCP measurement variance {sim_params.rho_a}
              TTW hydrodynamic measurement variance {sim_params.rho_t}
              GPS measurement variance {sim_params.rho_g}
              """
        )

    np.random.seed(sim_params.seed)
    random.seed(sim_params.seed)

    # Identify all times and depths of vehicle measurement
    depth_df = gen_dive(sim_params)
    adcp_df = gen_adcp_depths(depth_df, sim_params)
    all_depths = np.unique(
        np.concatenate((depth_df.values.flatten(), adcp_df.values.flatten()))
    )
    all_depths = sorted(all_depths)

    # Simulate the vehicle and environment
    curr_df = sim_current_profile(
        all_depths, sim_params, sim_params.curr_method
    )
    v_df = sim_vehicle_path(
        depth_df, curr_df, sim_params, sim_params.vehicle_method
    )

    # Choose which timepoints are measured by which instrument
    measure_times = select_times(depth_df, sim_params)
    ttw_times = measure_times["ttw"]
    range_times = measure_times["range"]
    gps_times = measure_times["gps"]
    adcp_times = measure_times["adcp"]

    measurements = sim_measurement_noise(
        depth_df,
        adcp_df,
        curr_df,
        v_df,
        ttw_times,
        adcp_times,
        gps_times,
        range_times,
        sim_params,
    )

    ddat, adat = construct_load_dicts(
        depth_df,
        adcp_df,
        measurements,
        ttw_times,
        adcp_times,
        gps_times,
        range_times,
        sim_params,
    )
    final_depths = dp.depthpoints(adat, ddat)
    x = true_solution(curr_df, v_df, final_depths, sim_params)

    return ddat, adat, x, curr_df, v_df
