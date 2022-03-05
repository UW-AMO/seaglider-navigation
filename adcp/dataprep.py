# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 00:45:24 2019

@author: 600301
"""
# built-in library
from pathlib import Path
import datetime as dt

# 3rd party libraries
import h5py
from scipy.io import loadmat
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd

data_dir = Path(__file__).parent.parent / "data"


def load_mooring(filename):
    """Load and return all moored current measurements

    Parameters:
        filename (str) : filename of mooring data.

    Returns:
        dict
    """
    arrays = {}
    with h5py.File(data_dir / filename, "r") as f:
        for k, v in f.items():
            arrays[k] = np.array(v)
    m2pd = np.vectorize(_mt2pd)
    arrays["time"] = m2pd(arrays["datenumber"])
    arrays["time"] = pd.to_datetime(arrays["time"].squeeze())
    arrays.pop("datenumber")
    return arrays


def load_dive(run_num):
    """Load and return all dive data for a run

    Parameters:
        run_num (int) : id of dive and ADCP data

    Returns:
        dict
    """
    dive = "dv" + str(run_num)
    dive_data = loadmat(data_dir / (dive + ".mat"))[dive]

    gps = {
        "unixtime": dive_data["GPS"][0][0]["unixtime"][0][0],
        "gps_nx_east": dive_data["GPS"][0][0]["nx_east"][0][0],
        "gps_ny_north": dive_data["GPS"][0][0]["ny_north"][0][0],
    }
    depth = {
        "depth": dive_data["DEPTH"][0][0]["depth"][0][0],
        "unixtime": dive_data["DEPTH"][0][0]["unixtime"][0][0],
    }
    uv = {
        "u_north": dive_data["UV"][0][0]["U_north"][0][0],
        "v_east": dive_data["UV"][0][0]["V_east"][0][0],
        "unixtime": dive_data["UV"][0][0]["unixtime"][0][0],
    }
    try:
        ranges = {
            "unixtime": dive_data["RANGE"][0][0]["unixtime"][0][0],
            "src_lat": dive_data["RANGE"][0][0]["src_lat"][0][0],
            "src_lon": dive_data["RANGE"][0][0]["src_lon"][0][0],
            "src_pos_e": dive_data["RANGE"][0][0]["src_pos_enu"][0][0][:, 0],
            "src_pos_n": dive_data["RANGE"][0][0]["src_pos_enu"][0][0][:, 1],
            "src_pos_u": dive_data["RANGE"][0][0]["src_pos_enu"][0][0][:, 2],
            "range": dive_data["RANGE"][0][0]["range"][0][0],
        }
    # When range is stored in field 'range_km' instead of 'range'
    except ValueError:
        try:
            ranges = {
                "unixtime": dive_data["RANGE"][0][0]["unixtime"][0][0],
                "src_lat": dive_data["RANGE"][0][0]["src_lat"][0][0],
                "src_lon": dive_data["RANGE"][0][0]["src_lon"][0][0],
                "src_pos_e": dive_data["RANGE"][0][0]["src_pos_enu"][0][0][
                    :, 0
                ],  # noqa: E501
                "src_pos_n": dive_data["RANGE"][0][0]["src_pos_enu"][0][0][
                    :, 1
                ],  # noqa: E501
                "src_pos_u": dive_data["RANGE"][0][0]["src_pos_enu"][0][0][
                    :, 2
                ],  # noqa: E501
                "range": 1000 * dive_data["RANGE"][0][0]["range_km"][0][0],
            }
        # When range is stored in field 'range_sphere_km' instead of 'range_km'
        except ValueError:
            ranges = {
                "unixtime": dive_data["RANGE"][0][0]["unixtime"][0][0],
                "src_lat": dive_data["RANGE"][0][0]["src_lat"][0][0],
                "src_lon": dive_data["RANGE"][0][0]["src_lon"][0][0],
                "src_pos_e": dive_data["RANGE"][0][0]["src_pos_enu"][0][0][
                    :, 0
                ],  # noqa: E501
                "src_pos_n": dive_data["RANGE"][0][0]["src_pos_enu"][0][0][
                    :, 1
                ],  # noqa: E501
                "src_pos_u": dive_data["RANGE"][0][0]["src_pos_enu"][0][0][
                    :, 2
                ],  # noqa: E501
                "range": 1000
                * dive_data["RANGE"][0][0]["range_sphere_km"][0][
                    0
                ],  # noqa: E501
            }

    gpsdf = _array_dict2df(gps)
    depthdf = _array_dict2df(depth)
    uvdf = _array_dict2df(uv)
    uvdf = uvdf.reset_index().drop_duplicates(subset=["time"], keep="last")
    uvdf = uvdf.set_index("time")
    rangedf = _array_dict2df(ranges)
    return {"gps": gpsdf, "depth": depthdf, "uv": uvdf, "range": rangedf}


def load_adcp(run_num):
    """Load and return all dive data for a run

    Parameters:
        run_num (int) : id of dive and ADCP data

    Returns:
        dict
    """
    glider = "Glider_ADCP_" + str(run_num)
    adcp_data = loadmat(data_dir / (glider + ".mat"))
    adcp_data.pop("__header__")
    adcp_data.pop("__version__")
    adcp_data.pop("__globals__")
    # Not collected in time zone aware time
    offset = np.timedelta64(7, "h")
    caster = np.vectorize(lambda time: _mt2pd(time) - offset)
    adcp_data["time"] = caster(adcp_data["Mtime"]).squeeze()
    adcp_data.pop("Mtime")
    #    adcp_copy = adcp_data.copy()
    #    for key in adcp_data.keys():
    #        if adcp_data[key].shape[0]>1:
    #            for ind in range(adcp_data[key].shape[0]):
    #                newkey = key+str(ind+1)
    #                adcp_copy[newkey] = adcp_data[key][ind,:]
    #
    #            adcp_copy.pop(key)
    #
    #    return adcp_copy
    return adcp_data


def _mt2pd(mat_datenum):
    """Cast a Matlab date number (float) into a numpy datetime64 object"""
    y0_leapyear = dt.timedelta(days=366)
    date = dt.datetime.fromordinal(int(mat_datenum)) - y0_leapyear
    time = dt.timedelta(days=mat_datenum % 1)
    return np.datetime64(date + time, "ms")


def _unix2pd(timestamp):
    """Cast a POSIX timestamp (int, float) into a numpy datetime64 object"""
    date_time = dt.datetime.fromtimestamp(timestamp)
    return np.datetime64(date_time, "ms")


def _array_dict2df(array_dict):
    """Converts a dictionary with (1xn) arrays as values into
    a DataFrame, standardizing dates along the way.
    """
    flat_dict = {
        key: list(array_dict[key].flatten()) for key in array_dict.keys()
    }
    df = pd.DataFrame(flat_dict)
    if "unixtime" in df.columns:
        df["time"] = df.unixtime.apply(_unix2pd)
        df = df.drop(columns=["unixtime"])
        df = df.set_index("time")
    return df


def depthpoints(adat, ddat):
    """identify the depth index for all dive and ADCP measurements

    Parameters:
        ddat (dict): the recorded dive data returned by load_dive()
        adat (dict): the recorded ADCP data returned by load_adcp()
    """
    times = timepoints(adat, ddat)
    depth_df = _depth_interpolator(times, ddat)
    turnaround = depth_df.ascending.idxmax()
    deepest = depth_df.loc[turnaround, "depth"]

    descending = pd.to_datetime(adat["time"]) <= turnaround
    down_depths = adat["Z"][:, descending].flatten()
    up_depths = 2 * deepest - adat["Z"][:, ~descending].flatten()
    depth_arrays = (
        down_depths,
        up_depths,
        depth_df.depth.to_numpy().flatten(),
    )
    depths = np.unique(np.concatenate(depth_arrays))
    return depths


def depth_rates(times, depths, ddat):
    """Calculate the depth rate of the vehicle for each interval between
    all possible measurement depths.

     Parameters:
        times ([numpy.datetime64,]) : the times of observations
        depths ([numpy.datetime64,]) : the depths of observations
        ddat (dict): the recorded dive data returned by load_dive()
    """

    depth_df = _depth_interpolator(times, ddat)
    depth_df = depth_df.reset_index().set_index("depth")
    x = depth_df.index
    y = depth_df["time"].values  # in nanoseconds
    interp_func = interp1d(x, y, fill_value="extrapolate")
    depth_df = pd.DataFrame(
        interp_func(depths), columns=["time"], index=depths
    )

    depth_diff = depth_df.index.values[1:] - depth_df.index.values[:-1]
    time_diff = depth_df["time"].values[1:] - depth_df["time"].values[:-1]
    time_diff = time_diff / 1e9  # back to seconds

    return depth_diff / time_diff


def _depth_interpolator(times, ddat):
    """Create a dataframe to interpolate depth to all the times in <times>

    Parameters:
        times ([numpy.datetime64,]) : all of the sample times to predict
            V_otg for.  returned by dataprep.timepoints()
        ddat (dict): the recorded dive data returned by load_dive()

    Returns:
        pandas DataFrame
    """
    new_times = pd.DataFrame([], index=times, columns=["depth"], dtype=float)
    new_times.index.name = "time"
    depth_df = ddat["depth"].append(new_times)
    depth_df = depth_df.interpolate(method="time", limit_direction="both")
    depth_df = depth_df.reset_index().drop_duplicates(subset=["time"])
    depth_df = depth_df.set_index("time")

    # Reflect depths below the deepest depth
    deepest = depth_df.max().iloc[0]
    argmax = depth_df.depth.idxmax()
    depth_df["ascending"] = False
    ascending = depth_df.index > argmax
    reflected = 2 * deepest - depth_df.loc[ascending, "depth"]
    depth_df.loc[ascending, "ascending"] = True
    depth_df.loc[ascending, "depth"] = reflected

    return depth_df.sort_index()


def timepoints(adat, ddat):
    """identify the sorted time index for all dive and ADCP measurements

    Parameters:
        ddat (dict): the recorded dive data returned by load_dive()
        adat (dict): the recorded ADCP data returned by load_adcp()
    """
    uv_time = ddat["uv"].index.to_numpy()
    depth_time = ddat["depth"].index.to_numpy()
    gps_time = ddat["gps"].index.to_numpy()
    range_time = ddat["range"].index.to_numpy()
    adcp_time = adat["time"]

    combined = np.concatenate(
        (uv_time, gps_time, range_time, adcp_time, depth_time)
    )
    return np.unique(combined)


def dead_reckon(ddat, final_posit=None):
    """Dead reckons motion with and without average current.

    Parameters:
        ddat (dict): the recorded dive data returned by load_dive()
        final_posit (Tuple(Float, Float)): the GPS final coordinates

    Returns:
        Tuple:
        (1) Pandas dataframe, an augmented version of ddat['uv'] with
        dead reckoned position columns
        (2) Boolean value whether corrections for depth averaged current is
        included or not.

    Note:
        Requires that ddat's uv DataFrame and gps DataFrame both have
        indexes that start and end roughly around the same time.
    """
    delta_times = ddat["uv"].index[1:] - ddat["uv"].index[:-1]
    delta_times = delta_times.insert(0, pd.to_timedelta(0))
    cum_secs = np.cumsum(delta_times.map(lambda t: t.total_seconds()))
    delta_x = delta_times.total_seconds() * ddat["uv"].loc[:, "v_east"]
    delta_y = delta_times.total_seconds() * ddat["uv"].loc[:, "u_north"]
    x_dead = np.cumsum(delta_x) + ddat["gps"].gps_nx_east.iloc[0]
    y_dead = np.cumsum(delta_y) + ddat["gps"].gps_ny_north.iloc[0]
    new_cols = {
        "x_dead": x_dead,
        "y_dead": y_dead,
    }
    dac = False
    # Now add constant drift to correct reckoning for final GPS position
    if final_posit is not None:
        ddat["gps"].loc[final_posit.name] = final_posit
    gps_pct = (
        ddat["gps"].index[-1] - ddat["gps"].index[0]
    ).total_seconds() / cum_secs[-1]
    if gps_pct > 0.1:  # DAC must be measured by at least 10% of a dive
        x_err = ddat["gps"].gps_nx_east.iloc[-1] - x_dead[-1]
        y_err = ddat["gps"].gps_ny_north.iloc[-1] - y_dead[-1]
        uv_time_elapsed = ddat["uv"].index[-1] - ddat["uv"].index[0]
        x_step = x_err / uv_time_elapsed.total_seconds()
        y_step = y_err / uv_time_elapsed.total_seconds()
        x_correction = x_step * cum_secs
        y_correction = y_step * cum_secs
        x_corrected = x_dead + x_correction
        y_corrected = y_dead + y_correction
        new_cols = {
            **new_cols,
            "x_corr": x_corrected,
            "y_corr": y_corrected,
        }
        dac = True
    new_cols = pd.DataFrame(new_cols, index=ddat["uv"].index)
    return ddat["uv"].join(new_cols, sort=False), dac
