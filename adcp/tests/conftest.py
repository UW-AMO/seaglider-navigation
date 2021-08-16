import pytest

import pandas as pd
import numpy as np

from adcp import simulation as sim
import adcp


@pytest.fixture(name="standard_sim")
def standard_sim_fixture():
    yield standard_sim()


def standard_sim():
    sp = sim.SimParams()
    ddat, adat, x, _, _ = sim.simulate(sp)
    return ddat, adat, x


@pytest.fixture(name="standard_prob")
def standard_prob_fixture(standard_sim):
    yield standard_prob(standard_sim)


def standard_prob(sim=None):
    if sim is None:
        sim = standard_sim()
    ddat, adat, _ = sim
    data = adcp.ProblemData(ddat, adat)
    return adcp.GliderProblem(data, adcp.ProblemConfig(), adcp.Weights())


@pytest.fixture(name="simple_sim")
def simple_sim_fixture():
    yield simple_sim()


def _generate_simple_sim():
    sp = sim.SimParams(
        duration=pd.Timedelta("1 min"), max_depth=10, n_timepoints=5
    )
    ddat, adat, x, _, _ = sim.simulate(sp)
    return ddat, adat, x


def simple_sim(auto=False):
    if auto:
        return _generate_simple_sim()
    gps_df = pd.DataFrame(
        [[0.0, 5.0], [100.0, 103.0]],
        columns=["gps_nx_east", "gps_ny_north"],
        index=pd.Index(
            pd.to_datetime(
                [
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:00:30",
                ]
            ),
            name="time",
        ),
    )
    uv_df = pd.DataFrame(
        [[0.3, 0.1]],
        columns=["u_north", "v_east"],
        index=pd.Index(
            pd.to_datetime(
                [
                    "2020-01-01 00:00:10",
                ]
            ),
            name="time",
        ),
    )
    range_df = pd.DataFrame(
        columns=["src_pos_e", "src_pos_n", "range"],
        index=pd.Index([], name="time", dtype="datetime64[ns]"),
    )
    depth_df = pd.DataFrame(
        [[0.0], [1.5], [1.5]],
        columns=["depth"],
        index=pd.Index(
            pd.to_datetime(
                [
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:00:10",
                    "2020-01-01 00:00:30",
                ]
            ),
            name="time",
        ),
    )
    ddat = {"gps": gps_df, "uv": uv_df, "range": range_df, "depth": depth_df}

    adat = {
        "time": np.array(["2020-01-01 00:00:20"], dtype="datetime64[ns]"),
        "Z": np.array(
            [
                [2.0],
                [1.0],
                [-1.0],
                [-2.0],
            ]
        ),
        "UV": np.array([[0.1 + 0.05j], [0.11 + 0.06j], [np.nan], [np.nan]]),
    }
    return ddat, adat, None  # stubbed to align return with standard_sim()


@pytest.fixture(name="simple_prob")
def simple_prob_fixture():
    yield simple_prob()


def simple_prob():
    ddat, adat, _ = simple_sim()
    data = adcp.ProblemData(ddat, adat)
    return adcp.GliderProblem(data, adcp.ProblemConfig(), adcp.Weights())


@pytest.fixture(name="simple_otg_cov_prob")
def simple_otg_cov_prob_fixture():
    yield simple_otg_cov_prob()


def simple_otg_cov_prob():
    ddat, adat, _ = simple_sim()
    data = adcp.ProblemData(ddat, adat)
    return adcp.GliderProblem(
        data, adcp.ProblemConfig(vehicle_vel="otg-cov"), adcp.Weights()
    )


@pytest.fixture(name="simple_otg_cov_prob_3rd_order")
def simple_otg_cov_prob_3rd_order_fixture():
    yield simple_otg_cov_prob_3rd_order()


def simple_otg_cov_prob_3rd_order():
    ddat, adat, _ = simple_sim()
    data = adcp.ProblemData(ddat, adat)
    return adcp.GliderProblem(
        data,
        adcp.ProblemConfig(
            vehicle_vel="otg-cov", vehicle_order=3, current_order=3
        ),
        adcp.Weights(),
    )
