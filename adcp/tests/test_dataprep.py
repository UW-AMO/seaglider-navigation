import numpy as np
import pandas as pd
import pytest

from adcp import dataprep as dp


@pytest.fixture
def mock_mdat():
    mdat = dp.load_mooring("CBX16_T3_AUG2017.mat")
    mdat = {
        "time": pd.Index(pd.to_datetime(["20170101", "20170101 010000"])),
        "depth": np.array([[1, 2, 3, 9, 12], [5, 6, 7, 11, 13]]),
        "u": np.concatenate((np.zeros((1, 5)), np.ones((1, 5))), 0),
        "v": np.concatenate((np.zeros((1, 5)), np.ones((1, 5))), 0),
    }
    return mdat


def test_interpolate_mooring(mock_mdat):
    target = pd.to_datetime("20170101 0024")
    result = dp.interpolate_mooring(mock_mdat, target)

    assert result["time"][0] == target
    assert (
        np.linalg.norm(result["depth"] - np.array([3.8, 8.2, 9.8, 11.6, 12.4]))
        < 1e-14
    )
    assert np.linalg.norm(result["u"] - result["v"]) < 1e-14
    assert np.linalg.norm(result["v"] - 0.4 * np.ones(5)) < 1e-14


def test_interpolate_mooring_idempotent(mock_mdat):
    target = pd.to_datetime("20170101 0024")
    result1 = dp.interpolate_mooring(mock_mdat, target)
    result2 = dp.interpolate_mooring(result1, target)

    assert result1["time"][0] == result2["time"][0]
    assert np.linalg.norm(result1["depth"] - result2["depth"]) < 1e-14
    assert np.linalg.norm(result1["u"] - result2["u"]) < 1e-14
    assert np.linalg.norm(result1["v"] - result1["v"]) < 1e-14
