from adcp import dataprep as dp


def test_create_ProblemData(standard_sim):
    ddat, adat, _ = standard_sim
    dd = dp.ProblemData(ddat, adat)
    result = dd.interpolator.shape
    assert result is not None
