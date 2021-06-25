import random
import numpy as np
import pytest
from adcp.exp.psearch2d import ParameterSearch2D
from adcp.exp.psearch2d_trials import (
    trial2,
    trial2c,
    trial12,
    trial12c,
    trial16,
    trial16c,
)


@pytest.mark.slow
def test_default_integration_trial2a():
    experiment = ParameterSearch2D(
        **{**trial2.prob_params, **{"rho_vs": [1e-2]}, **{"rho_cs": [1e-1]}},
        **trial2.sim_params,
    )
    run = experiment.run(visuals=False)
    results = run["metrics"]
    global x
    x = run["x"]

    assert np.random.get_state()[0] == "MT19937"
    assert np.random.get_state()[1][0] == 3115277058
    assert random.getstate()[0] == 3
    assert random.getstate()[1][0] == 727647865
    assert 5176938 < results[0] < 5176939
    assert 3.6809045 < results[1] < 3.6809046


@pytest.mark.slow
def test_default_integration_trial2c():
    experiment = ParameterSearch2D(
        **{**trial2c.prob_params, **{"rho_vs": [1e-3]}, **{"rho_cs": [1e-2]}},
        **trial2c.sim_params,
    )

    run = experiment.run(visuals=False)
    results = run["metrics"]
    global y
    y = run["x"]

    assert np.random.get_state()[0] == "MT19937"
    assert np.random.get_state()[1][0] == 3115277058
    assert random.getstate()[0] == 3
    assert random.getstate()[1][0] == 727647865
    assert 2143393 < results[0] < 2143394
    assert 15.602947 < results[1] < 15.602948


@pytest.mark.slow
def test_default_integration_trial12():
    experiment = ParameterSearch2D(
        **{**trial12.prob_params, **{"rho_vs": [1e-7]}, **{"rho_cs": [1e-6]}},
        **trial12.sim_params,
    )
    run = experiment.run(visuals=False)
    results = run["metrics"]
    global x
    assert all(x == run["x"])

    assert np.random.get_state()[0] == "MT19937"
    assert np.random.get_state()[1][0] == 3115277058
    assert random.getstate()[0] == 3
    assert random.getstate()[1][0] == 727647865
    assert 2053767 < results[0] < 2053768
    assert 4.1859312 < results[1] < 4.1859313


@pytest.mark.slow
def test_default_integration_trial12c():
    experiment = ParameterSearch2D(
        **{**trial12c.prob_params, **{"rho_vs": [1e-6]}, **{"rho_cs": [1e-3]}},
        **trial12c.sim_params,
    )
    run = experiment.run(visuals=False)
    results = run["metrics"]
    global y
    assert all(y == run["x"])

    assert np.random.get_state()[0] == "MT19937"
    assert np.random.get_state()[1][0] == 3115277058
    assert random.getstate()[0] == 3
    assert random.getstate()[1][0] == 727647865
    assert 4113853 < results[0] < 4113854
    assert 16.947697 < results[1] < 16.947698


@pytest.mark.slow
def test_default_integration_trial16():
    experiment = ParameterSearch2D(
        **{**trial16.prob_params, **{"rho_vs": [1e-6]}, **{"rho_cs": [1e-2]}},
        **trial16.sim_params,
    )
    results = experiment.run(visuals=False)["metrics"]

    assert np.random.get_state()[0] == "MT19937"
    assert np.random.get_state()[1][0] == 3115277058
    assert random.getstate()[0] == 3
    assert random.getstate()[1][0] == 727647865
    assert 2052558 < results[0] < 2052559
    assert 4.00 < results[1] < 4.01


@pytest.mark.slow
def test_default_integration_trial16c():
    experiment = ParameterSearch2D(
        **{**trial16c.prob_params, **{"rho_vs": [1e-6]}, **{"rho_cs": [1e-5]}},
        **trial16c.sim_params,
    )
    results = experiment.run(visuals=False)["metrics"]

    assert np.random.get_state()[0] == "MT19937"
    assert np.random.get_state()[1][0] == 3115277058
    assert random.getstate()[0] == 3
    assert random.getstate()[1][0] == 727647865
    assert 25834296 < results[0] < 25834297
    assert 136 < results[1] < 137
