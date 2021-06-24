import pytest
from adcp.exp.psearch2d import ParameterSearch2D
from adcp.exp.psearch2d_trials import trial2


@pytest.mark.gitlocked
@pytest.mark.slow
def test_default_integration_trial2a():
    experiment = ParameterSearch2D(
        **{**trial2.prob_params, **{"rho_vs": [1e-2]}, **{"rho_cs": [1e-1]}},
        **trial2.sim_params,
    )
    results = experiment.run()["metrics"]

    assert 5176938 < results[0] < 5176939
    assert 3.6809045 < results[1] < 3.6809046
