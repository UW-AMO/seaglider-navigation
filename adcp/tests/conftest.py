import pytest
from adcp import simulation as sim
import adcp


@pytest.fixture(name="standard_sim")
def standard_sim_fixture():
    yield standard_sim()


def standard_sim():
    sp = sim.SimParams()
    ddat, adat, x, _, _ = sim.simulate(sp)
    return ddat, adat, x


@pytest.fixture
def standard_prob_fixture(standard_sim, name="standard_prob"):
    yield standard_prob(standard_sim)


def standard_prob(sim=None):
    if sim is None:
        sim = standard_sim()
    ddat, adat, _ = sim
    data = adcp.ProblemData(ddat, adat)
    return adcp.GliderProblem(
        data, adcp.ProblemConfig(), adcp.Weights(1, 1, 1, 1, 1)
    )
