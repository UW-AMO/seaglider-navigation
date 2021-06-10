import pytest
from adcp import simulation as sim, optimization as op


@pytest.fixture
def standard_sim():
    sp = sim.SimParams()
    ddat, adat, x, _, _ = sim.simulate(sp)
    yield ddat, adat, x


@pytest.fixture
def standard_prob(standard_sim):
    ddat, adat, _ = standard_sim
    yield op.GliderProblem(ddat=ddat, adat=adat)
