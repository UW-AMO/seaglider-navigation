import pytest
from adcp.optimization import GliderProblem


@pytest.mark.skip("Not yet implemented")
class TestRescaling:
    def test_time_rescale_order_v2c2otg(self):
        prob = GliderProblem()
        assert prob is not None
