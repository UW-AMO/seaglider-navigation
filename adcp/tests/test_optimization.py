import pytest

from adcp.optimization import GliderProblem, time_rescale


class TestRescaling:
    def test_time_rescale_order_v2c2otg(self):
        prob = GliderProblem()
