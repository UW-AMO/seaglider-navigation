from adcp.optimization import GliderProblem


class TestRescaling:
    def test_time_rescale_order_v2c2otg(self):
        prob = GliderProblem()
        assert prob is not None
