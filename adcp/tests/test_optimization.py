import pytest
import numpy as np

import adcp.optimization as op
import adcp


@pytest.mark.skip("Not yet implemented")
class TestRescaling:
    def test_time_rescale_order_v2c2otg(self):
        prob = adcp.GliderProblem()
        assert prob is not None


def test_kalman_factor(standard_prob):
    km1 = op.gen_kalman_mat(standard_prob, root=False)
    M = op.gen_kalman_mat(standard_prob, root=True)
    km2 = M.T @ M
    assert (
        np.linalg.norm((km1 - km2).todense()) / np.linalg.norm(km1.todense())
        < 1e-15
    )
