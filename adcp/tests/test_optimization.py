import pytest
import numpy as np

import adcp.optimization as op
import adcp
import scipy.sparse


@pytest.mark.skip("Not yet implemented")
class TestRescaling:
    def test_time_rescale_order_v2c2otg(self):
        prob = adcp.GliderProblem()
        assert prob is not None


def test_kalman_factor(standard_prob):
    km1 = op.gen_kalman_mat(
        standard_prob.data,
        standard_prob.config,
        standard_prob.shape,
        standard_prob.weights,
        root=False,
    )
    M = op.gen_kalman_mat(
        standard_prob.data,
        standard_prob.config,
        standard_prob.shape,
        standard_prob.weights,
        root=True,
    )
    km2 = M.T @ M
    assert (
        np.linalg.norm((km1 - km2).todense()) / np.linalg.norm(km1.todense())
        < 1e-15
    )


def test_limited_inversion_dividend():
    mat, v_cols, c_cols = op._limited_inversion_dividend(
        20, 20, 2, 2, "otg", 2
    )

    m = 20
    n = 20
    current_order = 1
    vehicle_order = 2

    interesting_sections = np.arange(0.05, 0.95, 0.05)

    n_rows = 2 * m * vehicle_order + 2 * n * current_order
    mat_shape = (n_rows, vehicle_order * 5 + current_order * 5)

    v_mats = []
    c_mats = []

    for section in interesting_sections:
        v_mats.append(
            scipy.sparse.eye(
                mat_shape[0],
                1,
                -np.floor(section * vehicle_order * m),
            )
        )
        c_mats.append(
            scipy.sparse.eye(
                mat_shape[0],
                1,
                -2 * m * vehicle_order - np.floor(section * current_order * n),
            )
        )
    expected = scipy.sparse.hstack(v_mats + c_mats)

    assert mat.shape == expected.shape

    assert not (mat != expected).todense().any()

    # Each column has only a single nonzero entry, equal to one
    assert (mat.sum(axis=0) == np.ones(mat.shape[1])).all()
