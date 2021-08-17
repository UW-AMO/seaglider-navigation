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
    result, cols = op._limited_inversion_dividend(10, 10, 2, 2, "otg")

    m = 10
    n = 10
    current_order = 1
    vehicle_order = 2

    interesting_sections = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    n_rows = 2 * m * vehicle_order + 2 * n * current_order
    mat_shape = (n_rows, vehicle_order * 5 + current_order * 5)

    v_mats = []
    c_mats = []

    for section in interesting_sections:
        v_mats.append(
            scipy.sparse.eye(
                mat_shape[0],
                vehicle_order,
                -np.floor(section * vehicle_order * m),
            )
        )
        c_mats.append(
            scipy.sparse.eye(
                mat_shape[0],
                current_order,
                -2 * m * vehicle_order - np.floor(section * current_order * n),
            )
        )
    expected = scipy.sparse.hstack(v_mats + c_mats)

    assert result.shape == expected.shape

    assert not (result != expected).todense().any()

    assert (result.sum(axis=0) == np.ones((15,))).all()

    expected_col_sum = np.vstack(
        (
            np.tile(np.array([[0, 0, 1, 1]]).T, (5, 1)),
            np.zeros((20, 1)),
            np.tile(np.array([[0, 1]]).T, (5, 1)),
            np.zeros((10, 1)),
        )
    ).reshape((-1, 1))
    assert (result.sum(axis=1) == expected_col_sum).all()

    assert (np.where(expected_col_sum > 0)[0] == np.array(cols)).all()
