import pytest
import numpy as np

from adcp.matbuilder import (
    vehicle_Q,
    vehicle_Qinv,
    vehicle_G,
    depth_Q,
    depth_Qinv,
    depth_G,
    legacy_select,
    vehicle_G_given_C,
)

# %%


class TestKalman:
    def test_vehicle_q_3rd_order_smoothing(self):
        times = np.array([1e8, 2e8])  # nanoseconds
        Q = vehicle_Q(times, rho=1, order=3, conditioner=None, t_scale=1)
        expected_Q = np.array(
            [
                [1e-1, 1 / 2 * 1e-2, 1 / 6 * 1e-3],
                [1 / 2 * 1e-2, 1 / 3 * 1e-3, 1 / 8 * 1e-4],
                [1 / 6 * 1e-3, 1 / 8 * 1e-4, 1 / 20 * 1e-5],
            ]
        )
        assert np.linalg.norm(Q.todense() - expected_Q) < 1e-15

    def test_vehicle_q_2nd_order_smoothing(self):
        times = np.array([1e8, 2e8])  # nanoseconds
        Q = vehicle_Q(times, rho=1, order=2, conditioner=None, t_scale=1)
        expected_Q = np.array(
            [
                [1e-1, 1 / 2 * 1e-2],
                [1 / 2 * 1e-2, 1 / 3 * 1e-3],
            ]
        )
        assert np.linalg.norm(Q.todense() - expected_Q) < 1e-15

    def test_vehicle_qinv_3rd_order_smoothing(self):
        times = np.array([1e8, 2e8])  # nanoseconds
        Qi = vehicle_Qinv(times, rho=1, order=3, conditioner=None, t_scale=1)
        expected_Qi = np.linalg.inv(
            np.array(
                [
                    [1e-1, 1 / 2 * 1e-2, 1 / 6 * 1e-3],
                    [1 / 2 * 1e-2, 1 / 3 * 1e-3, 1 / 8 * 1e-4],
                    [1 / 6 * 1e-3, 1 / 8 * 1e-4, 1 / 20 * 1e-5],
                ]
            )
        )
        assert np.linalg.norm(Qi.todense() - expected_Qi) < 1e-6  # cond=7e6

    def test_vehicle_qinv_2nd_order_smoothing(self):
        times = np.array([1e8, 2e8])  # nanoseconds
        Qi = vehicle_Qinv(times, rho=1, order=2, conditioner=None, t_scale=1)
        expected_Qi = np.array([[40, -600], [-600, 12000]])
        assert np.linalg.norm(Qi.todense() - expected_Qi) < 1e-11  # cond=1e3

    def test_vehicle_g_3rd_order_smoothing(self):
        times = np.array([1e8, 2e8])  # nanoseconds
        G = vehicle_G(times, order=3, conditioner=None, t_scale=1)
        expected_G = np.array(
            [
                [-1, 0, 0, 1, 0, 0],
                [-0.1, -1, 0, 0, 1, 0],
                [-0.005, -0.1, -1, 0, 0, 1],
            ]
        )
        assert np.linalg.norm(G.todense() - expected_G) < 1e-15

    def test_vehicle_g_2nd_order_smoothing(self):
        times = np.array([1e8, 2e8])  # nanoseconds
        G = vehicle_G(times, order=2, conditioner=None, t_scale=1)
        expected_G = np.array(
            [
                [-1, 0, 1, 0],
                [-0.1, -1, 0, 1],
            ]
        )
        assert np.linalg.norm(G.todense() - expected_G) < 1e-15

    def test_current_q_3rd_order_smoothing(self):
        depths = np.array([1.1, 1.2])  # nanoseconds
        Q = depth_Q(
            depths,
            rho=1,
            order=3,
            depth_rate=None,
            conditioner=None,
            t_scale=1,
        )
        expected_Q = np.array(
            [
                [1e-1, 1 / 2 * 1e-2],
                [1 / 2 * 1e-2, 1 / 3 * 1e-3],
            ]
        )
        assert np.linalg.norm(Q.todense() - expected_Q) < 1e-15

    def test_current_q_2nd_order_smoothing(self):
        depths = np.array([1.1, 1.2])  # nanoseconds
        Q = depth_Q(
            depths,
            rho=1,
            order=2,
            depth_rate=None,
            conditioner=None,
            t_scale=1,
        )
        expected_Q = np.array([[0.1]])
        assert np.linalg.norm(Q.todense() - expected_Q) < 1e-15

    @pytest.mark.skip("Not yet implemented")
    def test_current_q_ttw_2nd_order_smoothing(self):
        pass

    @pytest.mark.skip("Not yet implemented")
    def test_current_q_ttw_3rd_order_smoothing(self):
        pass

    def test_current_qinv_3rd_order_smoothing(self):
        depths = np.array([1.1, 1.2])  # nanoseconds
        Qi = depth_Qinv(
            depths,
            rho=1,
            order=3,
            depth_rate=None,
            conditioner=None,
            t_scale=1,
        )
        expected_Qi = np.array([[40, -600], [-600, 12000]])
        assert np.linalg.norm(Qi.todense() - expected_Qi) < 1e-10

    def test_current_qinv_2nd_order_smoothing(self):
        depths = np.array([1.1, 1.2])  # nanoseconds
        Qi = depth_Qinv(
            depths,
            rho=1,
            order=2,
            depth_rate=None,
            conditioner=None,
            t_scale=1,
        )
        expected_Qi = np.array([[10]])
        assert np.linalg.norm(Qi.todense() - expected_Qi) < 1e-13

    @pytest.mark.skip("Not yet implemented")
    def test_current_qinv_ttw_2nd_order_smoothing(self):
        pass

    @pytest.mark.skip("Not yet implemented")
    def test_current_qinv_ttw_3nd_order_smoothing(self):
        pass

    def test_current_g_3rd_order_smoothing(self):
        depths = np.array([1.1, 1.2])  # nanoseconds
        G = depth_G(depths, order=3, depth_rate=None, conditioner=None)
        expected_G = np.array(
            [
                [-1, 0, 1, 0],
                [-0.1, -1, 0, 1],
            ]
        )
        assert np.linalg.norm(G.todense() - expected_G) < 1e-15

    def test_current_g_2nd_order_smoothing(self):
        depths = np.array([1.1, 1.2])  # nanoseconds
        G = depth_G(depths, order=2, depth_rate=None, conditioner=None)
        expected_G = np.array([[-1.0, 1.0]])
        assert np.linalg.norm(G.todense() - expected_G) < 1e-15

    @pytest.mark.skip("Not yet implemented")
    def test_current_g_ttw_2nd_order_smoothing(self):
        pass

    @pytest.mark.skip("Not yet implemented")
    def test_current_g_ttw_3rd_order_smoothing(self):
        pass

    @pytest.mark.skip("broken")
    def test_legacy_select_ttw(self):
        L = legacy_select(
            1, 1, vehicle_order=3, current_order=3, vehicle_vel="ttw"
        )
        X = np.array([[0.1, 1, 10, 0.2, 2, 20, 0.3, 3, 30, 0.4, 4, 40]]).T
        result = L @ X
        expected = np.array([[4, 40, 6, 60, 3, 4]]).T
        assert np.linalg.norm(result - expected) < 1e-15


def test_g_given_c_2nd_order(simple_otg_cov_prob):
    prob = simple_otg_cov_prob
    result = vehicle_G_given_C(
        prob.data.times,
        prob.config.vehicle_order,
        prob.config.t_scale,
        prob.data.depths,
        prob.data.idx_vehicle,
        prob.config.vehicle_vel,
        prob.config.current_order,
    ).todense()

    expected = np.array(
        [
            [0.0, 0.0, -5.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    assert np.linalg.norm(result - expected) < 1e-15


def test_g_given_c_3nd_order(simple_otg_cov_prob_3rd_order):
    prob = simple_otg_cov_prob_3rd_order
    result = vehicle_G_given_C(
        prob.data.times,
        prob.config.vehicle_order,
        prob.config.t_scale,
        prob.data.depths,
        prob.data.idx_vehicle,
        prob.config.vehicle_vel,
        prob.config.current_order,
    ).todense()

    expected = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, -0.15, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -1.25, -5.0, 0.0, 0.0, 1.25, 5.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert np.linalg.norm(result - expected) < 1e-15
