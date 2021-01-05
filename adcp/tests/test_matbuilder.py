import pytest
import numpy as np

from adcp.matbuilder import (
    vehicle_Q,
    vehicle_Qinv,
    vehicle_G,
    # depth_Q,
    # depth_Qinv,
    # depth_G,
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
        expected_Qi = np.linalg.inv(
            np.array(
                [
                    [1e-1, 1 / 2 * 1e-2],
                    [1 / 2 * 1e-2, 1 / 3 * 1e-3],
                ]
            )
        )
        assert np.linalg.norm(Qi.todense() - expected_Qi) < 1e-12  # cond=1e3

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

    @pytest.mark.skip("Not yet implemented")
    def test_current_q_3rd_order_smoothing(self):
        pass

    @pytest.mark.skip("Not yet implemented")
    def test_current_q_2nd_order_smoothing(self):
        pass

    @pytest.mark.skip("Not yet implemented")
    def test_current_q_ttw_2nd_order_smoothing(self):
        pass

    @pytest.mark.skip("Not yet implemented")
    def test_current_qinv_3rd_order_smoothing(self):
        pass

    @pytest.mark.skip("Not yet implemented")
    def test_current_qinv_2nd_order_smoothing(self):
        pass

    @pytest.mark.skip("Not yet implemented")
    def test_current_qinv_ttw_2nd_order_smoothing(self):
        pass

    @pytest.mark.skip("Not yet implemented")
    def test_current_g_3rd_order_smoothing(self):
        pass

    @pytest.mark.skip("Not yet implemented")
    def test_current_g_2nd_order_smoothing(self):
        pass

    @pytest.mark.skip("Not yet implemented")
    def test_current_g_ttw_2nd_order_smoothing(self):
        pass
