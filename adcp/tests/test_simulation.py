# Standard library imports
import unittest
import pytest

# Third party imports
import numpy as np

# Local application/library-specific imports
import adcp.simulation as sim


class SimTest(unittest.TestCase):
    """Tests the simulation module"""

    def test_dive_profile(self):
        sim_params = sim.SimParams()
        depth_df = sim.gen_dive(sim_params)
        mean_depth = float(depth_df.mean())
        self.assertTrue(
            abs(mean_depth - sim_params.max_depth) / sim_params.max_depth < 0.1
        )

    def test_adcp_depths(self):
        sim_params = sim.SimParams()
        depth_df = sim.gen_dive(sim_params)
        midpoint = depth_df.index[sim_params.n_timepoints // 2 + 1]
        adcp_df = sim.gen_adcp_depths(depth_df, sim_params)
        descending = depth_df.index > midpoint
        ascending = depth_df.index <= midpoint
        desc = (depth_df[descending].values < adcp_df[descending].values).all()
        asc = (depth_df[ascending].values > adcp_df[ascending].values).all()
        self.assertTrue(desc and asc)

    def test_curr_sim_cos(self):
        sim_params = sim.SimParams(curr_method="cos")
        depth_df = sim.gen_dive(sim_params)
        adcp_df = sim.gen_adcp_depths(depth_df, sim_params)
        all_depths = np.unique(
            np.concatenate(
                (depth_df.values.flatten(), adcp_df.values.flatten())
            )
        )
        all_depths = np.array(sorted(all_depths))
        curr_df = sim.sim_current_profile(all_depths, sim_params, "cos")
        inner1 = sum(np.sin(all_depths) * curr_df["curr_e"]) / len(all_depths)
        inner2 = sum(np.sin(all_depths) * curr_df["curr_n"]) / len(all_depths)
        self.assertTrue(inner1 < 0.1 and inner2 < 0.1)

    #    def test_curr_sim_linear(self):
    #        raise NotImplementedError
    #    def test_vehicle_sim_sin(self):
    #        raise NotImplementedError
    #    def test_vehicle_sim_linear(self):
    #        raise NotImplementedError

    @pytest.mark.skip("Not yet implemented")
    def test_vehicle_sim_constant(self):
        sim_params = sim.SimParams(vehicle_method="constant", rho_v=0)
        depth_df = sim.gen_dive(sim_params)
        adcp_df = sim.gen_adcp_depths(depth_df, sim_params)
        all_depths = np.unique(
            np.concatenate(
                (depth_df.values.flatten(), adcp_df.values.flatten())
            )
        )
        curr_df = sim.sim_current_profile(all_depths, sim_params, "constant")
        v_df = sim.sim_vehicle_path(depth_df, curr_df, sim_params, "constant")
        assert v_df is not None
        assert curr_df is not None


#    def test_select_times(self):
#        raise NotImplementedError
#    def test_sim_noise(self):
#        raise NotImplementedError
#    def test_construct_load_dict(self):
#        raise NotImplementedError
#    def test_true_solution(self):
#        raise NotImplementedError
#    def test_simulate(self):
#        raise NotImplementedError

if __name__ == "__main__":
    unittest.main()
