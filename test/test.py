"""
============================
Distribution Testing Module.
============================


"""
# Standard library imports
import unittest
import os
# Third party imports
import pandas as pd
import numpy as np
# Local application/library-specific imports
import adcp.matbuilder as mb
import adcp.simulation as sim

class SimTest(unittest.TestCase):
    """Tests the simulation module"""
    def test_defaults(self):
        old_defaults = (pd.Timedelta('3 hours'),750,1,1001,.1,.1,1,
                                  1,1,1,4,124,1,1,
                                  {'gps': 'endpoints', 'ttw': 0.6, 
                                       'range': 0.1})
        sim_params = sim.SimParams()
        self.assertEqual(sim_params, old_defaults)
    def test_dive_profile(self):
        sim_params = sim.SimParams()
        depth_df = sim.gen_dive(sim_params)
        mean_depth = float(depth_df.mean())
        self.assertTrue(abs(mean_depth-sim_params.max_depth) /
                        sim_params.max_depth<.1)
    def test_adcp_depths(self):
        sim_params = sim.SimParams()
        depth_df = sim.gen_dive(sim_params)
        midpoint = depth_df.index[sim_params.n_timepoints//2+1]
        adcp_df = sim.gen_adcp_depths(depth_df, sim_params)
        descending = depth_df.index > midpoint
        ascending = depth_df.index <= midpoint
        desc = (depth_df[descending].values < adcp_df[descending].values).all()
        asc = (depth_df[ascending].values > adcp_df[ascending].values).all()
        self.assertTrue(desc and asc)
    def test_curr_sim_cos(self):
        sim_params = sim.SimParams()
        depth_df = sim.gen_dive(sim_params)
        adcp_df = sim.gen_adcp_depths(depth_df, sim_params)
        all_depths = np.unique(np.concatenate((depth_df.values.flatten(),
                                               adcp_df.values.flatten())))
        all_depths = np.array(sorted(all_depths))
        curr_df = sim.sim_current_profile(all_depths, sim_params, 'cos')
        inner1 = sum(np.sin(all_depths)*curr_df['curr_e'])/len(all_depths)
        inner2 = sum(np.sin(all_depths)*curr_df['curr_n'])/len(all_depths)
        self.assertTrue(inner1 < .1 and inner2 < .1)
#    def test_curr_sim_linear(self):
#        raise NotImplementedError
#    def test_vehicle_sim_sin(self):
#        raise NotImplementedError
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

if __name__ == '__main__':
    unittest.main()
