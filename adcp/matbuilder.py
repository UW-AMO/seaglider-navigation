# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 21:25:07 2019

@author: 600301
"""

import scipy.sparse
import scipy.interpolate
import numpy as np
import pandas as pd

from . import dataprep as dp

def uv_select(times, depths, ddat):
    """Creates the matrices that will select the appropriate V_otg
    and current  values for comparing with hydrodynamic model uv measurements.
    
    Parameters:
        times ([numpy.datetime64,]) : all of the sample times to predict
            V_otg for.  returned by dataprep.timepoints()
        ddat (dict): the recorded dive data returned by load_dive()
        
    Returns:
        tuple of nupmy arrays.  The first multiplies the V_otg vector,
        the second multiplies the current vector
    """
    uv_times = ddat['uv'].index.unique().to_numpy()
    idxuv = [k for k, t in enumerate(times) if t in uv_times]
    mat_shape = (len(idxuv), len(times))
    A = scipy.sparse.coo_matrix((np.ones(len(idxuv)),
                                 (np.ones(len(idxuv)),idxuv)),
                                shape=mat_shape)
    
    uv_depths = dp._depth_interpolator(times, ddat).loc[uv_times, 'depth']
    uv_depths = uv_depths.to_numpy().flatten()
    d_list = list(depths)
    idxdepth = [d_list.index(d) for d in uv_depths]
    mat_shape = (len(idxdepth), len(depths))
    B = scipy.sparse.coo_matrix((np.ones(len(idxdepth)),
                                 (range(len(idxdepth)),idxdepth)),
                                shape=mat_shape)
    return A, B

def adcp_select(times, depths, ddat, adat):
    """Creates the matrices that will select the appropriate V_otg
    and current  values for comparing with ADCP measurements.
    
    Parameters:
        times ([numpy.datetime64,]) : all of the sample times to predict
            V_otg for.  returned by dataprep.timepoints()
        ddat (dict): the recorded dive data returned by load_dive()
        adat (dict): the recorded ADCP data returned by load_adcp()
        
    Returns:
        tuple of nupmy arrays.  The first multiplies the V_otg vector,
        the second multiplies the current vector
    """
    adcp_times = np.unique(adat['time'])
    idxadcp = [k for k, t in enumerate(times) if t in adcp_times]
    valid_obs = np.isfinite(adat['UV'])
    obs_per_t = valid_obs.sum(axis=0)
    col_idx = [np.repeat(idx, n_obs) for idx, n_obs in zip(idxadcp, obs_per_t)]
    col_idx = [el for arr in col_idx for el in arr] #unpack list o'lists
    mat_shape = (len(col_idx), len(times))
    A = scipy.sparse.coo_matrix((np.ones(len(col_idx)),
                                 (range(len(col_idx)),col_idx)),
                                shape=mat_shape)

    depth_df = dp._depth_interpolator(times, ddat)
    turnaround = depth_df.ascending.idxmax()
    deepest = depth_df.loc[turnaround, 'depth']
    rising_times = adat['time'] > turnaround
    adcp_depths = adat['Z']
    print(adcp_depths[0,420])
    adcp_depths[:,rising_times] = 2*deepest - adat['Z'][:,rising_times]
    print('I\'m here')
    print(adcp_depths[0,420])
    d_list = list(depths)
    print(adcp_depths[0,420])
    idxdepth = [d_list.index(d) for d in adcp_depths[valid_obs]]
    mat_shape = (len(idxdepth), len(depths))
    B = scipy.sparse.coo_matrix((np.ones(len(idxdepth)),
                                 (range(len(idxdepth)), idxdepth)),
                                shape=mat_shape)
    return A, B


#def _depth_func(ddat):
#    """Provides interpolation function for vehicle depths at any time."""
#    depth_times = ddat['depth'].index.to_numpy()
#    depths = ddat['depth'].depth.to_numpy()
#    depth_func = scipy.interpolate.interp1d(depth_times, depths, fill_value=0)
#    return depth_func