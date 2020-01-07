# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 13:00:31 2020

@author: 600301
"""
import random

import numpy as np
import pandas as pd
import scipy.sparse

from . import matbuilder as mb

def simulate(duration = pd.Timedelta('3 hours'), max_depth = 750, n_dives = 1,
             n_timepoints=1001, rho_c=.03, rho_t=.1, rho_a=1, rho_g=1, rho_r=1,
             adcp_bins=4):
    """Simulates the data recorded during a dive.
    
    Returns:
        tuple of dictionaries.  The first element of the tuple is the dive
        data, as would be loaded from adcp.dataprep.load_dive(dive_num).
        The second element of the tuple is the ADCP data, as would be
        loaded from adcp.dataprep.load_adcp(dive_num).
    """
    start_time = pd.Timestamp('2020-01-01')
    depths_down = np.arange(0,max_depth, max_depth / (n_timepoints//2))
    depths_up = max_depth + np.arange(.1,max_depth, max_depth / (n_timepoints//2))
    timepoints = np.linspace(start_time.value, (start_time+duration).value, 
                           n_timepoints)
    timepoints = pd.to_datetime(timepoints)
    v_ttw_n = 1 + .5 * np.random.normal(size = n_timepoints)
    v_ttw_e = .2 + .5 * np.random.normal(size = n_timepoints)

    range_times = pd.to_datetime(sorted(
                    random.sample(set(timepoints[1:-1]), 3)))    
    gps_times = timepoints[[0,-1]]

    adcp_times = pd.to_datetime(sorted(
                    random.sample(set(timepoints[1:-1])-set(range_times), 
                                  (n_timepoints-5)//3)))
    ttw_times = set(timepoints[1:-1]) - set(adcp_times) - set(range_times)
    ttw_times = pd.to_datetime(sorted(ttw_times))

    midpoint = timepoints[n_timepoints//2+1]    
    if n_timepoints % 2 == 1:
        depths = np.concatenate((depths_down.tolist(), [max_depth], 
                                 depths_up.tolist()))
    else:
        depths = np.concatenate((depths_down.tolist(), depths_up.tolist()))

    depth_df = pd.DataFrame(depths, index=timepoints, columns = ['depth']) 
    down_adcp_times = adcp_times[adcp_times <=midpoint]
    down_adcp_points = depth_df.loc[down_adcp_times
                                    ].depth.to_numpy().reshape((-1,1))
    down_adcp_depths = 12*np.random.random_sample((len(down_adcp_points), adcp_bins))
    down_adcp_depths = down_adcp_points - np.sort(down_adcp_depths, axis=1)
    down_adcp_depths[down_adcp_depths < 0] = np.nan

    up_adcp_times = adcp_times[adcp_times > midpoint]
    up_adcp_points = depth_df.loc[up_adcp_times
                                  ].depth.to_numpy().reshape((-1,1))
    up_adcp_depths = 12*np.random.random_sample((len(up_adcp_points), adcp_bins))
    up_adcp_depths = up_adcp_points + np.sort(up_adcp_depths, axis=1)
    up_adcp_depths[up_adcp_depths > 2* max_depth] = np.nan

    all_depths = np.concatenate((down_adcp_depths.flatten(), 
                                 up_adcp_depths.flatten(),
                                 depths))
    all_depths = all_depths[np.isfinite(all_depths)]
    all_depths.sort()

    ### Simulate current at all depths
    Qc = mb.depth_Q(all_depths, rho_c)
    first_n = .5 * np.random.normal(size=1)
    first_e = .5 * np.random.normal(size=1)
    delta_C_n = np.random.multivariate_normal(
                        mean=np.zeros(len(all_depths)-1),
                        cov = Qc.todense())
    delta_C_e = np.random.multivariate_normal(
                        mean=np.zeros(len(all_depths)-1),
                        cov = Qc.todense())
    cum_diff_n = np.cumsum(delta_C_n)
    cum_diff_e = np.cumsum(delta_C_e)
    current_n = np.concatenate((first_n, first_n+cum_diff_n))
    current_e = np.concatenate((first_e, first_n+cum_diff_e))

    curr_df = pd.DataFrame(data = np.hstack([current_e.reshape((-1,1)),
                                             current_n.reshape((-1,1))]), 
                            index=pd.Index(all_depths, name='depth'),
                            columns=['curr_e','curr_n'])

    ### Simulate V_otg and X_otg
    depth_index = depth_df.loc[timepoints].depth.to_numpy()
    v_df = curr_df.loc[pd.Index(depth_index, name='depth')]
    v_df = v_df.reset_index()
    v_df['time'] = timepoints
    v_df['ttw_e'] = v_ttw_e
    v_df['ttw_n'] = v_ttw_n
    v_df['v_otg_n'] = v_df['curr_n']+v_df['ttw_n']
    v_df['v_otg_e'] = v_df['curr_e']+v_df['ttw_e']
    delta_t = timepoints[1:]-timepoints[:-1]
    deltax_otg_n = (delta_t * v_df.v_otg_n.to_numpy()[1:]).total_seconds()
    deltax_otg_e = (delta_t * v_df.v_otg_e.to_numpy()[1:]).total_seconds()
    v_df['dx_otg_n'] = np.nan
    v_df['dx_otg_e'] = np.nan
    v_df.loc[timepoints[1:],'dx_otg_n'] = deltax_otg_n
    v_df.loc[timepoints[1:],'dx_otg_e'] = deltax_otg_e

    ### Calculate & simulate z_ttw, z_adcp, and z_gps
    v_df = v_df.set_index('time')
    z_ttw_n = np.random.normal(loc = v_df.loc[ttw_times, 'ttw_n'], scale=rho_t)
    z_ttw_e = np.random.normal(loc = v_df.loc[ttw_times, 'ttw_e'], scale=rho_t)
    def n_curr_selector(depth):
        return np.nan if np.isnan(depth) else curr_df.loc[depth,'curr_n']    
    def e_curr_selector(depth):
        return np.nan if np.isnan(depth) else curr_df.loc[depth,'curr_e']
    cse = np.vectorize(e_curr_selector)
    csn = np.vectorize(n_curr_selector)

    adcp_depths = np.vstack([down_adcp_depths, up_adcp_depths])
    n_adcp_mean = np.array([v_df.loc[adcp_times, 'v_otg_n'] 
                            + csn(adcp_depths[:,i]) 
                            for i in range(adcp_bins)]).T
    e_adcp_mean = np.array([v_df.loc[adcp_times, 'v_otg_e'] 
                            + cse(adcp_depths[:,i]) 
                            for i in range(adcp_bins)]).T
    z_adcp_n = np.random.normal(loc = n_adcp_mean.astype(float), scale = rho_a)
    z_adcp_e = np.random.normal(loc = e_adcp_mean.astype(float), scale = rho_a)

    dxn = deltax_otg_n[-1]
    dxe = deltax_otg_e[-1]
    z_gps_n = np.random.normal(loc = dxn, scale = rho_g)
    z_gps_e = np.random.normal(loc = dxe, scale = rho_g)

    ## Simulate range
    range_posits = np.random.random_sample((3,2)) * max(dxe, dxn)
    ranges = np.sqrt((range_posits[:,0]-v_df.loc[range_times, 'dx_otg_e'])**2
                     +(range_posits[:,1]-v_df.loc[range_times, 'dx_otg_n'])**2)
    z_range = np.random.normal(loc=ranges, scale=rho_r)

    ### Construct dataframes & dictionaries for dive data
    gps_df = pd.DataFrame([[0,0],[z_gps_e, z_gps_n]],
                          index=pd.Index(gps_times, name='time'),
                          columns=['gps_nx_east','gps_ny_north'])
    range_data = np.hstack((range_posits, z_range.reshape((-1,1))))
    range_df = pd.DataFrame(range_data,
                          index=pd.Index(range_times, name='time'),
                          columns=['src_pos_e','src_pos_n','range'])
    ttw_df = pd.DataFrame(np.vstack([z_ttw_n, z_ttw_e]).T, 
                          index=pd.Index(ttw_times, name='time'),
                          columns=['u_north','v_east'])

    ddat = {'gps':gps_df, 'depth':depth_df.loc[
            gps_times.union(range_times).union(ttw_times).sort_values()],
            'uv':ttw_df, 'range':range_df}

    ### Construct dataframes & dictionaries for ADCP data
    adat = {'time':adcp_times.to_numpy(), 
            'Z': np.vstack((down_adcp_depths, 2*max_depth - (up_adcp_depths))),
            'UV': (z_adcp_e+z_adcp_n*1j).T}

    return adat, ddat