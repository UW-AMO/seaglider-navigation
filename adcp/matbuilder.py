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

t_scale = 1e3
"""timescale, 1=seconds, 1e-3=milliseconds, 1e3=kiloseconds.  Used to
control condition number of problem.
"""
conditioner = 'tanh'

# %% Vector item selection
def uv_select(times, depths, ddat):
    """Creates the matrices that will select the appropriate V_otg
    and current values for comparing with hydrodynamic model
    uv measurements.
    
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
                                 (range(len(idxuv)),idxuv)),
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
    and current values for comparing with ADCP measurements.
    
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
    rising_times = pd.to_datetime(adat['time']) > turnaround
    adcp_depths = adat['Z'].copy()
    adcp_depths[:,rising_times] = 2*deepest - adat['Z'][:,rising_times]
    d_list = list(depths)
    idxdepth = [d_list.index(d) for d in adcp_depths[valid_obs]]
    mat_shape = (len(idxdepth), len(depths))
    B = scipy.sparse.coo_matrix((np.ones(len(idxdepth)),
                                 (range(len(idxdepth)), idxdepth)),
                                shape=mat_shape)
    return A, B

def gps_select(times, ddat):
    """Creates the matrix that will select the appropriate position
    for comparing with GPS measurements
    
    Parameters:
        times ([numpy.datetime64,]) : all of the sample times to predict
            V_otg for.  returned by dataprep.timepoints()
        ddat (dict): the recorded dive data returned by load_dive()
        
    Returns:
        Nupmy array to multiply the X_otg vector.
    """
    gps_times = ddat['gps'].index.unique().to_numpy()
    idxgps = [k for k, t in enumerate(times) if t in gps_times]
    mat_shape = (len(idxgps), len(times))
    A = scipy.sparse.coo_matrix((np.ones(len(idxgps)),
                                 (range(len(idxgps)),idxgps)),
                                shape=mat_shape)
    return A

def range_select(times, ddat):
    """Creates the matrix that will select the appropriate position
    for comparing with range measurements
    
    Parameters:
        times ([numpy.datetime64,]) : all of the sample times to predict
            V_otg for.  returned by dataprep.timepoints()
        ddat (dict): the recorded dive data returned by load_dive()
        
    Returns:
        Nupmys arrays to multiply the X_otg vectors.
    """
    range_times = ddat['range'].index.unique().to_numpy()
    idxrange = [k for k, t in enumerate(times) if t in range_times]
    mat_shape = (len(idxrange), len(times))
    A = scipy.sparse.coo_matrix((np.ones(len(idxrange)),
                                 (range(len(idxrange)),idxrange)),
                                shape=mat_shape)
    return A

# %% Kalman Process Matrices
def reduce_condition(deltas, method=None, minmax_ratio=.2):
    avg = deltas.mean()
    if method is None or method.lower()=='none':
        return deltas
    elif method.lower()=='avg':
        return avg*np.ones(len(deltas))
    elif method.lower()=='tanh':
        factor= (1-minmax_ratio)/(1+minmax_ratio)*avg
        return factor*np.tanh(deltas) + avg
    elif method.lower()=='sux':
        return np.ones(deltas)
    else:
        raise Exception('reduce_condition received bad method parameter')
def vehicle_Qinv(times, rho=1):
    """Creates the precision matrix for smoothing the vehicle with velocity
    covariance rho.
    """
    delta_times = times[1:]-times[:-1]
    dts = delta_times.astype(float)/1e9/t_scale
    dts = reduce_condition(dts, method=conditioner)
    Qs = [t_scale**3*rho*np.array([[dt, dt**2/2],[dt**2/2, dt**3/3]]) for dt in dts]
    Qinvs = [np.linalg.inv(Q) for Q in Qs]
    return scipy.sparse.block_diag(Qinvs)

def vehicle_Q(times, rho=1):
    """Creates the covariance matrix for smoothing the vehicle with velocity
    covariance rho.
    """
    delta_times = times[1:]-times[:-1]
    dts = delta_times.astype(float)/1e9/t_scale
    dts = reduce_condition(dts, method=conditioner)
    Qs = [t_scale**3*rho*np.array([[dt, dt**2/2],[dt**2/2, dt**3/3]]) for dt in dts]
    return scipy.sparse.block_diag(Qs)

def vehicle_G(times):
    """Creates the update matrix for smoothing the vehicle"""
    delta_times = times[1:]-times[:-1]
    m = len(delta_times)*2
    dts = delta_times.astype(float)/1e9/t_scale
    dts = reduce_condition(dts, method=conditioner)
    negGs = [np.array([[-1, 0],[-dt, -1]]) for dt in dts]
    negG = scipy.sparse.block_diag(negGs)
    posG = scipy.sparse.eye(m)
    append_me = scipy.sparse.coo_matrix(([],([],[])), (m,2))
    return (scipy.sparse.hstack((negG, append_me))
            + scipy.sparse.hstack((append_me,posG)))

def depth_Qinv(depths, rho=1):
    """Creates the precision matrix for smoothing the currint with depth
    covariance eta.
    """
    delta_depths = depths[1:]-depths[:-1]
    dds = reduce_condition(delta_depths, method=conditioner)
    return t_scale**(-2)/rho*scipy.sparse.diags(1/dds, dtype=float)

def depth_Q(depths, rho=1):
    """Creates the covariance matrix for smoothing the currint with depth
    covariance eta.
    """
    delta_depths = depths[1:]-depths[:-1]
    dds = reduce_condition(delta_depths, method=conditioner)
    return t_scale**(2)*rho*scipy.sparse.diags(dds, dtype=float)

def depth_G(depths):
    """Creates the update matrix for smoothing the current"""
    length = len(depths)
    return scipy.sparse.spdiags((-np.ones(length), np.ones(length)),
                                diags=(0,1), m=length-1, n=length)

# %% Data selection
def get_zttw(ddat, direction='north'):
    """Select the hydrodynamic measurements vector in the specified 
    direction.
    """
    if direction in ['u','north', 'u_north']:
        return ddat['uv'].u_north*t_scale
    elif direction in ['v','east', 'v_east']:
        return ddat['uv'].v_east*t_scale
    else:
        return None
    
def get_zadcp(adat, direction='north'):
    """Select the adcp measurements vector in the specified 
    direction.
    """
    valid_obs = np.isfinite(adat['UV'])
    if direction in ['u','north', 'u_north']:
        return adat['UV'][valid_obs].imag*t_scale
    elif direction in ['v','east', 'v_east']:
        return adat['UV'][valid_obs].real*t_scale
    else:
        return None
    
def get_zgps(ddat, direction='north'):
    """Select the GPS measurements vector in the specified 
    direction.
    """
    if direction in ['u','north', 'u_north', 'gps_ny_north']:
        return ddat['gps'].gps_ny_north.to_numpy()
    elif direction in ['v','east', 'v_east', 'gps_nx_east']:
        return ddat['gps'].gps_nx_east.to_numpy()
    else:
        return None

def get_zrange(ddat):
    """Select the range measurements vector as well as the range
    offset vectors
    
    Returns:
        Tuple of Numpy arrays. The first is the range measurement
        vector.  The second is the source position east of the origin.
        The third is the source position north of the origin.  All
        units in meters.
    """
    r = ddat['range'].range.to_numpy()
    y = ddat['range'].src_pos_n.to_numpy()
    x = ddat['range'].src_pos_e.to_numpy()
    return r, y, x

def v_select(timesteps):
    """Creates the matrix that selects the v entries of 
    [v1, x1, v2, x2, v3, ...].
    
    Parameters:
        timesteps (int) : number of timesteps
    """
    mat_shape = (timesteps, 2*timesteps)
    return scipy.sparse.coo_matrix((np.ones(timesteps),
                                    (range(timesteps), 
                                     range(0,2*timesteps, 2))),
                                    shape=mat_shape)

def x_select(timesteps):
    """Creates the matrix that selects the x entries of 
    [v1, x1, v2, x2, v3, ...].
    
    Parameters:
        timesteps (int) : number of timesteps
    """
    mat_shape = (timesteps, 2*timesteps)
    return scipy.sparse.coo_matrix((np.ones(timesteps),
                                    (range(timesteps), 
                                     range(1,2*timesteps, 2))),
                                    shape=mat_shape)