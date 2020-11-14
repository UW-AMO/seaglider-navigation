# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 21:25:07 2019

@author: 600301
"""
import warnings

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
    idxdepth = [d_list.index(d) for d in adcp_depths.T[valid_obs.T]]
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
class ConditionWarning(Warning):
    def __init__(self, cond):
        self.msg = f'The condition number of covariance matrix is high: {cond}'

def q_cond(dts, dim=2):
    """Calculates the condition number of a block diagonal covariance matrix
    with intervals dts

    Returns:
        float: Condition number of Q
    """
    min_dt = min(dts)
    max_dt = max(dts)
    if dim == 3:
        arr_func = lambda dt: np.array([
            [dt,      dt**2/2, dt**3/6],
            [dt**2/2, dt**3/3, dt**4/8],
            [dt**3/6, dt**4/8, dt**5/20],
        ])
        max_arr = arr_func(max_dt)
        min_arr = arr_func(min_dt)
        max_eig = max(abs(np.linalg.eigvals(max_arr)))
        min_eig = min(abs(np.linalg.eigvals(min_arr)))
        return max_eig / min_eig
    elif dim == 2:
        max_disc = 1+max_dt**2/3+max_dt**4/9
        max_eig = 1/2 * (max_dt+max_dt **3/3 + max_dt * np.sqrt(max_disc))
        min_disc = 1+min_dt**2/3+min_dt**4/9
        min_eig = 1/2 * (min_dt+min_dt **3/3 - min_dt * np.sqrt(min_disc))
        return max_eig / min_eig
    elif dim == 1:
        return max_dt/min_dt
    else:
        raise ValueError('Can only calculate condition number of 1, 2, or 3'
                         'dimensional kalman smoothing covariance.')

def reduce_condition(deltas, method=None, minmax_ratio=.2):
    avg = deltas.mean()
    if method is None or method.lower()=='none':
        return deltas
    elif method.lower()=='avg':
        return avg*np.ones(len(deltas))
    elif method.lower()=='tanh':
        factor= minmax_ratio*avg
        max_deviation = max(*abs(deltas-avg),1e-3)
        #tanh most active in [-3,3]
        return factor*np.tanh(3*(deltas-avg)/max_deviation) + avg
    elif method.lower()=='sux':
        return np.ones(deltas)
    else:
        raise Exception('reduce_condition received bad method parameter')


def vehicle_Qblocks(times, rho=1, order=2, conditioner=conditioner, t_scale=t_scale):
    """Create the diagonal blocks of the kalman matrix for smoothing
    vehicle motion"""

    delta_times = times[1:]-times[:-1]
    dts = delta_times.astype(float)/1e9/t_scale
    dts = reduce_condition(dts, method=conditioner)
    cond = q_cond(dts, dim=order)
    if cond > 1e3:
        warnings.warn(ConditionWarning(cond))
    elif cond <1:
        raise RuntimeError('Calculated invalid condition number for Q')
    if order == 2:
        Qs = [t_scale**3*rho*np.array([
            [dt,      dt**2/2],
            [dt**2/2, dt**3/3]
        ]) for dt in dts]
    elif order == 3:
        Qs = [t_scale**5*rho*np.array([
            [dt,      dt**2/2, dt**3/6],
            [dt**2/2, dt**3/3, dt**4/8],
            [dt**3/6, dt**4/8, dt**5/20]
        ]) for dt in dts]
    else:
        raise ValueError
    return Qs

def vehicle_Qinv(times, rho=1, order=2, conditioner=conditioner, t_scale, t_scale):
    """Creates the precision matrix for smoothing the vehicle with velocity
    covariance rho.
    """

    Qs = vehicle_Qblocks(times, rho, order, conditioner, t_scale)
    Qinvs = [np.linalg.inv(Q) for Q in Qs]
    return scipy.sparse.block_diag(Qinvs)


def vehicle_Q(times, rho=1, order=2, conditioner=conditioner, t_scale):
    """Creates the covariance matrix for smoothing the vehicle with velocity
    covariance rho.
    """

    Qs = vehicle_Qblocks(times, rho, order, conditioner, t_scale)
    return scipy.sparse.block_diag(Qs)


def vehicle_G(times, order=2, conditioner=conditioner, t_scale=t_scale):
    """Creates the update matrix for smoothing the vehicle"""
    delta_times = times[1:]-times[:-1]
    dts = delta_times.astype(float)/1e9/t_scale # raw dts in nanoseconds
    dts = reduce_condition(dts, method=conditioner)
    if order == 2:
        negGs = [np.array([[-1, 0],[-dt, -1]]) for dt in dts]
    elif order == 3:
        negGs = [np.array(
            [[-1,       0,   0],
            [-dt,      -1,   0],
            [-dt**2/2, -dt, -1]]) for dt in dts]
    else:
        raise ValueError
    negG = scipy.sparse.block_diag(negGs)
    m = len(delta_times)*order
    posG = scipy.sparse.eye(m)
    append_me = scipy.sparse.coo_matrix(([],([],[])), (m,order))
    return (scipy.sparse.hstack((negG, append_me))
            + scipy.sparse.hstack((append_me,posG)))


def depth_Qblocks(depths, rho=1, order=2, depth_rate=None, conditioner=conditioner, t_scale=t_scale):
    """Create the diagonal blocks of the kalman matrix for smoothing
    current"""

    delta_depths = depths[1:]-depths[:-1]
    if depth_rate is None:
        order -= 1
        depth_rate = np.ones(len(delta_depths))
    elif order == 1:
        raise ValueError('If including depth_rate/modeling vttw, minimum ' \
            'order is 2.')
    dds = reduce_condition(delta_depths, method=conditioner)
    cond = q_cond(dds, dim=1)

    dr_neg1 = depth_rate ** -1
    dr_neg2 = depth_rate ** -2
    if cond > 1e3:
        warnings.warn(ConditionWarning(cond))
    elif cond <1:
        raise RuntimeError('Calculated invalid condition number for Q')
    if order == 1:
        Qs = [t_scale**2 * rho * np.array([[dd]]) for dd in dds]
    elif order == 2:
        Qs = [t_scale**2 * rho * np.array([
             [dd,            dd**2/2 * dr1],
             [dd**2/2 * dr1, dd**3/3 * dr2]
        ]) for dd, dr1, dr2 in zip(dds, dr_neg1, dr_neg2)]
    elif order == 3:
        Qs = [t_scale**2 * rho * np.array([
             [dd,            dd**2/2,       dd**3/6 * dr1],
             [dd**2/2,       dd**3/3,       dd**4/8 * dr1],
             [dd**3/6 * dr1, dd**4/8 * dr1, dd**5/20 * dr2]
        ]) for dd, dr1, dr2 in zip(dds, dr_neg1, dr_neg2)]
    else:
        raise ValueError
    return Qs


def depth_Qinv(depths, rho=1, order=2, depth_rate=None, conditioner=conditioner, t_scale=t_scale):
    """Creates the precision matrix for smoothing the currint with depth
    covariance rho.
    """
    Qs = depth_Qblocks(depths, rho, order, depth_rate, conditioner, t_scale)
    Qinvs = [np.linalg.inv(Q) for Q in Qs]
    return scipy.sparse.block_diag(Qinvs)


def depth_Q(depths, rho=1, order=2, depth_rate=None, conditioner=conditioner):
    """Creates the covariance matrix for smoothing the current with depth
    covariance rho.
    """
    Qs = depth_Qblocks(depths, rho, order, depth_rate, conditioner, t_scale)
    return scipy.sparse.block_diag(Qs, dtype=float)


def depth_G(depths, order=2, depth_rate=None, conditioner=conditioner):
    """Creates the update matrix for smoothing the current"""
    delta_depths = depths[1:]-depths[:-1]
    if depth_rate is None:
        order -= 1
        depth_rate = np.ones(len(delta_depths))
    elif order == 1:
        raise ValueError('If including depth_rate/modeling vttw, minimum ' \
            'order is 2.')
    dds = reduce_condition(delta_depths, method=conditioner)

    dr_neg1 = depth_rate ** -1
    if order == 1:
        negGs = [np.array([[-1]]) for dd in dds]
    elif order == 2:
        negGs = [np.array([
            [-1,        0],
            [-dd * dr1, -1]
        ]) for dd, dr1 in zip(dds, dr_neg1)]
    elif order == 3:
        negGs = [np.array([
            [-1,              0,        0],
            [-dd,      -1,        0],
            [-dd**2/2 * dr1, -dd * dr1, -1]
        ]) for dd, dr1 in zip(dds, dr_neg1)]
    else:
        raise ValueError
    negG = scipy.sparse.block_diag(negGs)
    m = len(delta_depths)*(order)
    posG = scipy.sparse.eye(m)
    append_me = scipy.sparse.coo_matrix(([],([],[])), (m,order))
    return (scipy.sparse.hstack((negG, append_me))
            + scipy.sparse.hstack((append_me,posG)))


# %% Data selection
def get_zttw(ddat, direction='north', t_scale=t_scale):
    """Select the hydrodynamic measurements vector in the specified 
    direction.
    """
    if direction in ['u','north', 'u_north']:
        return ddat['uv'].u_north*t_scale
    elif direction in ['v','east', 'v_east']:
        return ddat['uv'].v_east*t_scale
    else:
        return None
    
def get_zadcp(adat, direction='north', t_scale=t_scale):
    """Select the adcp measurements vector in the specified 
    direction.
    """
    valid_obs = np.isfinite(adat['UV'])
    if direction in ['u','north', 'u_north']:
        return adat['UV'].T[valid_obs.T].imag*t_scale
    elif direction in ['v','east', 'v_east']:
        return adat['UV'].T[valid_obs.T].real*t_scale
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
    x = ddat['range'].src_pos_e.to_numpy()
    y = ddat['range'].src_pos_n.to_numpy()
    return r, x, y


def size_of_x(m: int, n: int, vehicle_order: int=2, current_order: int=1
) -> int:
    """Calculates the size of the state vector

    Parameters:
        m (int) : number of timepoints
        n (int) : number of depthpoints
        vehicle_order (int) : order of vehicle smoothing in matrix Q
        current_order (int) : order of current smoothing in matrix Q
    """
    return 2 * (m * vehicle_order + n * current_order)


def a_select(m, order=3):
    """Creates the matrix that selects the acceleration entries of the
    state vector for the vehicle in one direction,
    e.g. [a1, v1, x1, a2, v2, x2, v3, ...].

    Parameters:
        m (int) : number of timesteps
        order (int) : order of vehicle smoothing. 2=velocity, 3=accel
            Must be greater than or equal to 3
    """
    mat_shape = (m, order*m)
    if order < 3:
        return None
    return scipy.sparse.coo_matrix((np.ones(m),
                                    (range(m),
                                     range(order-3,order*m, order))),
                                    shape=mat_shape)


def v_select(m, order=2):
    """Creates the matrix that selects the velocity entries of the
    state vector for the vehicle in one direction,
    e.g. [v1, x1, v2, x2, v3, ...].
    
    Parameters:
        m (int) : number of timesteps
        order (int) : order of vehicle smoothing. 2=velocity, 3=accel
    """
    mat_shape = (m, order*m)
    return scipy.sparse.coo_matrix((np.ones(m),
                                    (range(m),
                                     range(order-2,order*m, order))),
                                    shape=mat_shape)


def x_select(m, order=2):
    """Creates the matrix that selects the position entries of the
    state vector for the vehicle in one direction,
    e.g. [v1, x1, v2, x2, v3, ...].

    Parameters:
        m (int) : number of timepoints
        order (int) : order of vehicle smoothing. 2=velocity, 3=accel
    """
    mat_shape = (m, order*m)
    return scipy.sparse.coo_matrix((np.ones(m),
                                    (range(m),
                                     range(order-1,order*m, order))),
                                    shape=mat_shape)


def ca_select(n, order=3, vehicle_vel='otg'):
    """Creates the matrix that selects the current velocity entries of
    the state vector for the vehicle in one direction,
    e.g. [v1, x1, v2, x2, v3, ...].

    Parameters:
        n (int) : number of depthpoints
        order (int) : order of current smoothing. 2=velocity, 3=accel
        vehicle_vel (str) : whether vehicle velocity models through-the
            -water velocity or over-the-ground
    """

    if order < 3:
        return None
    if vehicle_vel == 'otg':
        order = order-1
        cols = range(order-2,order*n, order)
    elif vehicle_vel == 'ttw':
        cols = range(order-3,order*n, order)
    else:
        raise ValueError
    mat_shape = (n, order*n)
    return scipy.sparse.coo_matrix((np.ones(n),
                                    (range(n),
                                     cols)),
                                    shape=mat_shape)


def cv_select(n, order=2, vehicle_vel='otg'):
    """Creates the matrix that selects the current velocity entries of
    the state vector for the vehicle in one direction,
    e.g. [v1, x1, v2, x2, v3, ...].

    Parameters:
        n (int) : number of depthpoints
        order (int) : order of current smoothing. 2=velocity, 3=accel
        vehicle_vel (str) : whether vehicle velocity models through-the
            -water velocity or over-the-ground
    """

    if vehicle_vel == 'otg':
        order = order-1
        cols = range(order-1,order*n, order)
    elif vehicle_vel == 'ttw':
        cols = range(order-2,order*n, order)
    else:
        raise ValueError
    mat_shape = (n, order*n)
    return scipy.sparse.coo_matrix((np.ones(n),
                                    (range(n),
                                     cols)),
                                    shape=mat_shape)


def cx_select(n, order=2, vehicle_vel='otg'):
    """Creates the matrix that selects the current velocity entries of
    the state vector for the vehicle in one direction,
    e.g. [v1, x1, v2, x2, v3, ...].

    Parameters:
        n (int) : number of depthpoints
        order (int) : order of current smoothing. 2=velocity, 3=accel
        vehicle_vel (str) : whether vehicle velocity models through-the
            -water velocity or over-the-ground
    """
    if vehicle_vel == 'otg':
        raise ValueError('Current X position not modeled when vehicle velocity'
                        'is measured over ground.')
    elif vehicle_vel == 'ttw':
        cols = range(order-1,order*n, order)
    else:
        raise ValueError
    mat_shape = (n, order*n)
    return scipy.sparse.coo_matrix((np.ones(n),
                                    (range(n),
                                     cols)),
                                    shape=mat_shape)


def e_select(m, n, vehicle_order=2, current_order=2, vehicle_vel='otg'):
    """Creates a selection matrix for choosing indexes of X
    related to easterly variables.

    Parameters:
        m (int) : number of timepoints
        n (int) : number of depthpoints
        vehicle_order (int) : order of vehicle smoothing in matrix Q
        current_order (int) : order of current smoothing in matrix Q
        vehicle_vel (str) : if "otg", then current skips modeling 1st order
    """

    current_order -= vehicle_vel == 'otg'
    EV = ev_select(m, n, vehicle_order, current_order)
    EC = ec_select(m, n, vehicle_order, current_order)
    return scipy.sparse.vstack((EV, EC))


def n_select(m, n, vehicle_order=2, current_order=2, vehicle_vel='otg'):
    """Creates a selection matrix for choosing indexes of X
    related to easterly vehicle kinematics.

    Parameters:
        m (int) : number of timepoints
        n (int) : number of depthpoints
        vehicle_order (int) : order of vehicle smoothing in matrix Q
        current_order (int) : order of current smoothing in matrix Q
        vehicle_vel (str) : if "otg", then current skips modeling 1st order
    """

    current_order -= vehicle_vel == 'otg'
    NV = nv_select(m, n, vehicle_order, current_order)
    NC = nc_select(m, n, vehicle_order, current_order)
    return scipy.sparse.vstack((NV, NC))


def ev_select(m, n, vehicle_order=2, current_order=2, vehicle_vel='otg'):
    """Creates a selection matrix for choosing indexes of X
    related to easterly vehicle kinematics.
    
    Parameters:
        m (int) : number of timepoints
        n (int) : number of depthpoints
        vehicle_order (int) : order of vehicle smoothing in matrix Q
        current_order (int) : order of current smoothing in matrix Q
        vehicle_vel (str) : if "otg", then current skips modeling 1st order
    """

    current_order -= vehicle_vel == 'otg'
    n_rows = vehicle_order*m
    n_cols = size_of_x(m, n, vehicle_order, current_order)
    return scipy.sparse.eye(n_rows, n_cols)


def nv_select(m, n, vehicle_order=2, current_order=2, vehicle_vel='otg'):
    """Creates a selection matrix for choosing indexes of X
    related to northerly vehicle kinematics.
    
    Parameters:
        m (int) : number of timepoints
        n (int) : number of depthpoints
        vehicle_order (int) : order of vehicle smoothing in matrix Q
        current_order (int) : order of current smoothing in matrix Q
        vehicle_vel (str) : if "otg", then current skips modeling 1st order
    """

    current_order -= vehicle_vel == 'otg'
    n_rows = vehicle_order * m
    n_cols = size_of_x(m, n, vehicle_order, current_order)
    diag = vehicle_order * m

    return scipy.sparse.eye(n_rows, n_cols, diag)


def ec_select(m, n, vehicle_order=2, current_order=2, vehicle_vel='otg'):
    """Creates a selection matrix for choosing indexes of X
    related to easterly current.
    
    Parameters:
        m (int) : number of timepoints
        n (int) : number of depthpoints
        vehicle_order (int) : order of vehicle smoothing in matrix Q
        current_order (int) : order of current smoothing in matrix Q
        vehicle_vel (str) : if "otg", then current skips modeling 1st order
    """

    current_order -= vehicle_vel == 'otg'
    n_rows = current_order * n
    n_cols = size_of_x(m, n, vehicle_order, current_order)
    diag = 2 * vehicle_order * m

    return scipy.sparse.eye(n_rows, n_cols, diag)


def nc_select(m, n, vehicle_order=2, current_order=2, vehicle_vel='otg'):
    """Creates a selection matrix for choosing indexes of X
    related to northerly current.
    
    Parameters:
        m (int) : number of timepoints
        n (int) : number of depthpoints
        vehicle_order (int) : order of vehicle smoothing in matrix Q
        current_order (int) : order of current smoothing in matrix Q
        vehicle_vel (str) : if "otg", then current skips modeling 1st order
    """

    current_order -= vehicle_vel == 'otg'
    n_rows = current_order * n
    n_cols = size_of_x(m, n, vehicle_order, current_order)
    diag = 2 * vehicle_order * m + current_order * n

    return scipy.sparse.eye(n_rows, n_cols, diag)
