# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 11:31:38 2019

@author: 600301

This module provides functions for creating an optimization problem
to solve for current and navigation profile given data.  Throughout
this module, profiles are assumed to be in a single vector.  The 
vector is structured as a vector of easterly kinematics, with velocity
and position interleaved, starting with the first time point's velocity.
This easterly kinematic vector is stacked on top of a northerly 
kinematic vector of the same format.  The kinematic vector is stacked
on top of an easterly current vector from 0 to 2*max depth which is 
in turn stacked on top of a similar northerly current vector.
"""
import random

import numpy as np
import scipy.sparse
from scipy.optimize import minimize

from . import matbuilder as mb
from . import dataprep as dp

def init_x(times, depths, ddat):
    ev0, nv0 = initial_kinematics(times, ddat)
    n = len(depths)
    ec0 = np.zeros(n)
    nc0 = np.zeros(n)
    return np.hstack((ev0, nv0, ec0, nc0))    

def initial_kinematics(times, ddat):
    """Creates an guess for X, both V_otg and position, based off of
    interpolation from gps.
    
    Returns:
        tuple of numpy arrays.  The first is the guess for 
        [v1, x1, v2, x2, v3, ...] in the easterly direction, the 
        second is the guess in the northerly direction.  
        The values are given in meters n and east from a reference 
        point in a rectangular coordinate system.
        
    Note:
        Can convert between meters in rectangular coordinate system
        to lat/lon using the positions of the sensors in range
        measurements
    """
    e0 = np.zeros((len(times), 2))
    n0 = np.zeros((len(times), 2))
    first_point = ddat['gps'].iloc[0].to_numpy()
    last_point = ddat['gps'].iloc[-1].to_numpy()
    first_time = ddat['gps'].index[0]
    last_time = ddat['gps'].index[-1]
    speed = (last_point-first_point)/(last_time-first_time).seconds
    e0[:,0] = speed[0]
    n0[:,0] = speed[1]

    time_offset = (times - first_time.to_numpy()).astype(float)/1e9 #ns -> s
    e0[:,1] = time_offset * speed[0] + first_point[0]
    n0[:,1] = time_offset * speed[1] + first_point[1]

    return e0.reshape(-1), n0.reshape(-1)

def e_select(m, n):
    """Creates a selection matrix for choosing indexes of X
    related to easterly variables.

    Parameters:
        m (int) : number of timepoints
        n (int) : number of depthpoints
    """
    EV = ev_select(m,n)
    EC = ec_select(m,n)
    return scipy.sparse.vstack((EV, EC))

def n_select(m, n):
    """Creates a selection matrix for choosing indexes of X
    related to easterly vehicle kinematics.

    Parameters:
        m (int) : number of timepoints
        n (int) : number of depthpoints
    """
    NV = nv_select(m,n)
    NC = nc_select(m,n)
    return scipy.sparse.vstack((NV, NC))

def ev_select(m, n):
    """Creates a selection matrix for choosing indexes of X
    related to easterly vehicle kinematics.
    
    Parameters:
        m (int) : number of timepoints
        n (int) : number of depthpoints
    """
    return scipy.sparse.eye(2*m, 4*m+2*n)

def nv_select(m, n):
    """Creates a selection matrix for choosing indexes of X
    related to northerly vehicle kinematics.
    
    Parameters:
        m (int) : number of timepoints
        n (int) : number of depthpoints
    """
    return scipy.sparse.eye(2*m, 4*m+2*n, 2*m)

def ec_select(m, n):
    """Creates a selection matrix for choosing indexes of X
    related to easterly current.
    
    Parameters:
        m (int) : number of timepoints
        n (int) : number of depthpoints
    """
    return scipy.sparse.eye(n, 4*m+2*n, 4*m)

def nc_select(m, n):
    """Creates a selection matrix for choosing indexes of X
    related to northerly current.
    
    Parameters:
        m (int) : number of timepoints
        n (int) : number of depthpoints
    """
    return scipy.sparse.eye(n, 4*m+2*n, 4*m+n)

def backsolve(ddat, adat, rho_v=1, rho_c=1, rho_t=1, rho_a=1, rho_g=1):
    """Solves the linear least squares problem

    Returns:
        tuple of solution vector, NV, EV, NC, EC, Xs, and Vs selector
        matrices.
    """
    times = dp.timepoints(adat, ddat)
    depths = dp.depthpoints(adat, ddat)
    A, b = solve_mats(times, depths, ddat, adat, rho_v=rho_v, rho_c=rho_c,
                      rho_t=rho_t, rho_a=rho_a, rho_g=rho_g)
    x = scipy.sparse.linalg.spsolve(A, b)
    Vs = mb.v_select(len(times))
    Xs = mb.x_select(len(times))
    m = len(times)
    n = len(depths)
    EV = ev_select(m, n)
    NV = nv_select(m, n)
    EC = ec_select(m, n)
    NC = nc_select(m, n)

    x = time_rescale(x, mb.t_scale, m, n)
    return x, (NV, EV, NC, EC, Xs, Vs)

def time_rescale(x, t_s, m, n):
    """Rescales the velocity measurements in a solution vector.

    Parameters:
        x (numpy array): Previous solution
        t_s (float): timescale (e.g. 1e3 for kiloseconds, 1e-3 for
            milliseconds)
        m (int) : number of timepoints
        n (int) : number of depthpoints
    """
    Vs = mb.v_select(m)
    Xs = mb.x_select(m)
    EV = ev_select(m, n)
    NV = nv_select(m, n)
    EC = ec_select(m, n)
    NC = nc_select(m, n)
    velocity_scaler = scipy.sparse.vstack((1/t_s * Vs @ NV,
                                           Xs @ NV,
                                           1/t_s * Vs @ EV,
                                           Xs @ EV,
                                           1/t_s * NC,
                                           1/t_s * EC))
    vel_reshaper = scipy.sparse.vstack((Vs @ NV,
                                        Xs @ NV,
                                        Vs @ EV,
                                        Xs @ EV,
                                        NC,
                                        EC))
    return vel_reshaper.T @ velocity_scaler @ x
def solve_mats(times, depths, ddat, adat, rho_v=1, rho_c=1, rho_t=1,
               rho_a=1, rho_g=1, verbose=False):
    """Create A, b for which Ax=b solves linear least squares problem

    Parameters:
        times ([numpy.datetime64,]) : all of the sample times to predict
            V_otg for.  returned by dataprep.timepoints()
        depths([int]) : all of the sample depths to predict current for.
            returned by dataprep.depthpoints()
        ddat (dict): the recorded dive data returned by load_dive()
        adat (dict): the recorded ADCP data returned by load_adcp()
        rho_v (float): weight for velocity kalman filter, equal to
            differential covariance of driving gaussian process.
        rho_c (float): weight for current kalman filter, equal to
                differential covariance of driving gaussian process.
        rho_t (float): hydrodynamic model measurement error covariance
        rho_a (float): ADCP measurement error variance
        rho_g (float): GPS measurement error variance
        rho_r (float): range measurement error variance

    Returns:
        tuple of numpy arrays, (A,b)
    """
    Gv = mb.vehicle_G(times)
    Gc = mb.depth_G(depths)
    if rho_v != 0:
        Qvinv = mb.vehicle_Qinv(times, rho=rho_v)
    else:
        Qvinv = scipy.sparse.csr_matrix((2*(len(times)-1), 2*(len(times)-1)))
    if rho_c != 0:
        Qcinv = mb.depth_Qinv(depths, rho=rho_c)
    else:
        Qcinv = scipy.sparse.csr_matrix((len(depths)-1, len(depths)-1))
    zttw_e = mb.get_zttw(ddat, 'east')
    zttw_n = mb.get_zttw(ddat, 'north')
    A_ttw, B_ttw = mb.uv_select(times, depths, ddat)
    Vs = mb.v_select(len(times))

    zadcp_e = mb.get_zadcp(adat, 'east')
    zadcp_n = mb.get_zadcp(adat, 'north')
    A_adcp, B_adcp = mb.adcp_select(times, depths, ddat, adat)

    zgps_e = mb.get_zgps(ddat, 'east')
    zgps_n = mb.get_zgps(ddat, 'north')
    A_gps = mb.gps_select(times, ddat)
    Xs = mb.x_select(len(times))

    m = len(times)
    n = len(depths)
    EV = ev_select(m, n)
    NV = nv_select(m, n)
    EC = ec_select(m, n)
    NC = nc_select(m, n)
    kalman_mat = 1/2* (EV.T @ Gv.T @ Qvinv @ Gv @ EV +
                  NV.T @ Gv.T @ Qvinv @ Gv @ NV +
                  EC.T @ Gc.T @ Qcinv @ Gc @ EC +
                  NC.T @ Gc.T @ Qcinv @ Gc @ NC)
    kalman_mat = (kalman_mat+kalman_mat.T)/2

    e_ttw_select = A_ttw @ Vs @ EV - B_ttw @ EC
    n_ttw_select = A_ttw @ Vs @ NV - B_ttw @ NC
    e_adcp_select = B_adcp @ EC - A_adcp @ Vs @ EV
    n_adcp_select = B_adcp @ NC - A_adcp @ Vs @ NV
    e_gps_select = A_gps @ Xs @ EV
    n_gps_select = A_gps @ Xs @ NV

    A = (  kalman_mat
         + 1/(rho_t)*n_ttw_select.T @ n_ttw_select
         + 1/(rho_t)*e_ttw_select.T @ e_ttw_select
         + 1/(rho_a)*n_adcp_select.T @ n_adcp_select
         + 1/(rho_a)*e_adcp_select.T @ e_adcp_select
         + 1/(rho_g)*n_gps_select.T @ n_gps_select
         + 1/(rho_g)*e_gps_select.T @ e_gps_select
         )
    if verbose:
        r100 = np.array(random.sample(range(0, 4*m+2*n), 100))
        r1000 = np.array(random.sample(range(0, 4*m+2*n), 1000))
        c1 = np.linalg.cond(kalman_mat.todense()[r1000[:,None],r1000])
        c2 = np.linalg.cond(kalman_mat.todense()[r100[:,None],r100])
        c3 = np.linalg.cond(A.todense()[r1000[:,None],r1000])
        c4 = np.linalg.cond(A.todense()[r100[:,None],r100])
        print('Condition number of kalman matrix (1000x1000): ',
              f'{c1:e}')
        print('Condition number of kalman matrix (100x100): ',
              f'{c2:e}')
        print('Condition number of A (1000x1000): ',
              f'{c3:e}')
        print('Condition number of A (100x100): ',
              f'{c4:e}')

    b = (  1/(rho_t)*n_ttw_select.T @ zttw_n
         + 1/(rho_t)*e_ttw_select.T @ zttw_e
         + 1/(rho_a)*n_adcp_select.T @ zadcp_n
         + 1/(rho_a)*e_adcp_select.T @ zadcp_e
         + 1/(rho_g)*n_gps_select.T @ zgps_n
         + 1/(rho_g)*e_gps_select.T @ zgps_e
        )

    return A, b

def f(times, depths, ddat, adat, rho_v=1, rho_c=1, rho_t=1, 
                                  rho_a=1, rho_g=1, rho_r=0, verbose=False):
    """Creates the sum of squares function for fitting a navigation
    and current profile
    
    Parameters:
        times ([numpy.datetime64,]) : all of the sample times to predict
            V_otg for.  returned by dataprep.timepoints()
        depths([int]) : all of the sample depths to predict current for.
            returned by dataprep.depthpoints()
        ddat (dict): the recorded dive data returned by load_dive()
        adat (dict): the recorded ADCP data returned by load_adcp()
        rho_v (float): weight for velocity kalman filter, equal to 
            differential covariance of driving gaussian process.
        rho_c (float): weight for current kalman filter, equal to 
                differential covariance of driving gaussian process.
        rho_t (float): weight for hydrodynamic model measurement error
        rho_a (float): weight for ADCP measurement error
        rho_g (float): weight for GPS measurement error
        rho_r (float): weight for range measurement error
        
    Returns:
        scalar-valued function for input of length m = 4*ts + cs, where
        ts is the number of timepoints and cs is the number of depth
        points.
    """
    Gv = mb.vehicle_G(times)
    Gc = mb.depth_G(depths)
    if rho_v != 0:
        Qvinv = mb.vehicle_Qinv(times, rho=rho_v)
    else:
        Qvinv = scipy.sparse.csr_matrix((2*(len(times)-1), 2*(len(times)-1)))
    if rho_c != 0:
        Qcinv = mb.depth_Qinv(depths, rho=rho_c)
    else:
        Qcinv = scipy.sparse.csr_matrix((len(depths)-1, len(depths)-1))
    zttw_e = mb.get_zttw(ddat, 'east')
    zttw_n = mb.get_zttw(ddat, 'north')
    A_ttw, B_ttw = mb.uv_select(times, depths, ddat)
    Vs = mb.v_select(len(times))

    zadcp_e = mb.get_zadcp(adat, 'east')
    zadcp_n = mb.get_zadcp(adat, 'north')
    A_adcp, B_adcp = mb.adcp_select(times, depths, ddat, adat)

    zgps_e = mb.get_zgps(ddat, 'east')
    zgps_n = mb.get_zgps(ddat, 'north')
    A_gps = mb.gps_select(times, ddat)
    Xs = mb.x_select(len(times))
    
    zr, zx, zy = mb.get_zrange(ddat)
    A_range = mb.range_select(times, ddat)
    
    m = len(times)
    n = len(depths)
    EV = ev_select(m, n)
    NV = nv_select(m, n)
    EC = ec_select(m, n)
    NC = nc_select(m, n)
    kalman_mat = 1/2* (EV.T @ Gv.T @ Qvinv @ Gv @ EV +
                  NV.T @ Gv.T @ Qvinv @ Gv @ NV +
                  EC.T @ Gc.T @ Qcinv @ Gc @ EC +
                  NC.T @ Gc.T @ Qcinv @ Gc @ NC)
    kalman_mat = (kalman_mat+kalman_mat.T)/2
    
    if verbose:
        r100 = np.array(random.sample(range(0, 4*m+2*n), 100))
        r1000 = np.array(random.sample(range(0, 4*m+2*n), 1000))
        c1 = np.linalg.cond(kalman_mat.todense()[r1000[:,None],r1000])
        c2 = np.linalg.cond(kalman_mat.todense()[r100[:,None],r100])
        print('Condition number of kalman matrix (1000x1000): ',
              f'{c1:e}')
        print('Condition number of kalman matrix (100x100): ',
              f'{c2:e}')

    e_ttw_select = A_ttw @ Vs @ EV - B_ttw @ EC
    n_ttw_select = A_ttw @ Vs @ NV - B_ttw @ NC
    e_adcp_select = B_adcp @ EC - A_adcp @ Vs @ EV
    n_adcp_select = B_adcp @ NC - A_adcp @ Vs @ NV
    e_gps_select = A_gps @ Xs @ EV
    n_gps_select = A_gps @ Xs @ NV
    e_range_select = A_range @ Xs @ EV 
    n_range_select = A_range @ Xs @ NV

    def f_eval(X):
        kalman_error = X.T @ kalman_mat @ X
        hydrodynamic_error = 1/(2*rho_t)*(
                                np.linalg.norm(zttw_n-n_ttw_select @ X)**2 +
                                np.linalg.norm(zttw_e-e_ttw_select @ X)**2)
        adcp_error = 1/(2*rho_a)*(
                        np.linalg.norm(zadcp_n-n_adcp_select @ X)**2 +
                        np.linalg.norm(zadcp_e-e_adcp_select @ X)**2)
        gps_error = 1/(2*rho_g)*(
                            np.linalg.norm(zgps_n-n_gps_select @ X)**2 +
                            np.linalg.norm(zgps_e-e_gps_select @ X)**2)
        if rho_r != 0:
            ranges = np.sqrt((zx-e_range_select @ X) ** 2 +
                             (zy-n_range_select @ X) ** 2)
            range_error = 1/(2*rho_r)*np.linalg.norm(zr-ranges)**2
        else: range_error=0
        return (kalman_error + hydrodynamic_error + adcp_error +
                gps_error + range_error)
    return f_eval
def g(times, depths, ddat, adat, rho_v=1, rho_c=1, rho_t=1, 
                                  rho_a=1, rho_g=1, rho_r=0):
    """Creates the sum of squares gradient function for fitting a 
    navigation and current profile
    
    Returns:
        vector-valued function for input & output of length m = 4*ts + cs,
        where ts is the number of timepoints and cs is the number of depth
        points.
    """
    Gv = mb.vehicle_G(times)
    Gc = mb.depth_G(depths)
    if rho_v != 0:
        Qvinv = mb.vehicle_Qinv(times, rho=rho_v)
    else:
        Qvinv = scipy.sparse.csr_matrix((2*(len(times)-1), 2*(len(times)-1)))
    if rho_c != 0:
        Qcinv = mb.depth_Qinv(depths, rho=rho_c)
    else:
        Qcinv = scipy.sparse.csr_matrix((len(depths)-1, len(depths)-1))
    zttw_e = mb.get_zttw(ddat, 'east')
    zttw_n = mb.get_zttw(ddat, 'north')
    A_ttw, B_ttw = mb.uv_select(times, depths, ddat)
    Vs = mb.v_select(len(times))

    zadcp_e = mb.get_zadcp(adat, 'east')
    zadcp_n = mb.get_zadcp(adat, 'north')
    A_adcp, B_adcp = mb.adcp_select(times, depths, ddat, adat)

    zgps_e = mb.get_zgps(ddat, 'east')
    zgps_n = mb.get_zgps(ddat, 'north')
    A_gps = mb.gps_select(times, ddat)
    Xs = mb.x_select(len(times))
    
    zr, zx, zy = mb.get_zrange(ddat)
    A_range = mb.range_select(times, ddat)
    
    m = len(times)
    n = len(depths)
    EV = ev_select(m, n)
    NV = nv_select(m, n)
    EC = ec_select(m, n)
    NC = nc_select(m, n)

    kalman_mat = 1/2* (EV.T @ Gv.T @ Qvinv @ Gv @ EV +
                  NV.T @ Gv.T @ Qvinv @ Gv @ NV +
                  EC.T @ Gc.T @ Qcinv @ Gc @ EC +
                  NC.T @ Gc.T @ Qcinv @ Gc @ NC)
    kalman_mat = (kalman_mat+kalman_mat.T)/2
    
    e_ttw_select = A_ttw @ Vs @ EV - B_ttw @ EC
    n_ttw_select = A_ttw @ Vs @ NV - B_ttw @ NC
    e_ttw_mat = 2 * e_ttw_select.T @ e_ttw_select 
    e_ttw_constant = 2*e_ttw_select.T @ zttw_e
    n_ttw_mat = 2 * n_ttw_select.T @ n_ttw_select 
    n_ttw_constant = 2*n_ttw_select.T @ zttw_n

    e_adcp_select = B_adcp @ EC - A_adcp @ Vs @ EV
    n_adcp_select = B_adcp @ NC - A_adcp @ Vs @ NV
    e_adcp_mat = 2* e_adcp_select.T @ e_adcp_select
    e_adcp_constant = 2 * e_adcp_select.T @ zadcp_e
    n_adcp_mat = 2* n_adcp_select.T @ n_adcp_select
    n_adcp_constant = 2 * n_adcp_select.T @ zadcp_n
    
    e_gps_select = A_gps @ Xs @ EV
    n_gps_select = A_gps @ Xs @ NV
    e_gps_mat = 2* e_gps_select.T @ e_gps_select
    e_gps_constant = 2 * e_gps_select.T @ zgps_e
    n_gps_mat = 2* n_gps_select.T @ n_gps_select
    n_gps_constant = 2 * n_gps_select.T @ zgps_n

    A1 = A_range @ Xs @ EV  #e_range_select
    A2 = A_range @ Xs @ NV  #n_range_select


    def g_eval(X):
        kalman_error = 2* kalman_mat @ X
        
        zttw_error = 1/(2*rho_t)*(e_ttw_mat @ X - e_ttw_constant+
                            n_ttw_mat @ X - n_ttw_constant)
        zadcp_error = 1/(2*rho_a)*(
                        e_adcp_mat @ X - e_adcp_constant +
                        n_adcp_mat @ X - n_adcp_constant)
        zgps_error = 1/(2*rho_g)*(e_gps_mat @ X - e_gps_constant +
                            n_gps_mat @ X - n_gps_constant)
        if rho_r != 0:
            denominator = np.sqrt((A1 @ X - zx)**2 +(A2 @ X - zy)**2)
            factor1 = (np.ones(len(zr)) - zr/denominator)
            factor2 = (2*A1.T.multiply(A1 @ X - zx) + 2*A2.T.multiply(A2 @ X - zy))
            range_error = 1/(2*rho_r)*factor2 * factor1
        else: range_error = 0
        return (kalman_error + 
                zttw_error + zadcp_error + 
                zgps_error + range_error)

    return g_eval
    
def h(times, depths, ddat, adat, rho_v=1, rho_c=1, rho_t=1, 
                                  rho_a=1, rho_g=1, rho_r=0):
    """Creates the sum of squares hessian function for fitting a 
    navigation and current profile
    
    Returns:
        vector-valued function for input of length m = 4*ts + cs and
        output of dimensions m x m, where ts is the number of timepoints 
        and cs is the number of depth points.

    Warning:
        rho_r is not yet implemented.  Only returns the hessian without
        range measurements.
    """
    Gv = mb.vehicle_G(times)
    Gc = mb.depth_G(depths)
    if rho_v != 0:
        Qvinv = mb.vehicle_Qinv(times, rho=rho_v)
    else:
        Qvinv = scipy.sparse.csr_matrix((2*(len(times)-1), 2*(len(times)-1)))
    if rho_c != 0:
        Qcinv = mb.depth_Qinv(depths, rho=rho_c)
    else:
        Qcinv = scipy.sparse.csr_matrix((len(depths)-1, len(depths)-1))
    A_ttw, B_ttw = mb.uv_select(times, depths, ddat)
    Vs = mb.v_select(len(times))

    A_adcp, B_adcp = mb.adcp_select(times, depths, ddat, adat)

    A_gps = mb.gps_select(times, ddat)
    Xs = mb.x_select(len(times))
    
    m = len(times)
    n = len(depths)
    EV = ev_select(m, n)
    NV = nv_select(m, n)
    EC = ec_select(m, n)
    NC = nc_select(m, n) 

    kalman_mat = 1/2* (EV.T @ Gv.T @ Qvinv @ Gv @ EV +
                  NV.T @ Gv.T @ Qvinv @ Gv @ NV +
                  EC.T @ Gc.T @ Qcinv @ Gc @ EC +
                  NC.T @ Gc.T @ Qcinv @ Gc @ NC)
    kalman_mat = (kalman_mat+kalman_mat.T)/2

    e_ttw_select = A_ttw @ Vs @ EV - B_ttw @ EC
    n_ttw_select = A_ttw @ Vs @ NV - B_ttw @ NC
    e_ttw_mat = 2 * e_ttw_select.T @ e_ttw_select 
    n_ttw_mat = 2 * n_ttw_select.T @ n_ttw_select 

    e_adcp_select = B_adcp @ EC - A_adcp @ Vs @ EV
    n_adcp_select = B_adcp @ NC - A_adcp @ Vs @ NV
    e_adcp_mat = 2* e_adcp_select.T @ e_adcp_select
    n_adcp_mat = 2* n_adcp_select.T @ n_adcp_select

    e_gps_select = A_gps @ Xs @ EV
    n_gps_select = A_gps @ Xs @ NV
    e_gps_mat = 2* e_gps_select.T @ e_gps_select
    n_gps_mat = 2* n_gps_select.T @ n_gps_select

    def h_eval(X):        
        zttw_error = 1/(2*rho_t)*(e_ttw_mat + n_ttw_mat)
        zadcp_error = 1/(2*rho_a)*(e_adcp_mat + n_adcp_mat)
        zgps_error = 1/(2*rho_g)*(e_gps_mat + n_gps_mat)
        return (kalman_mat + zttw_error + zadcp_error + zgps_error)
    return h_eval

def grad_test(x0, eps, f, g):
    """Tests the gradient and objective function calculation.  For small
    epsilon = ||x1-x0||, 2[f(x1)-f(x0)] should equal <x1-x0, g(x1)-g(x0)>

    Parameters:
        x0 (numpy array): test point 1
        eps (float): norm of x1-x0.  x1 randomly generated from this value.
        f (function): objective function with single argument
        g (function): gradient function with single argument

    Returns:
        Tuple of LHS, RHS, ||x1-x0||, and ||LHS-RHS||.  The first two
        should be close to equal.
    """
    x1 = x0+np.random.normal(scale=eps/len(x0), size = len(x0))
    LHS = 2*(f(x1)-f(x0))
    RHS = (x1-x0).dot(g(x1)+g(x0))
    return LHS, RHS, np.linalg.norm(x1-x0), LHS-RHS

def hess_test(x0, eps, g, h):
    """Tests the gradient and hessian function calculation.  For small
    epsilon = ||x1-x0||, 2[f(x1)-f(x0)] should equal <x1-x0, g(x1)-g(x0)>

    Parameters:
        x0 (numpy array): test point 1
        eps (float): norm of x1-x0.  x1 randomly generated from this value.
        g (function): gradient function with single argument
        h (function): hessian function with single argument

    Returns:
        Tuple of LHS, RHS, and ||x1-x0||.  The first two
        should be close to equal.
    """
    x1 = x0+np.random.normal(scale=eps/len(x0), size = len(x0))
    LHS = 2*np.dot(g(x1)-g(x0), x1-x0)
    RHS = (x1-x0).T @ (h(x1)+h(x0)) @ (x1-x0)
    return LHS, RHS, np.linalg.norm(x1-x0), np.linalg.norm(LHS-RHS)

def backsolve_test(x0, ddat, adat, rho_v=1, rho_c=1, rho_t=1, rho_a=1,
                   rho_g=1):
    """Tests whether the linear least squares solver agrees with the
    gradient function"""
    times=dp.timepoints(adat, ddat)
    depths=dp.depthpoints(adat, ddat)

    A, b = solve_mats(times, depths, ddat, adat, rho_v=rho_v, rho_c=rho_c,
                      rho_t=rho_t, rho_a=rho_a, rho_g=rho_g)
    grad_func = g(times, depths, ddat, adat,
                  rho_v=rho_v, rho_c=rho_c, rho_a=rho_a, rho_t=rho_t,
                  rho_g=rho_g)
    v1 = A @ x0 + grad_func(np.zeros(len(x0)))
    v2 = grad_func(x0)
    return v1, v2, np.linalg.norm(v1), np.linalg.norm(v1-v2)

def solve(ddat, adat, rho_v=1, rho_c=1, rho_t=1, rho_a=1, rho_g=1, rho_r=0,
          method='L-BFGS-B'):
    """Solve the ADCP navigation problem for given data."""
    times = dp.timepoints(adat, ddat)
    depths = dp.depthpoints(adat, ddat)
    x0 = init_x(times, depths, ddat)
    ffunc = f(times, depths, ddat, adat, rho_v=rho_v, rho_c=rho_c, rho_t=rho_t,
             rho_a=rho_a, rho_g=rho_g, rho_r=rho_r)
    gfunc = g(times, depths, ddat, adat, rho_v=rho_v, rho_c=rho_c, rho_t=rho_t,
             rho_a=rho_a, rho_g=rho_g, rho_r=rho_r)
    sol = minimize(ffunc, x0, method=method, jac=gfunc,
                   options={'maxiter':50000, 'maxfun':50000, 'disp':True})
    m = len(times)
    n = len(depths)
    sol.x = time_rescale(sol.x, mb.t_scale, m, n)

    return sol
