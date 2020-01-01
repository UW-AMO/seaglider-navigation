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

import numpy as np
import scipy.sparse

from . import matbuilder as mb

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

def f(times, depths, ddat, adat, rho_v=1, rho_c=1, rho_t=1, 
                                  rho_a=1, rho_g=1, rho_r=0):
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
    Qv = mb.vehicle_Q(times, rho=rho_v)
    Qc = mb.depth_Q(depths, rho=rho_c)
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
    def f_eval(X):
        ev_vec = Gv @ EV @ X
        nv_vec = Gv @ NV @ X
        kalman_vehicle = ev_vec.T @ Qv @ ev_vec + nv_vec.T @ Qv @ nv_vec
        ec_vec = Gc @ EC @ X
        nc_vec = Gc @ NC @ X
        kalman_current = ec_vec.T @ Qc @ ec_vec + nc_vec.T @ Qc @ nc_vec
        
        e_ttw_vels = A_ttw @ Vs @ EV @ X - B_ttw @ EC @ X
        n_ttw_vels = A_ttw @ Vs @ NV @ X - B_ttw @ NC @ X
        zttw_error = rho_t*(np.linalg.norm(zttw_n-n_ttw_vels
                                    )+np.linalg.norm(zttw_e-e_ttw_vels))
        e_adcp_vels = B_adcp @ EC @ X - A_adcp @ Vs @ EV @ X
        n_adcp_vels = B_adcp @ NC @ X - A_adcp @ Vs @ NV @ X
        zadcp_error = rho_a*(np.linalg.norm(zadcp_n-n_adcp_vels
                                    )+np.linalg.norm(zadcp_e-e_adcp_vels))

        e_pos = A_gps @ Xs @ EV @ X
        n_pos = A_gps @ Xs @ NV @ X
        zgps_error = rho_g*(np.linalg.norm(zgps_e-e_pos)+
                            np.linalg.norm(zgps_n-n_pos))

        xv = A_range @ Xs @ EV @ X
        yv = A_range @ Xs @ NV @ X
        ranges = np.sqrt((zx-xv) ** 2 + (zy-yv) ** 2)
        range_error = rho_r*np.linalg.norm(zr-ranges)
        return (kalman_vehicle + kalman_current + 
                zttw_error + zadcp_error + 
                zgps_error + range_error)
    
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
    Qv = mb.vehicle_Q(times, rho=rho_v)
    Qc = mb.depth_Q(depths, rho=rho_c)
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

    def g_eval(X):
        ev_mat = Gv @ EV
        nv_mat = Gv @ NV
        kalman_vehicle = 2* ev_mat.T @ Qv @ ev_mat @ X + 2* nv_mat.T @ Qv @ nv_mat @ X
        ec_mat = Gc @ EC
        nc_mat = Gc @ NC
        kalman_current = 2* ec_mat.T @ Qc @ ec_mat @ X + 2* nc_mat.T @ Qc @ nc_mat @ X
        
        e_ttw_mat = A_ttw @ Vs @ EV - B_ttw @ EC
        n_ttw_mat = A_ttw @ Vs @ NV - B_ttw @ NC
        zttw_error = rho_t*(
                        2* e_ttw_mat.T @ e_ttw_mat @ X - 2*e_ttw_mat.T @ zttw_e +
                        2* n_ttw_mat.T @ n_ttw_mat @ X - 2*n_ttw_mat.T @ zttw_n)
        e_adcp_mat = B_adcp @ EC - A_adcp @ Vs @ EV
        n_adcp_mat = B_adcp @ NC - A_adcp @ Vs @ NV
        zadcp_error = rho_a*(
                        2* e_adcp_mat.T @ e_adcp_mat @ X - 2*e_adcp_mat.T @ zadcp_e +
                        2* n_adcp_mat.T @ n_adcp_mat @ X - 2*n_adcp_mat.T @ zadcp_n)

        e_mat = A_gps @ Xs @ EV
        n_mat = A_gps @ Xs @ NV
        zgps_error = rho_g*(
                        2* e_mat.T @ e_mat @ X - 2*e_mat.T @ zgps_e +
                        2* n_mat.T @ n_mat @ X - 2*n_mat.T @ zgps_n)

        A1 = A_range @ Xs @ EV
        A2 = A_range @ Xs @ NV
        denominator = np.sqrt((A1 @ X - zx)**2 +(A2 @ X - zy)**2)
        factor1 = (np.ones(len(zr)) - zr/denominator)
        factor2 = (2*A1.T.multiply(A1 @ X - zx) + 2*A2.T.multiply(A2 @ X - zy))
        range_error = factor2 * factor1
        return (kalman_vehicle + kalman_current + 
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
    Qv = mb.vehicle_Q(times, rho=rho_v)
    Qc = mb.depth_Q(depths, rho=rho_c)
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

    def h_eval(X):
        ev_mat = Gv @ EV
        nv_mat = Gv @ NV
        kalman_vehicle = ev_mat.T @ Qv @ ev_mat + nv_mat.T @ Qv @ nv_mat
        ec_mat = Gc @ EC
        nc_mat = Gc @ NC
        kalman_current = ec_mat.T @ Qc @ ec_mat + nc_mat.T @ Qc @ nc_mat
        
        e_ttw_mat = A_ttw @ Vs @ EV - B_ttw @ EC
        n_ttw_mat = A_ttw @ Vs @ NV - B_ttw @ NC
        zttw_error = rho_t*(
                        2* e_ttw_mat.T @ e_ttw_mat +
                        2* n_ttw_mat.T @ n_ttw_mat)
        
        e_adcp_mat = B_adcp @ EC - A_adcp @ Vs @ EV
        n_adcp_mat = B_adcp @ NC - A_adcp @ Vs @ NV
        zadcp_error = rho_a*(
                        2* e_adcp_mat.T @ e_adcp_mat +
                        2* n_adcp_mat.T @ n_adcp_mat)

        e_mat = A_gps @ Xs @ EV
        n_mat = A_gps @ Xs @ NV
        zgps_error = rho_g*(
                        2* e_mat.T @ e_mat +
                        2* n_mat.T @ n_mat)
        
        return (kalman_vehicle + kalman_current + 
                zttw_error + zadcp_error + 
                zgps_error)
    return h_eval