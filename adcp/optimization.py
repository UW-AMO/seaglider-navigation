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

from adcp import matbuilder as mb
from adcp import dataprep as dp

class GliderProblem:
    defaults = dict(
        rho_v = 1,
        rho_c = 1,
        rho_t = 1,
        rho_a = 1,
        rho_g = 1,
        rho_r = 0,
        t_scale = 1e3,
        conditioner = 'tanh',
        vehicle_order = 2,
        current_order = 2,
        vehicle_vel = 'otg'
    )
    def __init__(self, ddat, adat, **kwargs):
        self.ddat = ddat
        self.adat = adat
        self.times = dp.timepoints(adat, ddat)
        self.depths = dp.depthpoints(adat, ddat)
        for k, v in self.defaults.items():
            if k in kwargs:
                setattr(self, k, kwargs[k])
            else:
                setattr(self, k, v)
        for k in kwargs:
            if k not in self.__dict__:
                raise AttributeError(f'{k} not a argument for Problem'\
                                     ' constructor')
        if self.vehicle_vel == 'ttw':
            self.depth_rates = dp.depth_rates(
                self.times,
                self.depths,
                self.ddat
            )
        else:
            self.depth_rates = None
        self.m = len(self.times)
        self.n = len(self.depths)
        self.As = mb.a_select(self.m, self.vehicle_order)
        self.Vs = mb.v_select(self.m, self.vehicle_order)
        self.Xs = mb.x_select(self.m, self.vehicle_order)
        self.CA = mb.ca_select(self.n, self.current_order, self.vehicle_vel)
        self.CV = mb.cv_select(self.n, self.current_order, self.vehicle_vel)
        self.CX = mb.cx_select(self.n, self.current_order, self.vehicle_vel)
        self.EV = mb.ev_select(
            self.m, self.n, self.vehicle_order, self.current_order, self.vehicle_vel
        )
        self.NV = mb.nv_select(
            self.m, self.n, self.vehicle_order, self.current_order, self.vehicle_vel
        )
        self.EC = mb.ec_select(
            self.m, self.n, self.vehicle_order, self.current_order, self.vehicle_vel
        )
        self.NC = mb.nc_select(
            self.m, self.n, self.vehicle_order, self.current_order, self.vehicle_vel
        )


# %%
def init_x(prob):
    ev0, nv0 = initial_kinematics(
        prob.times,
        prob.ddat,
        prob.vehicle_order,
    )
    n = len(prob.depths)
    current_order = prob.current_order
    if prob.vehicle_vel ==  'otg':
        current_order = prob.current_order - 1
    ec0 = np.zeros(current_order * n)
    nc0 = np.zeros(current_order * n)
    return np.hstack((ev0, nv0, ec0, nc0))

def initial_kinematics(times, ddat, vehicle_order):
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

    Note:
        Does not matter whether modeling v_otg or v_ttw
    """
    e0 = np.zeros((len(times), vehicle_order))
    n0 = np.zeros((len(times), vehicle_order))
    first_point = ddat['gps'].iloc[0].to_numpy()
    last_point = ddat['gps'].iloc[-1].to_numpy()
    first_time = ddat['gps'].index[0]
    last_time = ddat['gps'].index[-1]
    if len(ddat['gps']) == 1:
        speed = [0,0]
    else:
        speed = (last_point-first_point)/(last_time-first_time).seconds
    e0[:, vehicle_order-1] = speed[0]
    n0[:, vehicle_order-1] = speed[1]

    time_offset = (times - first_time.to_numpy()).astype(float)/1e9 #ns -> s
    e0[:,1] = time_offset * speed[0] + first_point[0]
    n0[:,1] = time_offset * speed[1] + first_point[1]

    return e0.reshape(-1), n0.reshape(-1)


def backsolve(prob):
    """Solves the linear least squares problem

    Parameters:
        prob (GliderProblem)
    
    Returns:
        tuple of solution vector, NV, EV, NC, EC, Xs, and Vs selector
        matrices.
    """
    A, b = solve_mats(prob)
    x = scipy.sparse.linalg.spsolve(A, b)
    As = prob.As
    Vs = prob.Vs
    Xs = prob.Xs
    CV = prob.CV
    EV = prob.EV
    NV = prob.NV
    EC = prob.EC
    NC = prob.NC

    x = time_rescale(x, prob.t_scale, prob)
    return x, (NV, EV, NC, EC, CV, Xs, Vs, As)

def time_rescale(x, t_s, prob):
    """Rescales the velocity measurements in a solution vector to undo
    scaling factor t_s

    Parameters:
        x (numpy array): Previous solution
        t_s (float): timescale to remove(e.g. 1e3 for kiloseconds, 1e-3
            for milliseconds)
        prob (GliderProblem) : A glider problem
    """

    As = prob.As
    Vs = prob.Vs
    Xs = prob.Xs
    CA = prob.CA
    CV = prob.CV
    CX = prob.CX
    EV = prob.EV
    NV = prob.NV
    EC = prob.EC
    NC = prob.NC

    nm = lambda x, y : None if x is None or y is None else x * y # nt = "None, multiplied"
    nmm = lambda A, B : None if A is None or B is None else A @ B # "None, matrix multiplied"
    velocity_scaler = scipy.sparse.vstack((
        nm(1/t_s**2, nmm(As, NV)),
        1/t_s * Vs @ NV,
        Xs @ NV,
        nm(1/t_s**2, nmm(As, EV)),
        1/t_s * Vs @ EV,
        Xs @ EV,
        nm(1/t_s, nmm(CA, NC)),
        1/t_s * CV @ NC,
        nmm(CX, NC),
        nm(1/t_s, nmm(CA, EC)),
        1/t_s * CV @ EC,
        nmm(CX, EC),
    ))
    vel_reshaper = scipy.sparse.vstack((
        nmm(As, NV),
        1/t_s * Vs @ NV,
        Xs @ NV,
        nmm(As, EV),
        1/t_s * Vs @ EV,
        Xs @ EV,
        nmm(CA, NC),
        1/t_s * CV @ NC,
        nmm(CX, NC),
        nmm(CA, EC),
        1/t_s * CV @ EC,
        nmm(CX, EC),
    ))
    return vel_reshaper.T @ velocity_scaler @ x


def solve_mats(prob, verbose=False):
    """Create A, b for which Ax=b solves linear least squares problem

    Parameters:
        prob (GliderProblem)

    Returns:
        tuple of numpy arrays, (A,b)
    """
    m = len(prob.times)
    n = len(prob.depths)
    zttw_e = mb.get_zttw(prob.ddat, 'east', prob.t_scale)
    zttw_n = mb.get_zttw(prob.ddat, 'north', prob.t_scale)
    A_ttw, B_ttw = mb.uv_select(prob.times, prob.depths, prob.ddat)
    Vs = mb.v_select(m, prob.vehicle_order)

    zadcp_e = mb.get_zadcp(prob.adat, 'east', prob.t_scale)
    zadcp_n = mb.get_zadcp(prob.adat, 'north', prob.t_scale)
    A_adcp, B_adcp = mb.adcp_select(prob.times, prob.depths, prob.ddat,
                                    prob.adat, prob.vehicle_vel)

    zgps_e = mb.get_zgps(prob.ddat, 'east')
    zgps_n = mb.get_zgps(prob.ddat, 'north')
    A_gps = mb.gps_select(prob.times, prob.ddat)

    Xs = prob.Xs
    CV = prob.CV
    EV = prob.EV
    NV = prob.NV
    EC = prob.EC
    NC = prob.NC

    kalman_mat = gen_kalman_mat(prob)

    e_ttw_select = A_ttw @ Vs @ EV - B_ttw @ CV @ EC
    n_ttw_select = A_ttw @ Vs @ NV - B_ttw @ CV @ NC
    e_adcp_select = B_adcp @ CV @ EC - A_adcp @ Vs @ EV
    n_adcp_select = B_adcp @ CV @ NC - A_adcp @ Vs @ NV
    e_gps_select = A_gps @ Xs @ EV
    n_gps_select = A_gps @ Xs @ NV

    A = (  2* kalman_mat
         + 1/(prob.rho_t)*n_ttw_select.T @ n_ttw_select
         + 1/(prob.rho_t)*e_ttw_select.T @ e_ttw_select
         + 1/(prob.rho_a)*n_adcp_select.T @ n_adcp_select
         + 1/(prob.rho_a)*e_adcp_select.T @ e_adcp_select
         + 1/(prob.rho_g)*n_gps_select.T @ n_gps_select
         + 1/(prob.rho_g)*e_gps_select.T @ e_gps_select
         )
    A = (A+A.T)/2

    if verbose:
        r100 = np.array(random.sample(range(0, 4*m+2*n), 100))
        # r1000 = np.array(random.sample(range(0, 4*m+2*n), 1000))
        # c1 = np.linalg.cond(kalman_mat.todense()[r1000[:,None],r1000])
        c2 = np.linalg.cond(kalman_mat.todense()[r100[:,None],r100])
        # c3 = np.linalg.cond(A.todense()[r1000[:,None],r1000])
        c4 = np.linalg.cond(A.todense()[r100[:,None],r100])
        # print('Condition number of kalman matrix (1000x1000): ',
        #       f'{c1:e}')
        print('Condition number of kalman matrix (100x100): ',
              f'{c2:e}')
        # print('Condition number of A (1000x1000): ',
        #       f'{c3:e}')
        print('Condition number of A (100x100): ',
              f'{c4:e}')

    b = (  1/(prob.rho_t)*n_ttw_select.T @ zttw_n
         + 1/(prob.rho_t)*e_ttw_select.T @ zttw_e
         + 1/(prob.rho_a)*n_adcp_select.T @ zadcp_n
         + 1/(prob.rho_a)*e_adcp_select.T @ zadcp_e
         + 1/(prob.rho_g)*n_gps_select.T @ zgps_n
         + 1/(prob.rho_g)*e_gps_select.T @ zgps_e
        )

    return A, b

def gen_kalman_mat(prob):
    """Create the combined kalman process covariance matrix for the vehicle
    and current.  Specifically, it is 1/2 the symmetrized inverse covariance
    matrix for the problem.

    Parameters:
        prob (GliderProblem) : the glider problem to consider
    """
    Gv = mb.vehicle_G(prob.times, prob.vehicle_order, prob.conditioner, prob.t_scale)
    Gc = mb.depth_G(prob.depths, prob.current_order, prob.depth_rates, prob.conditioner)
    if prob.rho_v != 0:
        Qvinv = mb.vehicle_Qinv(
            prob.times,
            prob.rho_v,
            prob.vehicle_order,
            prob.conditioner,
            prob.t_scale
        )
    else:
        Qvinv = scipy.sparse.csr_matrix(
            (2*(len(prob.times)-1),
            2*(len(prob.times)-1))
        )
    if prob.rho_c != 0:
        Qcinv = mb.depth_Qinv(
            prob.depths,
            prob.rho_c,
            prob.current_order,
            prob.depth_rates,
            prob.conditioner,
            prob.t_scale
        )
    else:
        Qcinv = scipy.sparse.csr_matrix(
            (len(prob.depths)-1,
            len(prob.depths)-1)
        )
    EV = prob.EV
    NV = prob.NV
    EC = prob.EC
    NC = prob.NC
    kalman_mat = 1/2* (EV.T @ Gv.T @ Qvinv @ Gv @ EV +
                       NV.T @ Gv.T @ Qvinv @ Gv @ NV +
                       EC.T @ Gc.T @ Qcinv @ Gc @ EC +
                       NC.T @ Gc.T @ Qcinv @ Gc @ NC)
    kalman_mat = (kalman_mat+kalman_mat.T)/2

    return kalman_mat
# %%
def _f_kalman(prob):
    kalman_mat = gen_kalman_mat(prob)
    def f_eval(X):
        kalman_error = X.T @ kalman_mat @ X
        return kalman_error
    return f_eval
def _f_ttw(prob):
    zttw_e = mb.get_zttw(prob.ddat, 'east', prob.t_scale)
    zttw_n = mb.get_zttw(prob.ddat, 'north', prob.t_scale)
    A_ttw, B_ttw = mb.uv_select(prob.times, prob.depths, prob.ddat)

    Vs = prob.Vs
    CV = prob.CV
    EV = prob.EV
    NV = prob.NV
    EC = prob.EC
    NC = prob.NC
    v_otg = prob.vehicle_vel == 'otg'

    e_ttw_select = A_ttw @ Vs @ EV - v_otg * B_ttw @ CV @ EC
    n_ttw_select = A_ttw @ Vs @ NV - v_otg * B_ttw @ CV @ NC
    def f_eval(X):
        hydrodynamic_error = 1/(2*prob.rho_t)*(
                                np.square(zttw_n-n_ttw_select @ X).sum() +
                                np.square(zttw_e-e_ttw_select @ X).sum())
        return hydrodynamic_error
    return f_eval
def _f_adcp(prob):
    zadcp_e = mb.get_zadcp(prob.adat, 'east', prob.t_scale)
    zadcp_n = mb.get_zadcp(prob.adat, 'north', prob.t_scale)
    A_adcp, B_adcp = mb.adcp_select(prob.times, prob.depths, prob.ddat,
                                    prob.adat, prob.vehicle_vel)
    Vs = prob.Vs
    CV = prob.CV
    EV = prob.EV
    NV = prob.NV
    EC = prob.EC
    NC = prob.NC
    e_adcp_select = B_adcp @ CV @ EC - A_adcp @ Vs @ EV
    n_adcp_select = B_adcp @ CV @ NC - A_adcp @ Vs @ NV
    def f_eval(X):
        adcp_error = 1/(2*prob.rho_a)*(
                                np.square(zadcp_n-n_adcp_select @ X).sum() +
                                np.square(zadcp_e-e_adcp_select @ X).sum())
        return adcp_error
    return f_eval

def _f_gps(prob):
    zgps_e = mb.get_zgps(prob.ddat, 'east')
    zgps_n = mb.get_zgps(prob.ddat, 'north')
    A_gps = mb.gps_select(prob.times, prob.ddat)

    EV = prob.EV
    NV = prob.NV
    Xs = prob.Xs

    EC = prob.EC
    NC = prob.NC
    CX = prob.CX

    e_gps_select = A_gps @ Xs @ EV
    n_gps_select = A_gps @ Xs @ NV
    def f_eval(X):
        gps_error = 1/(2*prob.rho_g)*(
                    np.square(zgps_n-n_gps_select @ X).sum() +
                    np.square(zgps_e-e_gps_select @ X).sum())
        return gps_error
    return f_eval

def _f_range(prob):
    zr, zx, zy = mb.get_zrange(prob.ddat)
    A_range = mb.range_select(prob.times, prob.ddat)

    EV = prob.EV
    NV = prob.NV
    Xs = prob.Xs

    e_range_select = A_range @ Xs @ EV 
    n_range_select = A_range @ Xs @ NV

    def f_eval(X):
        if prob.rho_r != 0:
            ranges = np.sqrt((zx-e_range_select @ X) ** 2 +
                             (zy-n_range_select @ X) ** 2)
            range_error = 1/(2*prob.rho_r)*np.square(zr-ranges).sum()
        else: range_error = 0
        return range_error
    return f_eval

def f(prob, verbose=False):
    """Creates the sum of squares evaluation function for fitting a 
    navigation and current profile
    
    Returns:
        scalar-valued function for input of length m = 4*ts + 2*cs, where ts is
        the number of timepoints and cs is the number of depth points.
    """
    f1 = _f_kalman(prob)
    f2 = _f_ttw(prob)
    f3 = _f_adcp(prob)
    f4 = _f_gps(prob)
    f5 = _f_range(prob)

    def f_eval(X):
        return f1(X)+f2(X)+f3(X)+f4(X)+f5(X)
    return f_eval
# %%
def _g_kalman(prob):
    kalman_mat = gen_kalman_mat(prob)
    def g_eval(X):
        kalman_error = 2* kalman_mat @ X
        return kalman_error
    return g_eval

def _g_ttw(prob):
    zttw_e = mb.get_zttw(prob.ddat, 'east', prob.t_scale)
    zttw_n = mb.get_zttw(prob.ddat, 'north', prob.t_scale)
    A_ttw, B_ttw = mb.uv_select(prob.times, prob.depths, prob.ddat)

    Vs = prob.Vs
    CV = prob.CV
    EV = prob.EV
    NV = prob.NV
    EC = prob.EC
    NC = prob.NC

    e_ttw_select = A_ttw @ Vs @ EV - B_ttw @ CV @ EC
    n_ttw_select = A_ttw @ Vs @ NV - B_ttw @ CV @ NC
    e_ttw_mat = 2 * e_ttw_select.T @ e_ttw_select 
    e_ttw_constant = 2*e_ttw_select.T @ zttw_e
    n_ttw_mat = 2 * n_ttw_select.T @ n_ttw_select 
    n_ttw_constant = 2*n_ttw_select.T @ zttw_n
    def g_eval(X):
        zttw_error = 1/(2*prob.rho_t)*(e_ttw_mat @ X - e_ttw_constant+
                            n_ttw_mat @ X - n_ttw_constant)
        return zttw_error
    return g_eval

def _g_adcp(prob):
    zadcp_e = mb.get_zadcp(prob.adat, 'east', prob.t_scale)
    zadcp_n = mb.get_zadcp(prob.adat, 'north', prob.t_scale)
    A_adcp, B_adcp = mb.adcp_select(prob.times, prob.depths, prob.ddat,
                                    prob.adat, prob.vehicle_vel)

    Vs = prob.Vs
    CV = prob.CV
    EV = prob.EV
    NV = prob.NV
    EC = prob.EC
    NC = prob.NC
    
    e_adcp_select = B_adcp @ CV @ EC - A_adcp @ Vs @ EV
    n_adcp_select = B_adcp @ CV @ NC - A_adcp @ Vs @ NV
    e_adcp_mat = 2* e_adcp_select.T @ e_adcp_select
    e_adcp_constant = 2 * e_adcp_select.T @ zadcp_e
    n_adcp_mat = 2* n_adcp_select.T @ n_adcp_select
    n_adcp_constant = 2 * n_adcp_select.T @ zadcp_n

    def g_eval(X):
        zadcp_error = 1/(2*prob.rho_a)*(
                e_adcp_mat @ X - e_adcp_constant +
                n_adcp_mat @ X - n_adcp_constant)
        return zadcp_error
    return g_eval

def _g_gps(prob):
    zgps_e = mb.get_zgps(prob.ddat, 'east')
    zgps_n = mb.get_zgps(prob.ddat, 'north')
    A_gps = mb.gps_select(prob.times, prob.ddat)

    EV = prob.EV
    NV = prob.NV
    Xs = prob.Xs
    
    e_gps_select = A_gps @ Xs @ EV
    n_gps_select = A_gps @ Xs @ NV
    e_gps_mat = 2* e_gps_select.T @ e_gps_select
    e_gps_constant = 2 * e_gps_select.T @ zgps_e
    n_gps_mat = 2* n_gps_select.T @ n_gps_select
    n_gps_constant = 2 * n_gps_select.T @ zgps_n

    def g_eval(X):
        zgps_error = 1/(2*prob.rho_g)*(e_gps_mat @ X - e_gps_constant +
                            n_gps_mat @ X - n_gps_constant)
        return zgps_error
    return g_eval
def _g_range(prob):
    zr, zx, zy = mb.get_zrange(prob.ddat)
    A_range = mb.range_select(prob.times, prob.ddat)

    EV = prob.EV
    NV = prob.NV
    Xs = prob.Xs
    
    A1 = A_range @ Xs @ EV  #e_range_select
    A2 = A_range @ Xs @ NV  #n_range_select

    def g_eval(X):
        if prob.rho_r != 0:
            denominator = np.sqrt((A1 @ X - zx)**2 +(A2 @ X - zy)**2)
            factor1 = (np.ones(len(zr)) - zr/denominator)
            factor2 = (2*A1.T.multiply(A1 @ X - zx) + 2*A2.T.multiply(A2 @ X - zy))
            range_error = 1/(2*prob.rho_r)*factor2 * factor1
        else: range_error = 0
        return range_error
    return g_eval

def g(prob, verbose=False):
    """Creates the sum of squares gradient function for fitting a 
    navigation and current profile
    
    Returns:
        vector-valued function for input & output of length m = 4*ts + 2*cs,
        where ts is the number of timepoints and cs is the number of depth
        points.
    """
    g1 = _g_kalman(prob)
    g2 = _g_ttw(prob)
    g3 = _g_adcp(prob)
    g4 = _g_gps(prob)
    g5 = _g_range(prob)

    def g_eval(X):
        return g1(X)+g2(X)+g3(X)+g4(X)+g5(X)
    return g_eval

# %%
def _h_kalman(prob):
    kalman_mat = gen_kalman_mat(prob)
    def h_eval(X):
        return 2* kalman_mat
    return h_eval
def _h_ttw(prob):
    A_ttw, B_ttw = mb.uv_select(prob.times, prob.depths, prob.ddat)

    Vs = prob.Vs
    CV = prob.CV
    EV = prob.EV
    NV = prob.NV
    EC = prob.EC
    NC = prob.NC

    e_ttw_select = A_ttw @ Vs @ EV - B_ttw @ CV @ EC
    n_ttw_select = A_ttw @ Vs @ NV - B_ttw @ CV @ NC
    e_ttw_mat = 2 * e_ttw_select.T @ e_ttw_select 
    n_ttw_mat = 2 * n_ttw_select.T @ n_ttw_select 
    def h_eval(X):
        zttw_error = 1/(2*prob.rho_t)*(e_ttw_mat + n_ttw_mat)
        return zttw_error
    return h_eval

def _h_adcp(prob):
    A_adcp, B_adcp = mb.adcp_select(prob.times, prob.depths, prob.ddat,
                                    prob.adat, prob.vehicle_vel)

    Vs = prob.Vs
    CV = prob.CV
    EV = prob.EV
    NV = prob.NV
    EC = prob.EC
    NC = prob.NC

    e_adcp_select = B_adcp @ CV @ EC - A_adcp @ Vs @ EV
    n_adcp_select = B_adcp @ CV @ NC - A_adcp @ Vs @ NV
    e_adcp_mat = 2* e_adcp_select.T @ e_adcp_select
    n_adcp_mat = 2* n_adcp_select.T @ n_adcp_select
    def h_eval(X):
        zadcp_error = 1/(2*prob.rho_a)*(e_adcp_mat + n_adcp_mat)
        return zadcp_error
    return h_eval

def _h_gps(prob):
    A_gps = mb.gps_select(prob.times, prob.ddat)

    EV = prob.EV
    NV = prob.NV
    Xs = prob.Xs

    e_gps_select = A_gps @ Xs @ EV
    n_gps_select = A_gps @ Xs @ NV
    e_gps_mat = 2* e_gps_select.T @ e_gps_select
    n_gps_mat = 2* n_gps_select.T @ n_gps_select

    def h_eval(X):
        zgps_error = 1/(2*prob.rho_g)*(e_gps_mat + n_gps_mat)
        return zgps_error
    return h_eval
def h(prob):
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
    h1 = _h_kalman(prob)
    h2 = _h_ttw(prob)
    h3 = _h_adcp(prob)
    h4 = _h_gps(prob)
    def h_eval(X):        
        return h1(X)+h2(X)+h3(X)+h4(X)
    return h_eval

# %% Tests
def complex_step_test(x0, eps, f, g):
    x0 = np.array(x0)
    g1 = g(x0)
    step = eps * 1j * np.ones(len(x0))
    repeatx0 = (np.repeat(x0.reshape(-1,1), len(x0), axis=1))
    add_mat = scipy.sparse.spdiags([step], (0), len(step), len(step))
    eval_mat = repeatx0 + add_mat
    fs = [np.imag(f(np.ravel(eval_mat[:,col]))) for col in range(len(step))]
    g2 = np.array(fs)/eps
    return g1, g2, np.linalg.norm(g1), np.linalg.norm(g2-g1)
    

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

def backsolve_test(x0, prob):
    """Tests whether the linear least squares solver agrees with the
    gradient function"""
    A, b = solve_mats(prob)
    grad_func = g(prob)
    v1 = A @ x0 + grad_func(np.zeros(len(x0)))
    v2 = grad_func(x0)
    return v1, v2, np.linalg.norm(v1), np.linalg.norm(v1-v2)

def solve(prob, method='L-BFGS-B', maxiter=50000, maxfun=50000):
    """Solve the ADCP navigation problem for given data."""
    x0 = init_x(prob)
    x0 = time_rescale(x0, 1/prob.t_scale, prob)
    ffunc = f(prob)
    gfunc = g(prob)
    hfunc = h(prob)
    sol = minimize(ffunc, x0, method=method, jac=gfunc, hess=hfunc,
                   options={'maxiter':maxiter, 'maxfun':maxfun, 'disp':True})
    sol.x = time_rescale(sol.x, prob.t_scale, prob)

    return sol