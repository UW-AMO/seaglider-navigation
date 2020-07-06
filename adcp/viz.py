# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 18:44:43 2020

@author: 600301
"""
from matplotlib import pyplot as plt
import numpy as np

from adcp import dataprep as dp
from adcp import matbuilder as mb

cmap = plt.get_cmap("tab10")

# %%
def inferred_adcp_error_plot(solx, adat, ddat, direction='north', x_true=None,
                             x_sol=None):
    if direction.lower()=='both':
        plt.figure(figsize=[12,6])
        plt.subplot(1,2,1)
        ax1 = inferred_adcp_error_plot(solx, adat, ddat, direction='north',
                                 x_true=x_true, x_sol=x_sol)
        plt.subplot(1,2,2)
        ax2 = inferred_adcp_error_plot(solx, adat, ddat, direction='east',
                                 x_true=x_true, x_sol=x_sol)
        return ax1, ax2

    ax = plt.gca()
    ax.set_title(direction.title()+' Shear velocities')
    ax.set_xlabel('meters/second')
    ax.set_ylabel('depth')
    times = dp.timepoints(adat, ddat)
    depths = dp.depthpoints(adat, ddat)
    m = len(times)
    n = len(depths)
    Vs= mb.v_select(m)
    if direction.lower() in {'north','south'}:
        zadcp = mb.get_zadcp(adat, 'north')/mb.t_scale
        XV = mb.nv_select(m, n)
        XC = mb.nc_select(m, n)
    elif direction.lower() in {'east','west'}:
        zadcp = mb.get_zadcp(adat, 'east')/mb.t_scale
        XV = mb.ev_select(m, n)
        XC = mb.ec_select(m, n)
    else:
        raise ValueError

    depth_df = dp._depth_interpolator(times, ddat)
    turnaround = depth_df.ascending.idxmax()
    deepest = depth_df.loc[turnaround, 'depth']

    A, B = mb.adcp_select(times, depths, ddat, adat)
    sinking_meas = zadcp[(B @ depths < deepest) & (B @ depths >0)]
    sinking_depths = depths[np.array(B.sum(axis=0)).squeeze().astype(bool)
                            & (depths < deepest)
                            & (depths >0)]
    rising_meas = zadcp[(B @ depths > deepest) & (B @ depths < deepest*2)]
    rising_depths = depths[np.array(B.sum(axis=0)).squeeze().astype(bool)
                            & (depths > deepest)
                            & (depths < deepest*2)] 
    ln0 = ax.plot(sinking_meas, sinking_depths, 'k-', label='Descending-measured')
    ln1 = ax.plot(rising_meas, 2*deepest - rising_depths, 'g-', label='Ascending-measured')
    lines = [ln0, ln1]

    adcp_lbfgs = (B @ XC - A @ Vs @ XV) @ solx
    sinking_lbfgs = adcp_lbfgs[(B @ depths < deepest) & (B @ depths >0)]
    rising_lbfgs = adcp_lbfgs[(B @ depths > deepest) & (B @ depths < deepest*2)]
    ln2 = ax.plot(sinking_lbfgs, sinking_depths, 'b--', label='Descending-LBFGS')
    ln3 = ax.plot(rising_lbfgs, 2*deepest - rising_depths, 'r--', label='Ascending-LBFGS')    

    lines = [*lines, ln2, ln3]

    if x_true is not None:
        adcp_true = (B @ XC - A @ Vs @ XV) @ x_true
        sinking_true = adcp_true[(B @ depths < deepest) & (B @ depths >0)]
        rising_true = adcp_true[(B @ depths > deepest) & (B @ depths < deepest*2)]
        ln4 = ax.plot(sinking_true, sinking_depths, 'b--', label='Descending-true')
        ln5 = ax.plot(rising_true, 2*deepest - rising_depths, 'r--', label='Ascending-true')
        lines = [*lines, ln4, ln5]

    if x_sol is not None:
        adcp_back = (B @ XC - A @ Vs @ XV) @ x_sol
        sinking_back = adcp_back[(B @ depths < deepest) & (B @ depths >0)]
        rising_back = adcp_back[(B @ depths > deepest) & (B @ depths < deepest*2)]
        ln6 = ax.plot(sinking_back, sinking_depths, 'b--', label='Descending-baksolve')
        ln7 = ax.plot(rising_back, 2*deepest - rising_depths, 'r--', label='Ascending-backsolve')
        lines = [*lines, ln6, ln7]

    ax.legend()
    return ax

def inferred_ttw_error_plot(solx, adat, ddat, direction='north', x_true=None,
                             x_sol=None):

    if direction.lower()=='both':
        plt.figure(figsize=[12,6])
        plt.subplot(1,2,1)
        ax1 = inferred_ttw_error_plot(solx, adat, ddat, direction='north',
                                 x_true=x_true, x_sol=x_sol)
        plt.subplot(1,2,2)
        ax2 = inferred_ttw_error_plot(solx, adat, ddat, direction='east',
                                 x_true=x_true, x_sol=x_sol)
        return ax1, ax2

    ax = plt.gca()
    ax.set_title(direction.title()+' TTW velocities')
    ax.set_xlabel('Time')
    ax.set_ylabel('meters/second')
    times = dp.timepoints(adat, ddat)
    depths = dp.depthpoints(adat, ddat)
    m = len(times)
    n = len(depths)
    Vs= mb.v_select(m)
    if direction.lower() in {'north','south'}:
        zttw = mb.get_zttw(ddat, 'north')
        XV = mb.nv_select(m, n)
        XC = mb.nc_select(m, n)
    elif direction.lower() in {'east','west'}:
        zttw = mb.get_zttw(ddat, 'east')
        XV = mb.ev_select(m, n)
        XC = mb.ec_select(m, n)
    else:
        raise ValueError

    A, B = mb.uv_select(times, depths, ddat)

    ln0 = ax.plot(zttw.index, zttw.values/1e3, 'b-',
                  label='TTW measured')

    ttw_lbfgs = (A @ Vs @ XV - B @ XC) @ solx
    ln1 = ax.plot(zttw.index, ttw_lbfgs, 'r--', label='LBFGS')

    lines = [ln0, ln1]

    if x_true is not None:
        ttw_true = (A @ Vs @ XV - B @ XC) @ x_true
        ln2 = ax.plot(zttw.index, ttw_true, 'g--', label='True')
        lines = [*lines, ln2]

    if x_sol is not None:
        ttw_back = (A @ Vs @ XV - B @ XC) @ x_sol
        ln3 = ax.plot(zttw.index, ttw_back, 'k:', label='Backsolve')
        lines = [*lines, ln3]

    ax.legend()
    return ax

# %%
def current_depth_plot(solx, adat, ddat, direction='north', x_true=None,
                       x_sol=None, mdat=None, adcp=False):
    """Produce a current by depth plot, showing the current during 
    descending and ascending measurements.  Various other options.

    Parameters:
        solx (numpy.array): LBFGS solution for state vector
        adat (dict): ADCP data. See dataprep.load_adcp() or
            simulation.construct_load_dicts()
        ddat (dict): dive data. See dataprep.load_dive() or
            simulation.construct_load_dicts()
        direction (str): either 'north' or 'south'
        x_true (numpy.array): true state vector
        x_sol (numpy.array): backsolve solution for state vector
        mdat (dict): Moored ADCP measurements.  See
            dataprep.load_mooring()
        adat (bool): Whether to include ADCP measurement data or not
    """
    
    if direction.lower()=='both':
        plt.figure(figsize=[12,6])
        plt.subplot(1,2,1)
        ax1 = current_depth_plot(solx, adat, ddat, direction='north',
                                 x_true=x_true, x_sol=x_sol, mdat=mdat,
                                 adcp=adcp)
        plt.subplot(1,2,2)
        ax2 = current_depth_plot(solx, adat, ddat, direction='east',
                                 x_true=x_true, x_sol=x_sol, mdat=mdat,
                                 adcp=adcp)
        return ax1, ax2
    ax=plt.gca()
    ax.set_title('')
    times = dp.timepoints(adat, ddat)
    depths = dp.depthpoints(adat, ddat)    
    m = len(times)
    n = len(depths)
    if direction.lower() in {'north','south'}:
        currs = mb.nc_select(m, n) @ solx
    elif direction.lower() in {'east','west'}:
        currs = mb.ec_select(m, n) @ solx
    depth_df = dp._depth_interpolator(times, ddat)
    turnaround = depth_df.ascending.idxmax()
    deepest = depth_df.loc[turnaround, 'depth']
    sinking = currs[(depths < deepest) & (depths >0)] 
    sinking_depths = depths[(depths < deepest) & (depths >0)] 
    rising = currs[(depths > deepest) & (depths < deepest*2)] 
    rising_depths = depths[(depths > deepest) & (depths < deepest*2)] 
    ln0 = ax.plot(sinking, sinking_depths, 'b--', 
                  label='Descending-Inferred')
    ln1 = ax.plot(rising, 2*deepest - rising_depths, 'r--',
                  label='Ascending-Inferred')
    lines = [ln0, ln1]

    #ADCP traces
    if adcp:
        if direction.lower() in {'north','south'}:
            zadcp = mb.get_zadcp(adat, 'north')/mb.t_scale
        elif direction.lower() in {'east','west'}:
            zadcp = mb.get_zadcp(adat, 'east')/mb.t_scale
        _, B_adcp = mb.adcp_select(times, depths, ddat, adat)
        sinking_a = zadcp[(B_adcp @ depths < deepest) & (B_adcp @ depths >0)]
        sinking_depths_a = depths[np.array(B_adcp.sum(axis=0)).squeeze().astype(bool)
                                & (depths < deepest)
                                & (depths >0)]
        rising_a = zadcp[(B_adcp @ depths > deepest) & (B_adcp @ depths < deepest*2)] 
        rising_depths_a = depths[np.array(B_adcp.sum(axis=0)).squeeze().astype(bool)
                                & (depths > deepest)
                                & (depths < deepest*2)] 
        lna0 = ax.plot(sinking_a, sinking_depths_a, 'k:', label='Descending-ADCP')
        lna1 = ax.plot(rising_a, 2*deepest - rising_depths_a, 'g:', label='Ascending-ADCP')
        lines = [*lines, lna0, lna1]
    # Add in true simulated profiles, if available
    if x_true is not None:
        if direction.lower() in {'north','south'}:
            true_currs = mb.nc_select(m, n) @ x_true
        elif direction.lower() in {'east','west'}:
            true_currs = mb.ec_select(m, n) @ x_true
        sinking_true = true_currs[(depths < deepest) & (depths >0)] 
        rising_true = true_currs[(depths > deepest) & (depths < deepest*2)]
        ln2 = ax.plot(sinking_true, sinking_depths, 'b-',
                label='Descending-True')
        ln3 = ax.plot(rising_true, 2*deepest - rising_depths, 'r-',
                label='Ascending-True')
        lines = [*lines, ln2, ln3]
    #Add in backsolve solution, if available
    if x_sol is not None:
        if direction.lower() in {'north','south'}:
            true_currs = mb.nc_select(m, n) @ x_true
        elif direction.lower() in {'east','west'}:
            true_currs = mb.ec_select(m, n) @ x_true
        sinking_true = true_currs[(depths < deepest) & (depths >0)]
        rising_true = true_currs[(depths > deepest) & (depths < deepest*2)]
        ln4 = ax.plot(sinking_true, sinking_depths, 'c--',
                label='Descending-Backsolve')
        ln5 = ax.plot(rising_true, 2*deepest - rising_depths, 'y--',
                label='Ascending-Backsolve')
        lines = [*lines, ln4, ln5]
        
    # Preprocess to get mooring data, if necessary:
    if mdat is not None:
        first_time = ddat['depth'].index.min()
        last_time = ddat['depth'].index.max()
        first_truth_idx = np.argmin(np.abs(first_time-mdat['time']))
        last_truth_idx = np.argmin(np.abs(last_time-mdat['time']))
        true_currs = mdat['u'] if direction.lower()=='north' else mdat['v']
        ln6 = ax.plot(true_currs[first_truth_idx,:], mdat['depth'][first_truth_idx,:],
                'bo', label='Descending-Mooring')
        ln7 = ax.plot(true_currs[last_truth_idx,:], mdat['depth'][last_truth_idx,:],
                'ro', label='Ascending-Mooring')
#        ax.legend(loc='lower left')
        lines = [*lines, ln6, ln7]

    ax.legend()
    ax.invert_yaxis()
    ax.set_title(direction.title()+'erly Current')
    ax.set_xlabel('Current (meters/sec)'.title())
    ax.set_ylabel('Depth (Meters)')
    plt.tight_layout()
    
    #Adjust ylim if just plotting surface
    if mdat is not None:
        max_depth = max(*mdat['depth'][first_truth_idx,:],
                        *mdat['depth'][last_truth_idx,:])
        ax.set_ylim(max_depth, 0)
    
    return lines

# %%
def vehicle_speed_plot(solx, ddat, times, depths, direction='north', 
                       x_sol=None, x_true=None, x0=None):
    """Plots the vehicle's solved speed, optionally with different
    comparison solutions.

    Parameters:
        solx (numpy.array): LBFGS solution for state vector
        ddat (dict): dive data. See dataprep.load_dive() or
            simulation.construct_load_dicts()
        times (numpy.array): times at which a measurement occured
        depths (numpy.array): depths at which a measurement occured
        direction (str): either 'north' or 'south'
        x_sol (numpy.array): backsolve solution for state vector
        x_true (numpy.array): true state vector
        x0 (numpy.array): starting state vector for LBFGS
    """
    if direction.lower()=='both':
        plt.figure(figsize=[12,6])
        plt.subplot(1,2,1)
        ax1 = vehicle_speed_plot(solx, ddat, times, depths, direction='north',
                                 x_true=x_true, x_sol=x_sol, x0=x0)
        plt.subplot(1,2,2)
        ax2 = vehicle_speed_plot(solx, ddat, times, depths, direction='east',
                                 x_true=x_true, x_sol=x_sol, x0=x0)
        return ax1, ax2

    ax = plt.gca()
    m = len(times)
    n = len(depths)
    dirV = mb.nv_select(m,n) if direction.lower()=='north' else mb.ev_select(m,n)
    Vs = mb.v_select(m)
    cmap = plt.get_cmap("tab10")
    ax.set_title(f'{direction}ward Vehicle Velocity'.title())
    ln1 = ax.plot(times, Vs @ dirV @ solx, color=cmap(1), label='LBFGS Votg')
    lns = [ln1[0]]
    if x_sol is not None:
        ln2 = ax.plot(times, Vs @ dirV @ x_sol, color=cmap(0), label='backsolve Votg')
        lns.append(ln2[0])
    if x0 is not None:
        ln3 = ax.plot(times, Vs @ dirV @ x0, color=cmap(3), label='x0 Votg')
        lns.append(ln3[0])
    if x_true is not None:
        ln4 = ax.plot(times, Vs @ dirV @ x_true, color=cmap(3), label='Votg_true')
        lns.append(ln4[0])
    ln5 = ax.plot(mb.get_zttw(ddat).index,
                   mb.get_zttw(ddat).values/1e3,
                   color=cmap(2), label='TTW measured')
    lns.append(ln5[0])
    ax.set_ylabel('meters/second')
    ax.set_xlabel('time')
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs)
    return ax

def current_plot(solx, x_sol, adat, times, depths, direction='north'):
    """ Deprecated """
    fig = plt.figure()
    m = len(times)
    n = len(depths)
    dirC = mb.nc_select(m,n) if direction.lower()=='north' else mb.ec_select(m,n)
    ax = fig.gca()
    cmap = ax.get_cmap("tab10")
    ax.set_title(f'{direction}ward Current'.title())
    ax.plot(dirC @ x_sol, color=cmap(0), label='backsolve')
    ax.plot(dirC @ solx, color=cmap(1), label='LBFGS')
    ax.legend()
    ax.twiny()
    ax.plot(mb.get_zadcp(adat)/1e3, color=cmap(3), label='z_ttw')
    return ax

def vehicle_posit_plot(x, ddat, times, depths, x0=None, backsolve=None,
                       x_true=None, dead_reckon=True):
    """Plots the vehicle's position in x-y coordinates, optionally
    with different comparison solutions.

         Parameters:
         x (numpy.array): LBFGS solution for state vector
         ddat (dict): dive data. See dataprep.load_dive() or
             simulation.construct_load_dicts()
         times (numpy.array): times at which a measurement occured
         depths (numpy.array): depths at which a measurement occured
         x0 (numpy.array): starting state vector for LBFGS
         backsolve (numpy.array): backsolve solution for state vector
         x_true (numpy.array): true state vector
         dead_reckon (bool): Whether to include dead reckoning solution,
             treating TTW measurements as over-the-ground truth.
    """
    m = len(times)
    n = len(depths)
    NV = mb.nv_select(m,n)
    EV = mb.ev_select(m,n)
    Xs = mb.x_select(m)    
    plt.figure()
    ax=plt.gca()
    ax.set_title('Vehicle Position')
    ln1 = ax.plot(Xs @ EV @ x/1000, Xs @ NV @ x/1000, label='Optimization Solution')
    lns = [ln1[0]]
    if backsolve is not None:
        ln2 = ax.plot(Xs @ EV @ backsolve/1000, Xs @ NV @ backsolve/1000, 
                      label='backsolve')
        lns.append(ln2[0])
    if x0 is not None:
        ln3 = ax.plot(Xs @ EV @ x0/1000, Xs @ NV @ x0/1000, label='x0')
        lns.append(ln3[0])
    if x_true is not None:
        ln4 = ax.plot(Xs @ EV @ x_true/1000, Xs @ NV @ x_true/1000, label='X_true')
        lns.append(ln4[0])
    if dead_reckon:
        df = dp.dead_reckon(ddat)
        ln5 = ax.plot(df.x_dead/1000, df.y_dead/1000, label='Dead Reckoning')
        ln6 = ax.plot(df.x_corr/1000, df.y_corr/1000, label='Reckoning, Corrected')
        lns.append(ln5[0])
        lns.append(ln6[0])
    ax.legend()
    ax.set_xlabel('Easting (km)')
    ax.set_ylabel('Northing (km)')
    return ax