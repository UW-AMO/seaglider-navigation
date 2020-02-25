# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 18:44:43 2020

@author: 600301
"""
from matplotlib import pyplot as plt
import numpy as np

from adcp import dataprep as dp
from adcp import matbuilder as mb
from adcp import optimization as op

cmap = plt.get_cmap("tab10")


def current_depth_plot(x, adat, ddat, direction='north', mdat=None):
    """Produce a current by depth plot, showing the current during 
    descending and ascending measurements.  Optionally truncate to just the
    region when mooring ground truth data is available.
    """
    
    if direction.lower()=='both':
        plt.figure(figsize=[4,6])
        plt.subplot(1,2,1)
        current_depth_plot(x, adat, ddat, direction='north', mdat=mdat)
        plt.subplot(1,2,2)
        current_depth_plot(x, adat, ddat, direction='east', mdat=mdat)
        return
#    plt.figure()
    ax=plt.gca()
    ax.set_title('')
    
    # Preprocess to get mooring data, if necessary:
    if mdat is not None:
        first_time = ddat['depth'].index.min()
        last_time = ddat['depth'].index.max()
        first_truth_idx = np.argmin(np.abs(first_time-mdat['time']))
        last_truth_idx = np.argmin(np.abs(last_time-mdat['time']))
        first_truth_time = mdat['time'][first_truth_idx]
        last_truth_time = mdat['time'][last_truth_idx]
        true_currs = mdat['u'] if direction.lower()=='north' else mdat['v']
        ax.plot(true_currs[first_truth_idx,:], mdat['depth'][first_truth_idx,:],
                'bo', label='Descending-True')
        ax.plot(true_currs[last_truth_idx,:], mdat['depth'][last_truth_idx,:],
                'ro', label='Ascending-True')
#        ax.legend(loc='lower left')
        
    times = dp.timepoints(adat, ddat)
    depths = dp.depthpoints(adat, ddat)    
    m = len(times)
    n = len(depths)
    if direction.lower() in {'north','south'}:
        currs = op.nc_select(m, n) @ x
    elif direction.lower() in {'east','west'}:
        currs = op.ec_select(m, n) @ x

    depth_df = dp._depth_interpolator(times, ddat)
    turnaround = depth_df.ascending.idxmax()
    deepest = depth_df.loc[turnaround, 'depth']
    sinking = currs[(depths < deepest) & (depths >0)] 
    sinking_depths = depths[(depths < deepest) & (depths >0)] 
    rising = currs[(depths > deepest) & (depths < deepest*2)] 
    rising_depths = depths[(depths > deepest) & (depths < deepest*2)] 
    ax.plot(sinking, sinking_depths, 'b-', label='Descending-Inferred')
    ax.plot(rising, 2*deepest - rising_depths, 'r-', label='Ascending-Inferred')
#    ax.legend()
    ax.invert_yaxis()
    ax.set_title(direction.title()+'erly Current')
    ax.set_xlabel(f'Current (meters/sec)'.title())
    ax.set_ylabel('Depth (Meters)')
    plt.tight_layout()
    
    #Adjust ylim if just plotting surface
    if mdat is not None:
        max_depth = max(*mdat['depth'][first_truth_idx,:],
                        *mdat['depth'][last_truth_idx,:])
        ax.set_ylim(max_depth, 0)
    return ax

def vehicle_speed_plot(solx, ddat, times, depths, direction='north', 
                       x_sol=None, x0=None):
    plt.figure()
    m = len(times)
    n = len(depths)
    dirV = op.nv_select(m,n) if direction.lower()=='north' else op.ev_select(m,n)
    Vs = mb.v_select(m)
    cmap = plt.get_cmap("tab10")
    plt.title(f'{direction}ward Vehicle Velocity'.title())
    ln1 = plt.plot(times, Vs @ dirV @ solx, color=cmap(1), label='LBFGS Votg')
    lns = [ln1[0]]
    if x_sol is not None:
        ln2 = plt.plot(times, Vs @ dirV @ x_sol, color=cmap(0), label='backsolve Votg')
        lns.append(ln2[0])
    if x0 is not None:
        ln3 = plt.plot(times, Vs @ dirV @ x0, color=cmap(3), label='x0 Votg')
        lns.append(ln3[0])
#    ticks, labels = plt.xticks()
#    inds = np.linspace(0,len(ticks),6, dtype=int, endpoint=False)
#    plt.xticks(ticks[inds], np.array(labels)[inds])
    plt.twiny()
    ln4 = plt.plot(mb.get_zttw(ddat).values/1e3, color=cmap(2), label='TTW measured')
    lns.append(ln4[0])
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs)

def current_plot(solx, x_sol, adat, times, depths, direction='north'):
    plt.figure()
    m = len(times)
    n = len(depths)
    dirC = op.nc_select(m,n) if direction.lower()=='north' else op.ec_select(m,n)
    cmap = plt.get_cmap("tab10")
    plt.title(f'{direction}ward Current'.title())
    plt.plot(dirC @ x_sol, color=cmap(0), label='backsolve')
    plt.plot(dirC @ solx, color=cmap(1), label='LBFGS')
    plt.legend()
    plt.twiny()
    plt.plot(mb.get_zadcp(adat)/1e3, color=cmap(3), label='z_ttw')
    
def vehicle_posit_plot(x, ddat, times, depths, x0=None, backsolve=None,
                       dead_reckon=True):
    m = len(times)
    n = len(depths)
    NV = op.nv_select(m,n) 
    EV = op.ev_select(m,n)
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
    if dead_reckon:
        df = dp.dead_reckon(ddat)
        ln4 = ax.plot(df.x_dead/1000, df.y_dead/1000, label='Dead Reckoning')
        ln5 = ax.plot(df.x_corr/1000, df.y_corr/1000, label='Reckoning, Corrected')
        lns.append(ln4[0])
        lns.append(ln5[0])
    ax.legend()
    ax.set_xlabel('Easting (km)')
    ax.set_ylabel('Northing (km)')