# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 14:52:24 2020

@author: 600301
"""

from matplotlib import pyplot as plt
from scipy.optimize import minimize
import numpy as np

from adcp import dataprep as dp
from adcp import matbuilder as mb
from adcp.simulation import simulate
from adcp import optimization as op
from adcp import viz

# %% Load existing data...

dive_num = 1980099
ddat = dp.load_dive(dive_num)
adat = dp.load_adcp(dive_num)
depths = dp.depthpoints(adat, ddat)
times = dp.timepoints(adat, ddat)

# %% ...or simulate new data
ddat, adat, x, v_df = simulate(n_timepoints=1000, rho_v=1e-1, rho_c=1e-2,
                               rho_t=1e-1, rho_a=1e-1, rho_g=1e0,
                               sigma_t=.3, sigma_c=.3, seed=321)
depths = dp.depthpoints(adat, ddat)
times = dp.timepoints(adat, ddat)

# %% Strip last GPS ovservation(s) for solution compare

num_to_remove = 1
ddat2 = ddat.copy()
ddat2['gps'] = ddat2['gps'].iloc[1:-1*num_to_remove,:]
depths2 = dp.depthpoints(adat, ddat2)
times2 = dp.timepoints(adat, ddat2)

# %% Adjust parameters here to re-test different solver parameters on same data
#Range only
rho_v=1e3
rho_c=1e3
rho_g=1e3
rho_t=1e3
rho_a=1e3
rho_r=1e-5
# %% No Range
rho_v=1e-7
rho_c=1e-8
rho_g=1e-3
rho_t=1e-1
rho_a=1e-1
rho_r=0
# %%  Solve problem
seed = 3453


#A, b = op.solve_mats(times, depths, ddat, adat, rho_v=rho_v, rho_c=rho_c,
#                      rho_t=rho_t, rho_a=rho_a, rho_g=rho_g, verbose=True)
x0 = op.init_x(times, depths, ddat)
print(f'problem is of size {len(x0)}')
#x_sol, (NV, EV, NC, EC, Xs, Vs) = op.backsolve(ddat, adat, rho_v=rho_v, 
#                                               rho_c=rho_c, rho_t=rho_t,
#                                               rho_a=rho_a, rho_g=rho_g)

f = op.f(times, depths, ddat, adat, rho_v=rho_v, rho_c=rho_c, rho_a=rho_a,
         rho_t=rho_t, rho_g=rho_g, verbose=True)
g = op.g(times, depths, ddat, adat, rho_v=rho_v, rho_c=rho_c, rho_a=rho_a,
         rho_t=rho_t, rho_g=rho_g)
h = op.h(times, depths, ddat, adat, rho_v=rho_v, rho_c=rho_c, rho_a=rho_a,
         rho_t=rho_t, rho_g=rho_g)
sol = op.solve(ddat, adat, rho_v=rho_v, rho_c=rho_c, rho_t=rho_t, rho_a=rho_a,
               rho_g=rho_g, rho_r=rho_r, method='L-BFGS-B')
print(f'LBFGS took {sol.nit} iterations and converged: {sol.success}')
_, _, eps, err = op.grad_test(x0, 1e-2, f, g)
print(f"Gradient test for vectors {eps:e} apart yeilds an error of {err:e}")
_, _, eps, err = op.hess_test(x0, 1e-2, g, h)
print(f"Hessian test for vectors {eps:e} apart yeilds an error of {err:e}")
#_, _, eps, err = op.backsolve_test(x0, ddat, adat, rho_v=rho_v, rho_c=rho_c,
#                                   rho_t=rho_t, rho_a=rho_a, rho_g=rho_g)
#print(f"Backsolve test for gradient of norm {eps:e} yeilds an error of {err:e}")

# %%  Solve problem without last GPS hint
seed = 3453


#A, b = op.solve_mats(times, depths, ddat, adat, rho_v=rho_v, rho_c=rho_c,
#                      rho_t=rho_t, rho_a=rho_a, rho_g=rho_g, verbose=True)
x02 = op.init_x(times2, depths2, ddat2)
print(f'problem is of size {len(x0)}')
#x_sol, (NV, EV, NC, EC, Xs, Vs) = op.backsolve(ddat, adat, rho_v=rho_v,
#                                               rho_c=rho_c, rho_t=rho_t,
#                                               rho_a=rho_a, rho_g=rho_g)

f2 = op.f(times2, depths2, ddat2, adat, rho_v=rho_v, rho_c=rho_c, rho_a=rho_a,
         rho_t=rho_t, rho_g=rho_g, verbose=True)
g2 = op.g(times2, depths2, ddat2, adat, rho_v=rho_v, rho_c=rho_c, rho_a=rho_a,
         rho_t=rho_t, rho_g=rho_g)
h2 = op.h(times2, depths2, ddat2, adat, rho_v=rho_v, rho_c=rho_c, rho_a=rho_a,
         rho_t=rho_t, rho_g=rho_g)
sol2 = op.solve(ddat2, adat, rho_v=rho_v, rho_c=rho_c, rho_t=rho_t, rho_a=rho_a,
               rho_g=rho_g, rho_r=rho_r, method='L-BFGS-B')
print(f'LBFGS took {sol.nit} iterations and converged: {sol.success}')
_, _, eps, err = op.grad_test(x02, 1e-2, f, g)
print(f"Gradient test for vectors {eps:e} apart yeilds an error of {err:e}")
_, _, eps, err = op.hess_test(x02, 1e-2, g, h)
print(f"Hessian test for vectors {eps:e} apart yeilds an error of {err:e}")


# %% Plotting

#viz.current_plot(sol.x, x_sol, adat, times, depths)
ax1 = viz.vehicle_speed_plot(sol.x, ddat, times, depths)
ax2 = viz.current_depth_plot(sol.x, adat, ddat, direction='both')
ax3 = viz.vehicle_posit_plot(sol['x'], ddat, times, depths, dead_reckon=True)

mdat = dp.load_mooring('CBX16_T3_AUG2017.mat')
ax4 = viz.current_depth_plot(sol.x, adat, ddat, direction='both', mdat=mdat)
