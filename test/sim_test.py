# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 11:04:36 2020

@author: 600301
"""

from adcp import dataprep as dp
from adcp import simulation as sim
from adcp import matbuilder as mb
from adcp import optimization as op
from adcp import viz

# %% ...or simulate new data
sp = sim.SimParams(rho_t=0, rho_a=0, rho_v=0, rho_c=0, sigma_t=.4, sigma_c=.3)
ddat, adat, x, curr_df, v_df = sim.simulate(sp, verbose=True)
depths = dp.depthpoints(adat, ddat)
times = dp.timepoints(adat, ddat)

# %% No Range
rho_v=1e-5
rho_c=1e-5
rho_g=1e-3
rho_t=1e-3
rho_a=1e-3
rho_r=0
print(f"""Solution method covariances:
    vehicle process: {rho_v}
    current process:{rho_c}
    GPS measurement: {rho_g}
    TTW measurement: {rho_t}
    ADCP meawsurement: {rho_a}""")

prob=op.GliderProblem(ddat, adat, rho_v=rho_v, rho_c=rho_c, rho_g=rho_g,
                      rho_t=rho_t, rho_a=rho_a, rho_r=rho_r)
# %%  Solve problem
seed = 3453

A, b = op.solve_mats(prob, verbose=True)
x0 = op.init_x(prob)
print(f'problem is of size {len(x0)}')
x_sol, (NV, EV, NC, EC, Xs, Vs) = op.backsolve(prob)

f = op.f(prob, verbose=True)
g = op.g(prob)
h = op.h(prob)
sol = op.solve(prob, method='L-BFGS-B', maxiter=50000, maxfun=50000)
print(f'LBFGS took {sol.nit} iterations and converged: {sol.success}')
#_, _, eps, err = op.grad_test(x0, 1e-2, f, g)
#print(f"Gradient test for vectors {eps:e} apart yeilds an error of {err:e}")
#_, _, eps, err = op.hess_test(x0, 1e-2, g, h)
#print(f"Hessian test for vectors {eps:e} apart yeilds an error of {err:e}")
#_, _, eps, err = op.backsolve_test(x0, prob)
#print(f"Backsolve test for gradient of norm {eps:e} yeilds an error of {err:e}")

# %% Plotting

#viz.current_plot(sol.x, x_sol, adat, times, depths)
ax1 = viz.vehicle_speed_plot(sol.x, ddat, times, depths, x_true=x)
ax2 = viz.current_depth_plot(sol.x, adat, ddat, direction='both', x_true=x, adcp=True)
ax3 = viz.vehicle_posit_plot(sol['x'], ddat, times, depths, x_true=x, dead_reckon=True)

#ax4 = viz.current_depth_plot(sol.x, adat, ddat, direction='both', mdat=mdat)

