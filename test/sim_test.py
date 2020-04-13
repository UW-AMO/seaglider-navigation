# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 11:04:36 2020

@author: 600301
"""

from adcp import dataprep as dp
from adcp import simulation as sim
from adcp import optimization as op
from adcp import viz

# %% ...or simulate new data
sp = sim.SimParams()
ddat, adat, x = sim.simulate(sp)
depths = dp.depthpoints(adat, ddat)
times = dp.timepoints(adat, ddat)

# %% No Range
rho_v=1e-7
rho_c=1e-8
rho_g=1e-3
rho_t=1e-1
rho_a=1e-1
rho_r=0

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
sol = op.solve(prob, method='L-BFGS-B')
print(f'LBFGS took {sol.nit} iterations and converged: {sol.success}')
_, _, eps, err = op.grad_test(x0, 1e-2, f, g)
print(f"Gradient test for vectors {eps:e} apart yeilds an error of {err:e}")
_, _, eps, err = op.hess_test(x0, 1e-2, g, h)
print(f"Hessian test for vectors {eps:e} apart yeilds an error of {err:e}")
_, _, eps, err = op.backsolve_test(x0, prob)
print(f"Backsolve test for gradient of norm {eps:e} yeilds an error of {err:e}")

# %% Plotting

#viz.current_plot(sol.x, x_sol, adat, times, depths)
ax1 = viz.vehicle_speed_plot(sol.x, ddat, times, depths, x_true=x)
ax2 = viz.current_depth_plot(sol.x, adat, ddat, direction='both', x_true=x)
ax3 = viz.vehicle_posit_plot(sol['x'], ddat, times, depths, x_true=x, dead_reckon=True)

#ax4 = viz.current_depth_plot(sol.x, adat, ddat, direction='both', mdat=mdat)

