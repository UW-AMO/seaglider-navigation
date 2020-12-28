# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 14:52:24 2020

@author: 600301
"""

from adcp import dataprep as dp
from adcp.simulation import simulate
from adcp import optimization as op
from adcp import viz

# %% Load existing data...

dive_num = 1980099
ddat = dp.load_dive(dive_num)
adat = dp.load_adcp(dive_num)
depths = dp.depthpoints(adat, ddat)
times = dp.timepoints(adat, ddat)

# %% Strip last GPS ovservation(s) for solution compare

num_to_remove = 1
ddat2 = ddat.copy()
ddat2["gps"] = ddat2["gps"].iloc[1 : -1 * num_to_remove, :]
depths2 = dp.depthpoints(adat, ddat2)
times2 = dp.timepoints(adat, ddat2)

# %% No Range
rho_v = 1e-7
rho_c = 1e-8
rho_g = 1e-3
rho_t = 1e-1
rho_a = 1e-1
rho_r = 0

prob = op.GliderProblem(
    ddat,
    adat,
    rho_v=rho_v,
    rho_c=rho_c,
    rho_g=rho_g,
    rho_t=rho_t,
    rho_a=rho_a,
    rho_r=rho_r,
)
prob2 = op.GliderProblem(
    ddat2,
    adat2,
    rho_v=rho_v,
    rho_c=rho_c,
    rho_g=rho_g,
    rho_t=rho_t,
    rho_a=rho_a,
    rho_r=rho_r,
)

# %%  Solve problem
seed = 3453


A, b = op.solve_mats(prob, verbose=True)
x0 = op.init_x(prob)
print(f"problem is of size {len(x0)}")
x_sol, (NV, EV, NC, EC, Xs, Vs) = op.backsolve(prob)

f = op.f(prob, verbose=True)
g = op.g(prob)
h = op.h(prob)
sol = op.solve(prob, method="L-BFGS-B")
print(f"LBFGS took {sol.nit} iterations and converged: {sol.success}")
_, _, eps, err = op.grad_test(x0, 1e-2, f, g)
print(f"Gradient test for vectors {eps:e} apart yeilds an error of {err:e}")
_, _, eps, err = op.hess_test(x0, 1e-2, g, h)
print(f"Hessian test for vectors {eps:e} apart yeilds an error of {err:e}")
_, _, eps, err = op.backsolve_test(x0, prob)
print(
    f"Backsolve test for gradient of norm {eps:e} yeilds an error of {err:e}"
)

# %%  Solve problem without last GPS hint
seed = 3453


# A, b = op.solve_mats(times, depths, ddat, adat, rho_v=rho_v, rho_c=rho_c,
#                      rho_t=rho_t, rho_a=rho_a, rho_g=rho_g, verbose=True)
x02 = op.init_x(prob2)
print(f"problem is of size {len(x02)}")
# x_sol, (NV, EV, NC, EC, Xs, Vs) = op.backsolve(ddat, adat, rho_v=rho_v,
#                                               rho_c=rho_c, rho_t=rho_t,
#                                               rho_a=rho_a, rho_g=rho_g)

f2 = op.f(prob2, verbose=True)
g2 = op.g(prob2)
h2 = op.h(prob2)
sol2 = op.solve(prob2, method="L-BFGS-B")
print(f"LBFGS took {sol.nit} iterations and converged: {sol.success}")
_, _, eps, err = op.grad_test(x02, 1e-2, f, g)
print(f"Gradient test for vectors {eps:e} apart yeilds an error of {err:e}")
_, _, eps, err = op.hess_test(x02, 1e-2, g, h)
print(f"Hessian test for vectors {eps:e} apart yeilds an error of {err:e}")


# %% Plotting

# viz.current_plot(sol.x, x_sol, adat, times, depths)
ax1 = viz.vehicle_speed_plot(sol.x, ddat, times, depths)
ax2 = viz.current_depth_plot(sol.x, adat, ddat, direction="both")
ax3 = viz.vehicle_posit_plot(sol["x"], ddat, times, depths, dead_reckon=True)

mdat = dp.load_mooring("CBX16_T3_AUG2017.mat")
ax4 = viz.current_depth_plot(sol.x, adat, ddat, direction="both", mdat=mdat)
