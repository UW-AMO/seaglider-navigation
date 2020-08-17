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
sp = sim.SimParams(rho_t=0, rho_a=0, rho_v=0, rho_c=0, sigma_t=.4, sigma_c=.3,
                   measure_points=dict(gps='endpoints', ttw=.5, range=.05,),
                   curr_method='curved', vehicle_method='curved',)
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
sol = op.solve(prob, method='L-BFGS-B', maxiter=1000000, maxfun=1000000)
print(f'LBFGS took {sol.nit} iterations and converged: {sol.success}')
#_, _, eps, err = op.grad_test(x0, 1e-2, f, g)
#print(f"Gradient test for vectors {eps:e} apart yeilds an error of {err:e}")
#_, _, eps, err = op.hess_test(x0, 1e-2, g, h)
#print(f"Hessian test for vectors {eps:e} apart yeilds an error of {err:e}")
#_, _, eps, err = op.backsolve_test(x0, prob)
#print(f"Backsolve test for gradient of norm {eps:e} yeilds an error of {err:e}")

# %% Plotting

#viz.current_plot(sol.x, x_sol, adat, times, depths)
ax1 = viz.vehicle_speed_plot(sol.x, ddat, times, depths, direction='both',
                             x_sol=x_sol, x_true=x, ttw=False)
ax2 = viz.inferred_ttw_error_plot(sol.x, adat, ddat, direction='both',
                                  x_true=x, x_sol=x_sol)
ax3 = viz.current_depth_plot(sol.x, adat, ddat, direction='both', x_true=x,
                             x_sol=x_sol, adcp=True)
ax4 = viz.inferred_adcp_error_plot(sol.x, adat, ddat, direction='both',
                                   x_true=x, x_sol=x_sol)
ax5 = viz.vehicle_posit_plot(sol['x'], ddat, times, depths, backsolve=x_sol,
                              x_true=x, dead_reckon=True)

#ax4 = viz.current_depth_plot(sol.x, adat, ddat, direction='both', mdat=mdat)

# %% Validation

import numpy as np
print(sol.message)

m = len(times)
n = len(depths)

x_native = op.time_rescale(x, 1/mb.t_scale, m, n)
back_native = op.time_rescale(x_sol, 1/mb.t_scale, m, n)
sol_native = op.time_rescale(sol.x, 1/mb.t_scale, m, n)
x0_native = op.time_rescale(x0, 1/mb.t_scale, m, n)

print('LBFGS solution better than true value: ', f(sol_native) < f(x_native))
print('LBFGS solution better than starting point: ', f(sol_native) < f(x0_native))
print('LBFGS solution better than backsolve solution: ', f(sol_native) < f(back_native))
print('LBFGS solution distance from backsolve solution (rel): ',
      np.linalg.norm(sol_native-back_native)/np.linalg.norm(sol_native))
print('LBFGS solution distance from starting solution (rel): ',
      np.linalg.norm(sol_native-x0_native)/np.linalg.norm(sol_native))


from adcp.optimization import (_f_kalman, _f_ttw, _f_adcp, _f_gps, _f_range,
                               _g_kalman, _g_ttw, _g_adcp, _g_gps, _g_range,
                               _h_kalman, _h_ttw, _h_adcp, _h_gps, )#_h_range)

f1 = _f_kalman(prob)
f2 = _f_ttw(prob)
f3 = _f_adcp(prob)
f4 = _f_gps(prob)
f5 = _f_range(prob)

g1 = _g_kalman(prob)
g2 = _g_ttw(prob)
g3 = _g_adcp(prob)
g4 = _g_gps(prob)
g5 = _g_range(prob)

h1 = _h_kalman(prob)
h2 = _h_ttw(prob)
h3 = _h_adcp(prob)
h4 = _h_gps(prob)


###################################
####### Gradient Testing ##########
###################################
delta = 1e-2
_, _, eps, err = op.grad_test(x_native, delta, f, g)
print(f"Overall gradient test for vectors {eps:e} apart yeilds an error of {err:e}")
_, _, eps, err = op.grad_test(x_native, delta, f1, g1)
print(f"Kalman gradient test for vectors {eps:e} apart yeilds an error of {err:e}")
_, _, eps, err = op.grad_test(x_native, delta, f2, g2)
print(f"TTW measurement gradient test for vectors {eps:e} apart yeilds an error of {err:e}")
_, _, eps, err = op.grad_test(x_native, delta, f3, g3)
print(f"ADCP measurement gradient test for vectors {eps:e} apart yeilds an error of {err:e}")
_, _, eps, err = op.grad_test(x_native, delta, f4, g4)
print(f"GPS measurement gradient test for vectors {eps:e} apart yeilds an error of {err:e}")
#_, _, eps, err = op.grad_test(x_native, delta, f5, g5)
#print(f"Range measurement gradient test for vectors {eps:e} apart yeilds an error of {err:e}")

_, _, eps, err = op.complex_step_test(x_native, delta, f, g)
print(f"Overall complex step test for gradient of norm {eps:e} yeilds an error of {err:e}")
_, _, eps, err = op.complex_step_test(x_native, delta, f1, g1)
print(f"Kalman complex step test for gradient of norm {eps:e} yeilds an error of {err:e}")
_, _, eps, err = op.complex_step_test(x_native, delta, f2, g2)
print(f"TTW measurement complex step test for gradient of norm {eps:e} yeilds an error of {err:e}")
_, _, eps, err = op.complex_step_test(x_native, delta, f3, g3)
print(f"ADCP measurement complex step test for gradient of norm {eps:e} yeilds an error of {err:e}")
_, _, eps, err = op.complex_step_test(x_native, delta, f4, g4)
print(f"GPS measurement complex step test for gradient of norm {eps:e} yeilds an error of {err:e}")
#_, _, eps, err = op.complex_step_test(x_native, delta, f5, g5)
#print(f"Range measurement complex step test for gradient of norm {eps:e} yeilds an error of {err:e}")

_, _, eps, err = op.hess_test(x_native, delta, g, h)
print(f"Overall hessian test for vectors {eps:e} apart yeilds an error of {err:e}")
_, _, eps, err = op.hess_test(x_native, delta, g1, h1)
print(f"Kalman hessian test for vectors {eps:e} apart yeilds an error of {err:e}")
_, _, eps, err = op.hess_test(x_native, delta, g2, h2)
print(f"TTW measurement hessian test for vectors {eps:e} apart yeilds an error of {err:e}")
_, _, eps, err = op.hess_test(x_native, delta, g3, h3)
print(f"ADCP measurement hessian test for vectors {eps:e} apart yeilds an error of {err:e}")
_, _, eps, err = op.hess_test(x_native, delta, g4, h4)
print(f"GPS measurement hessian test for vectors {eps:e} apart yeilds an error of {err:e}")
_, _, eps, err = op.hess_test(x_native, delta, g4, h4)
print(f"GPS measurement hessian test for vectors {eps:e} apart yeilds an error of {err:e}")
#_, _, eps, err = op.hess_test(x_native, delta, g5, h5)
#print(f"Range measurement hessian test for vectors {eps:e} apart yeilds an error of {err:e}")

_, _, eps, err = op.backsolve_test(x_native, prob)
print(f"Backsolve test for gradient of norm {eps:e} yeilds an error of {err:e}")
