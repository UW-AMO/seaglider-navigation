# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 11:37:53 2020

@author: 600301
made to run validation tests on the results of sim_test.  Requires
variables in namespace from that run.
"""
import numpy as np

print(sol.message)

m = len(times)
n = len(depths)

x_native = op.time_rescale(x, 1 / mb.t_scale, m, n)
back_native = op.time_rescale(x_sol, 1 / mb.t_scale, m, n)
sol_native = op.time_rescale(sol.x, 1 / mb.t_scale, m, n)
x0_native = op.time_rescale(x0, 1 / mb.t_scale, m, n)

print("LBFGS solution better than true value: ", f(sol_native) < f(x_native))
print(
    "LBFGS solution better than starting point: ", f(sol_native) < f(x0_native)
)
print(
    "LBFGS solution better than backsolve solution: ",
    f(sol_native) < f(back_native),
)
print(
    "LBFGS solution distance from backsolve solution (rel): ",
    np.linalg.norm(sol_native - back_native) / np.linalg.norm(sol_native),
)
print(
    "LBFGS solution distance from starting solution (rel): ",
    np.linalg.norm(sol_native - x0_native) / np.linalg.norm(sol_native),
)


from adcp.optimization import (
    _f_kalman,
    _f_ttw,
    _f_adcp,
    _f_gps,
    _f_range,
    _g_kalman,
    _g_ttw,
    _g_adcp,
    _g_gps,
    _g_range,
    _h_kalman,
    _h_ttw,
    _h_adcp,
    _h_gps,
)  # _h_range)

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
print(
    f"Overall gradient test for vectors {eps:e} apart yeilds an error of {err:e}"
)
_, _, eps, err = op.grad_test(x_native, delta, f1, g1)
print(
    f"Kalman gradient test for vectors {eps:e} apart yeilds an error of {err:e}"
)
_, _, eps, err = op.grad_test(x_native, delta, f2, g2)
print(
    f"TTW measurement gradient test for vectors {eps:e} apart yeilds an error of {err:e}"
)
_, _, eps, err = op.grad_test(x_native, delta, f3, g3)
print(
    f"ADCP measurement gradient test for vectors {eps:e} apart yeilds an error of {err:e}"
)
_, _, eps, err = op.grad_test(x_native, delta, f4, g4)
print(
    f"GPS measurement gradient test for vectors {eps:e} apart yeilds an error of {err:e}"
)
# _, _, eps, err = op.grad_test(x_native, delta, f5, g5)
# print(f"Range measurement gradient test for vectors {eps:e} apart yeilds an error of {err:e}")

_, _, eps, err = op.complex_step_test(x_native, delta, f, g)
print(
    f"Overall complex step test for gradient of norm {eps:e} yeilds an error of {err:e}"
)
_, _, eps, err = op.complex_step_test(x_native, delta, f1, g1)
print(
    f"Kalman complex step test for gradient of norm {eps:e} yeilds an error of {err:e}"
)
_, _, eps, err = op.complex_step_test(x_native, delta, f2, g2)
print(
    f"TTW measurement complex step test for gradient of norm {eps:e} yeilds an error of {err:e}"
)
_, _, eps, err = op.complex_step_test(x_native, delta, f3, g3)
print(
    f"ADCP measurement complex step test for gradient of norm {eps:e} yeilds an error of {err:e}"
)
_, _, eps, err = op.complex_step_test(x_native, delta, f4, g4)
print(
    f"GPS measurement complex step test for gradient of norm {eps:e} yeilds an error of {err:e}"
)
# _, _, eps, err = op.complex_step_test(x_native, delta, f5, g5)
# print(f"Range measurement complex step test for gradient of norm {eps:e} yeilds an error of {err:e}")

_, _, eps, err = op.hess_test(x_native, delta, g, h)
print(
    f"Overall hessian test for vectors {eps:e} apart yeilds an error of {err:e}"
)
_, _, eps, err = op.hess_test(x_native, delta, g1, h1)
print(
    f"Kalman hessian test for vectors {eps:e} apart yeilds an error of {err:e}"
)
_, _, eps, err = op.hess_test(x_native, delta, g2, h2)
print(
    f"TTW measurement hessian test for vectors {eps:e} apart yeilds an error of {err:e}"
)
_, _, eps, err = op.hess_test(x_native, delta, g3, h3)
print(
    f"ADCP measurement hessian test for vectors {eps:e} apart yeilds an error of {err:e}"
)
_, _, eps, err = op.hess_test(x_native, delta, g4, h4)
print(
    f"GPS measurement hessian test for vectors {eps:e} apart yeilds an error of {err:e}"
)
_, _, eps, err = op.hess_test(x_native, delta, g4, h4)
print(
    f"GPS measurement hessian test for vectors {eps:e} apart yeilds an error of {err:e}"
)
# _, _, eps, err = op.hess_test(x_native, delta, g5, h5)
# print(f"Range measurement hessian test for vectors {eps:e} apart yeilds an error of {err:e}")

_, _, eps, err = op.backsolve_test(x_native, prob)
print(
    f"Backsolve test for gradient of norm {eps:e} yeilds an error of {err:e}"
)
