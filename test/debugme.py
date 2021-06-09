from adcp import viz, matbuilder as mb
from adcp.tests.test_integration import standard_sim

results = standard_sim(
    t_scale=1,
    current_order=3,
    vehicle_order=3,
    rho_v=1e-7,
    rho_c=1e-4,
)

prob = results["prob"]
legacy = mb.legacy_select(
    prob.m,
    prob.n,
    prob.vehicle_order,
    prob.current_order,
    prob.vehicle_vel,
    prob=prob,
)
x_plot = legacy @ results["x_sol"]
x_true = results["x_true"]
legacy_size_prob = results["prob"].legacy_size_prob()

viz.plot_bundle(
    x_plot,
    legacy_size_prob.adat,
    legacy_size_prob.ddat,
    results["prob"].times,
    results["prob"].depths,
    x_true,
)
