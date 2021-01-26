# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

from adcp import simulation as sim
from adcp import optimization as op


class IntegrationSuite:
    def time_standard(self):
        rho_t = 1e-3
        rho_a = 1e-3
        rho_g = 1e-1

        sim_rho_v = 0
        sim_rho_c = 0
        sp = sim.SimParams(
            rho_t=rho_t,
            rho_a=rho_a,
            rho_g=rho_g,
            rho_v=sim_rho_v,
            rho_c=sim_rho_c,
            sigma_t=0.4,
            sigma_c=0.3,
            n_timepoints=2000,
            measure_points={"gps": "endpoints", "ttw": 0.5, "range": 0.05},
            vehicle_method="curved",
            curr_method="curved",
        )
        ddat, adat, x, curr_df, v_df = sim.simulate(sp, verbose=True)

        # %% No Range
        rho_c = 1e0
        rho_v = 1e0
        rho_v = 1e-8
        rho_c = 1e-8
        rho_g = rho_g
        rho_t = rho_t
        rho_a = rho_a
        rho_r = 0
        print(
            f"""Solution method covariances:
        vehicle process: {rho_v}
        current process:{rho_c}
        GPS measurement: {rho_g}
        TTW measurement: {rho_t}
        ADCP meawsurement: {rho_a}"""
        )

        prob = op.GliderProblem(
            ddat,
            adat,
            rho_v=rho_v,
            rho_c=rho_c,
            rho_g=rho_g,
            rho_t=rho_t,
            rho_a=rho_a,
            rho_r=rho_r,
            t_scale=1e3,
            conditioner="tanh",
            vehicle_vel="otg",
            current_order=2,
            vehicle_order=2,
        )

        # %%  Solve problem
        op.backsolve(prob)

        # Xs = prob.Xs
        # EV = prob.EV
        # NV = prob.NV
        # EC = prob.EC
        # NC = prob.NC

        # err = x_sol - x
        # path_error = (
        #     np.linalg.norm(Xs @ NV @ err) ** 2
        #     + np.linalg.norm(Xs @ EV @ err) ** 2
        # )
        # current_error = (
        #     np.linalg.norm(EC @ err) ** 2 + np.linalg.norm(NC @ err) ** 2
        # )

    def time_no_scaling(self):
        pass

    def time_no_conditioner(self):
        pass
