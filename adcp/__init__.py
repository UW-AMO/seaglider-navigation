import traceback
from functools import cached_property
from dataclasses import dataclass
from typing import Optional as O  # noqa: E741
import warnings

import pandas as pd
import numpy as np

from adcp import dataprep as dp
from adcp import matbuilder as mb
from adcp import optimization as op


@dataclass(frozen=True, repr=False)
class ProblemData:
    ddat: dict
    adat: dict

    @cached_property
    def depths(self):
        return dp.depthpoints(self.adat, self.ddat)

    @cached_property
    def times(self):
        return dp.timepoints(self.adat, self.ddat)

    @cached_property
    def depth_rates(self):
        return dp.depth_rates(self.times, self.depths, self.ddat)

    @cached_property
    def interpolator(self) -> pd.DataFrame:
        return dp._depth_interpolator(self.times, self.ddat)

    @cached_property
    def idx_vehicle(self) -> np.array:
        vehicle_depths = dp._depth_interpolator(self.times, self.ddat)["depth"]
        idxdepth = np.array(
            [np.argwhere(self.depths == d) for d in vehicle_depths]
        ).flatten()
        return idxdepth


@dataclass(frozen=True)
class ProblemConfig:
    t_scale: int = 1
    conditioner: O[str] = "tanh"
    vehicle_vel: str = "otg"
    current_order: int = 2
    vehicle_order: int = 2


@dataclass(frozen=True)
class StateVectorShape:
    data: ProblemData
    config: ProblemConfig

    m = cached_property(lambda self: len(self.data.times))
    n = cached_property(lambda self: len(self.data.depths))

    @cached_property
    def __uv_sel_mats(self):
        return mb.uv_select(
            self.data.times,
            self.data.depths,
            self.data.ddat,
            self.data.adat,
            self.config.vehicle_vel,
        )

    @cached_property
    def A_ttw(self):
        A, _ = self.__uv_sel_mats
        return A

    @cached_property
    def B_ttw(self):
        _, B = self.__uv_sel_mats
        return B

    @cached_property
    def __adcp_sel_mats(self):
        return mb.adcp_select(
            self.data.times,
            self.data.depths,
            self.data.ddat,
            self.data.adat,
            self.config.vehicle_vel,
        )

    @cached_property
    def A_adcp(self):
        A, _ = self.__adcp_sel_mats
        return A

    @cached_property
    def B_adcp(self):
        _, B = self.__adcp_sel_mats
        return B

    @cached_property
    def __gps_sel_mats(self):
        return mb.gps_select(
            self.data.times,
            self.data.depths,
            self.data.ddat,
            self.data.adat,
            self.config.vehicle_vel,
        )

    @cached_property
    def A_gps(self):
        A, _ = self.__gps_sel_mats
        return A

    @cached_property
    def B_gps(self):
        _, B = self.__gps_sel_mats
        return B

    @cached_property
    def N(self):
        return mb.n_select(
            self.m,
            self.n,
            self.config.vehicle_order,
            self.config.current_order,
            self.config.vehicle_vel,
        )

    @cached_property
    def E(self):
        return mb.e_select(
            self.m,
            self.n,
            self.config.vehicle_order,
            self.config.current_order,
            self.config.vehicle_vel,
        )

    @cached_property
    def EV(self):
        return mb.ev_select(
            self.m,
            self.n,
            self.config.vehicle_order,
            self.config.current_order,
            self.config.vehicle_vel,
        )

    @cached_property
    def NV(self):
        return mb.nv_select(
            self.m,
            self.n,
            self.config.vehicle_order,
            self.config.current_order,
            self.config.vehicle_vel,
        )

    @cached_property
    def EC(self):
        return mb.ec_select(
            self.m,
            self.n,
            self.config.vehicle_order,
            self.config.current_order,
            self.config.vehicle_vel,
        )

    @cached_property
    def NC(self):
        return mb.nc_select(
            self.m,
            self.n,
            self.config.vehicle_order,
            self.config.current_order,
            self.config.vehicle_vel,
        )

    @cached_property
    def As(self):
        return mb.a_select(self.m, self.config.vehicle_order)

    @cached_property
    def Vs(self):
        return mb.v_select(self.m, self.config.vehicle_order)

    @cached_property
    def Xs(self):
        return mb.x_select(self.m, self.config.vehicle_order)

    @cached_property
    def CA(self):
        return mb.ca_select(
            self.n, self.config.current_order, self.config.vehicle_vel
        )

    @cached_property
    def CV(self):
        return mb.cv_select(
            self.n, self.config.current_order, self.config.vehicle_vel
        )

    @cached_property
    def CX(self):
        return mb.cx_select(
            self.n, self.config.current_order, self.config.vehicle_vel
        )

    @cached_property
    def Qv(self):
        return mb.vehicle_Q(
            self.data.times,
            1,
            self.config.vehicle_order,
            self.config.conditioner,
            self.config.t_scale,
        )

    @cached_property
    def Qvinv(self):
        return mb.vehicle_Qinv(
            self.data.times,
            1,
            self.config.vehicle_order,
            self.config.conditioner,
            self.config.t_scale,
        )

    @cached_property
    def Gv(self):
        return mb.vehicle_G(
            self.data.times,
            self.config.vehicle_order,
            self.config.conditioner,
            self.config.t_scale,
        )

    @cached_property
    def Gvc(self):
        return mb.vehicle_G_given_C(
            self.data.times,
            self.config.vehicle_order,
            self.config.t_scale,
            self.data.depths,
            self.data.idx_vehicle,
            self.config.vehicle_vel,
            self.config.current_order,
        )

    @cached_property
    def Qc(self):
        return mb.depth_Q(
            self.data.depths,
            1,
            self.config.current_order,
            self.data.depth_rates,
            self.config.conditioner,
            self.config.t_scale,
            self.config.vehicle_vel,
        )

    @cached_property
    def Qcinv(self):
        return mb.depth_Qinv(
            self.data.depths,
            1,
            self.config.current_order,
            self.data.depth_rates,
            self.config.conditioner,
            self.config.t_scale,
            self.config.vehicle_vel,
        )

    @cached_property
    def Gc(self):
        return mb.depth_G(
            self.data.depths,
            self.config.vehicle_order,
            self.data.depth_rates,
            self.config.conditioner,
            self.config.vehicle_vel,
        )


@dataclass
class Weights:
    rho_v: float = 1
    rho_c: float = 1
    rho_t: float = 1
    rho_a: float = 1
    rho_g: float = 1
    rho_r: float = 0


@dataclass(repr=False)
class GliderProblem:
    data: ProblemData
    config: ProblemConfig
    weights: Weights

    def __init__(self, data, config, weights):
        self.data = data
        self.config = config
        self.weights = weights
        self.shape = StateVectorShape(data, config)

    # @cached_property
    # def shape(self):
    #     return StateVectorShape(self.data, self.config)

    @cached_property
    def __A_b(self):
        return op.basic_A_b(self)

    @cached_property
    def A(self):
        A, _ = self.__A_b
        return A

    @cached_property
    def b(self):
        _, b = self.__A_b
        return b

    @cached_property
    def __AtA_Atb(self):
        return op.solve_mats(self)

    @cached_property
    def AtA(self):
        AtA, _ = self.__AtA_Atb
        return AtA

    @cached_property
    def __AtAinv(self):
        return op.solution_variance_estimator(
            self.AtA,
            self.shape.m,
            self.shape.n,
            self.config.vehicle_order,
            self.config.current_order,
            self.config.vehicle_vel,
        )

    @cached_property
    def AtAinv(self):
        AtAinv, _, _ = self.__AtAinv
        return AtAinv

    @cached_property
    def AtAinv_v_points(self):
        _, rows, _ = self.__AtAinv
        return rows

    @cached_property
    def AtAinv_c_points(self):
        _, _, rows = self.__AtAinv
        return rows

    @cached_property
    def Atb(self):
        _, Atb = self.__AtA_Atb
        return Atb

    @cached_property
    def kalman_mat(self):
        return op.gen_kalman_mat(
            self.data, self.config, self.shape, self.weights, root=False
        )

    @cached_property
    def kalman_root(self):
        return op.gen_kalman_mat(
            self.data, self.config, self.shape, self.weights, root=True
        )

    @cached_property
    def f(self):
        return op.f(self)

    @cached_property
    def g(self):
        return op.g(self)

    @cached_property
    def h(self):
        return op.h(self)

    def __getattr__(self, name):
        for v in [self.data, self.config, self.weights, self.shape]:
            try:
                val = getattr(v, name)
                warnings.warn(
                    f"Found {name} in {v.__class__} when accessing"
                    f" GliderProblem.{name}.  Call {v.__class__}.name in the"
                    " future"
                )
                traceback.print_stack(limit=5)
                return val
            except AttributeError:
                pass
        raise AttributeError(
            f"Neither {self} nor it's fields have a {name} attribute"
        )

    def legacy_size_prob(self):
        new_config = ProblemConfig(
            vehicle_order=2, current_order=2, vehicle_vel="otg"
        )
        new_problem = GliderProblem(self.data, new_config, self.weights)
        return new_problem


def create_legacy_shape(shape: StateVectorShape) -> StateVectorShape:
    new_config = ProblemConfig(
        vehicle_order=2, current_order=2, vehicle_vel="otg"
    )
    return StateVectorShape(shape.data, new_config)


def create_legacy_shape_problem(prob: GliderProblem) -> GliderProblem:
    new_config = ProblemConfig(
        vehicle_order=2, current_order=2, vehicle_vel="otg"
    )
    return GliderProblem(prob.data, new_config, prob.weights)
