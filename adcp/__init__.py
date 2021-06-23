from functools import cached_property
from dataclasses import dataclass
from typing import Optional as O  # noqa: E741

import pandas as pd

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
    def vehicle_depth_mat(self):
        return mb.vehicle_select(
            self.data.times,
            self.data.depths,
            self.data.ddat,
        )

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

    def vehicle_Q(self):
        if self.config.vehicle_vel != "otg-cov":
            return mb.vehicle_Q(
                self.data.times,
                1,
                self.config.vehicle_order,
                self.config.conditioner,
                self.config.t_scale,
            )
        pass

    def vehicle_Qinv(self):
        if self.config.vehicle_vel != "otg-cov":
            return mb.vehicle_Qinv(
                self.data.times,
                1,
                self.config.vehicle_order,
                self.config.conditioner,
                self.config.t_scale,
            )
        pass

    def vehicle_G(self):
        partial_G = mb.vehicle_G(
            self.data.times,
            self.config.vehicle_order,
            self.config.conditioner,
            self.config.t_scale,
        )
        if self.config.vehicle_vel != "otg-cov":
            return partial_G


@dataclass
class Weights:
    rho_v: float
    rho_c: float
    rho_t: float
    rho_a: float
    rho_g: float
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
    def Atb(self):
        _, Atb = self.__AtA_Atb
        return Atb

    @cached_property
    def kalman_mat(self):
        return op.gen_kalman_mat(self, root=False)

    @cached_property
    def kalman_root(self):
        return op.gen_kalman_mat(self, root=True)

    @cached_property
    def f(self):
        return op.f(self)

    @cached_property
    def g(self):
        return op.g(self)

    @cached_property
    def h(self):
        return op.h(self)

    def __getattribute__(self, name: str):
        print(f"Looking for {name} in {self}")
        return super().__getattribute__(name)

    def __getattr__(self, name):
        print(f"__getattribute__({self}, {name}) failed")
        # try:
        for k, v in self.__dict__.items():
            print(f"Looking in {k} for {name}")
            try:
                val = getattr(v, name)
                print(f"found {name} in {v}")
                return val
            except AttributeError as err:
                print(err)
        raise AttributeError(
            f"Neither {self} nor it's fields have a {name} attribute"
        )
