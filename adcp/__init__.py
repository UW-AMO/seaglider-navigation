from functools import cached_property
from dataclasses import dataclass
from typing import Optional as O  # noqa: E741

import pandas as pd

from adcp import dataprep as dp
from adcp import matbuilder as mb


@dataclass(frozen=True)
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
    def dr(self):
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
            self.m, self.config.current_order, self.config.vehicle_vel
        )

    @cached_property
    def CV(self):
        return mb.cv_select(
            self.m, self.config.current_order, self.config.vehicle_vel
        )

    @cached_property
    def CX(self):
        return mb.cx_select(
            self.m, self.config.current_order, self.config.vehicle_vel
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


@dataclass(frozen=True)
class GliderProblem:
    data: ProblemData
    config: ProblemConfig

    @cached_property
    def shape(self):
        return StateVectorShape(self.data, self.config)
