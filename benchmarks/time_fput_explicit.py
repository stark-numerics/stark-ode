from __future__ import annotations

from stark import Executor, Integrator, Interval, Marcher, Safety, Tolerance
from stark.schemes.explicit_adaptive import SchemeCashKarp, SchemeDormandPrince
from stark.schemes.explicit_fixed import SchemeEuler, SchemeRK4

from benchmarks.common import (
    FPUT_ALGEBRAIST,
    FPUT_SIZES,
    FPUTDerivative,
    FPUTParameters,
    FPUTWorkbench,
    initial_fput_state,
)


FixedScheme = type[SchemeEuler] | type[SchemeRK4]
AdaptiveScheme = type[SchemeCashKarp] | type[SchemeDormandPrince]


FIXED_STOP = 0.2
ADAPTIVE_STOP = 20.0
INITIAL_STEP = 1.0e-3
RTOL = 1.0e-6
ATOL = 1.0e-8


def fixed_parameters(chain_size: int) -> FPUTParameters:
    return FPUTParameters(
        chain_size=chain_size,
        t1=FIXED_STOP,
        initial_step=INITIAL_STEP,
    )


def adaptive_parameters(chain_size: int) -> FPUTParameters:
    return FPUTParameters(
        chain_size=chain_size,
        t1=ADAPTIVE_STOP,
        initial_step=INITIAL_STEP,
    )


def executor() -> Executor:
    return Executor(
        tolerance=Tolerance(atol=ATOL, rtol=RTOL),
        safety=Safety.fast(),
    )


class FPUTExplicitCase:
    __slots__ = ("integrator", "interval", "marcher", "state")

    def __init__(
        self,
        scheme_type: FixedScheme | AdaptiveScheme,
        parameters: FPUTParameters,
        *,
        algebraist: object | None = None,
    ) -> None:
        derivative = FPUTDerivative(parameters)
        workbench = FPUTWorkbench(parameters)
        scheme = scheme_type(derivative, workbench, algebraist=algebraist)
        run_executor = executor()
        self.marcher = Marcher(scheme, run_executor)
        self.integrator = Integrator(executor=run_executor)
        self.interval = Interval(parameters.t0, parameters.initial_step, parameters.t1)
        self.state = initial_fput_state(parameters)

    def solve_once(self) -> None:
        interval = self.interval.copy()
        state = self.state.copy()
        for _interval, _state in self.integrator.live(self.marcher, interval, state):
            pass


class TimeFPUTExplicit:
    params = (FPUT_SIZES,)
    param_names = ("chain_size",)

    def setup(self, chain_size: int) -> None:
        fixed = fixed_parameters(chain_size)
        adaptive = adaptive_parameters(chain_size)
        self.euler = FPUTExplicitCase(SchemeEuler, fixed)
        self.euler_algebraist = FPUTExplicitCase(SchemeEuler, fixed, algebraist=FPUT_ALGEBRAIST)
        self.rk4 = FPUTExplicitCase(SchemeRK4, fixed)
        self.rk4_algebraist = FPUTExplicitCase(SchemeRK4, fixed, algebraist=FPUT_ALGEBRAIST)
        self.cash_karp = FPUTExplicitCase(SchemeCashKarp, adaptive)
        self.cash_karp_algebraist = FPUTExplicitCase(SchemeCashKarp, adaptive, algebraist=FPUT_ALGEBRAIST)
        self.dormand_prince = FPUTExplicitCase(SchemeDormandPrince, adaptive)
        self.dormand_prince_algebraist = FPUTExplicitCase(
            SchemeDormandPrince,
            adaptive,
            algebraist=FPUT_ALGEBRAIST,
        )

    def time_euler(self, chain_size: int) -> None:
        del chain_size
        self.euler.solve_once()

    def time_euler_algebraist(self, chain_size: int) -> None:
        del chain_size
        self.euler_algebraist.solve_once()

    def time_rk4(self, chain_size: int) -> None:
        del chain_size
        self.rk4.solve_once()

    def time_rk4_algebraist(self, chain_size: int) -> None:
        del chain_size
        self.rk4_algebraist.solve_once()

    def time_cash_karp(self, chain_size: int) -> None:
        del chain_size
        self.cash_karp.solve_once()

    def time_cash_karp_algebraist(self, chain_size: int) -> None:
        del chain_size
        self.cash_karp_algebraist.solve_once()

    def time_dormand_prince(self, chain_size: int) -> None:
        del chain_size
        self.dormand_prince.solve_once()

    def time_dormand_prince_algebraist(self, chain_size: int) -> None:
        del chain_size
        self.dormand_prince_algebraist.solve_once()


class TimeFPUTExplicitSetup:
    params = (FPUT_SIZES,)
    param_names = ("chain_size",)

    def setup(self, chain_size: int) -> None:
        self.fixed_parameters = fixed_parameters(chain_size)
        self.adaptive_parameters = adaptive_parameters(chain_size)

    def time_euler_setup(self, chain_size: int) -> None:
        del chain_size
        FPUTExplicitCase(SchemeEuler, self.fixed_parameters)

    def time_euler_algebraist_setup(self, chain_size: int) -> None:
        del chain_size
        FPUTExplicitCase(SchemeEuler, self.fixed_parameters, algebraist=FPUT_ALGEBRAIST)

    def time_rk4_setup(self, chain_size: int) -> None:
        del chain_size
        FPUTExplicitCase(SchemeRK4, self.fixed_parameters)

    def time_rk4_algebraist_setup(self, chain_size: int) -> None:
        del chain_size
        FPUTExplicitCase(SchemeRK4, self.fixed_parameters, algebraist=FPUT_ALGEBRAIST)

    def time_cash_karp_setup(self, chain_size: int) -> None:
        del chain_size
        FPUTExplicitCase(SchemeCashKarp, self.adaptive_parameters)

    def time_cash_karp_algebraist_setup(self, chain_size: int) -> None:
        del chain_size
        FPUTExplicitCase(SchemeCashKarp, self.adaptive_parameters, algebraist=FPUT_ALGEBRAIST)

    def time_dormand_prince_setup(self, chain_size: int) -> None:
        del chain_size
        FPUTExplicitCase(SchemeDormandPrince, self.adaptive_parameters)

    def time_dormand_prince_algebraist_setup(self, chain_size: int) -> None:
        del chain_size
        FPUTExplicitCase(
            SchemeDormandPrince,
            self.adaptive_parameters,
            algebraist=FPUT_ALGEBRAIST,
        )
