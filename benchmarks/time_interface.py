from __future__ import annotations

from stark import Executor, Integrator, Interval, Marcher, Safety, Tolerance
from stark.carriers import CarrierNumpy
from stark.interface import StarkDerivative, StarkIVP, StarkVector
from stark.interface.derivative import StarkDerivative as InterfaceDerivative
from stark.interface.vector import StarkVectorWorkbench
from stark.schemes.explicit_adaptive import SchemeCashKarp

from benchmarks.common import (
    FPUT_SIZES,
    FPUTDerivative,
    FPUTMatrixDerivative,
    FPUTMatrixReturnDerivative,
    FPUTParameters,
    FPUTVectorDerivative,
    FPUTVectorReturnDerivative,
    FPUTWorkbench,
    initial_fput_matrix,
    initial_fput_state,
    initial_fput_vector,
)


STOP = 20.0
INITIAL_STEP = 1.0e-3
RTOL = 1.0e-6
ATOL = 1.0e-8


def parameters(chain_size: int) -> FPUTParameters:
    return FPUTParameters(
        chain_size=chain_size,
        t1=STOP,
        initial_step=INITIAL_STEP,
    )


def executor() -> Executor:
    return Executor(
        tolerance=Tolerance(atol=ATOL, rtol=RTOL),
        safety=Safety.fast(),
    )


class CoreFPUTCase:
    __slots__ = ("integrator", "interval", "marcher", "state")

    def __init__(self, problem: FPUTParameters) -> None:
        derivative = FPUTDerivative(problem)
        workbench = FPUTWorkbench(problem)
        run_executor = executor()
        scheme = SchemeCashKarp(derivative, workbench)
        self.marcher = Marcher(scheme, run_executor)
        self.integrator = Integrator(executor=run_executor)
        self.interval = Interval(problem.t0, problem.initial_step, problem.t1)
        self.state = initial_fput_state(problem)

    def solve_once(self) -> None:
        interval = self.interval.copy()
        state = self.state.copy()
        for _interval, _state in self.integrator.live(self.marcher, interval, state):
            pass


class VectorCoreFPUTCase:
    __slots__ = ("carrier", "initial_value", "integrator", "interval", "marcher")

    def __init__(self, problem: FPUTParameters, initial: object, derivative: object) -> None:
        carrier = CarrierNumpy().bind(initial)
        workbench = StarkVectorWorkbench(carrier)
        bound_derivative = InterfaceDerivative.in_place(derivative).bind(carrier)
        run_executor = executor()
        scheme = SchemeCashKarp(bound_derivative, workbench)
        self.marcher = Marcher(scheme, run_executor)
        self.integrator = Integrator(executor=run_executor)
        self.interval = Interval(problem.t0, problem.initial_step, problem.t1)
        self.carrier = carrier
        self.initial_value = initial

    def solve_once(self) -> None:
        interval = self.interval.copy()
        state = StarkVector(self.initial_value.copy(), self.carrier)
        for _interval, _state in self.integrator.live(self.marcher, interval, state):
            pass


class InterfaceFPUTCase:
    __slots__ = ("build", "carrier", "initial_value", "interval")

    def __init__(self, problem: FPUTParameters, initial: object, derivative: object) -> None:
        ivp = StarkIVP(
            derivative=derivative,
            initial=initial,
            interval=Interval(problem.t0, problem.initial_step, problem.t1),
            scheme=SchemeCashKarp,
            executor=executor(),
        )
        build = ivp.build()
        self.build = build
        self.carrier = build.initial.carrier
        self.initial_value = build.initial.value
        self.interval = build.interval

    def solve_once(self) -> None:
        interval = self.interval.copy()
        state = StarkVector(self.initial_value.copy(), self.carrier)
        for _interval, _state in self.build.integrator.live(self.build.marcher, interval, state):
            pass


class TimeInterface:
    params = (FPUT_SIZES,)
    param_names = ("chain_size",)

    def setup(self, chain_size: int) -> None:
        problem = parameters(chain_size)
        self.core = CoreFPUTCase(problem)
        self.vector_core = VectorCoreFPUTCase(
            problem,
            initial_fput_vector(problem),
            FPUTVectorDerivative(problem),
        )
        self.matrix_core = VectorCoreFPUTCase(
            problem,
            initial_fput_matrix(problem),
            FPUTMatrixDerivative(problem),
        )
        self.interface_vector_in_place = InterfaceFPUTCase(
            problem,
            initial_fput_vector(problem),
            StarkDerivative.in_place(FPUTVectorDerivative(problem)),
        )
        self.interface_vector_return = InterfaceFPUTCase(
            problem,
            initial_fput_vector(problem),
            FPUTVectorReturnDerivative(problem),
        )
        self.interface_matrix_in_place = InterfaceFPUTCase(
            problem,
            initial_fput_matrix(problem),
            StarkDerivative.in_place(FPUTMatrixDerivative(problem)),
        )
        self.interface_matrix_return = InterfaceFPUTCase(
            problem,
            initial_fput_matrix(problem),
            FPUTMatrixReturnDerivative(problem),
        )

    def time_core_cash_karp(self, chain_size: int) -> None:
        del chain_size
        self.core.solve_once()

    def time_vector_core_cash_karp(self, chain_size: int) -> None:
        del chain_size
        self.vector_core.solve_once()

    def time_matrix_core_cash_karp(self, chain_size: int) -> None:
        del chain_size
        self.matrix_core.solve_once()

    def time_interface_cash_karp_vector_in_place(self, chain_size: int) -> None:
        del chain_size
        self.interface_vector_in_place.solve_once()

    def time_interface_cash_karp_vector_return(self, chain_size: int) -> None:
        del chain_size
        self.interface_vector_return.solve_once()

    def time_interface_cash_karp_matrix_in_place(self, chain_size: int) -> None:
        del chain_size
        self.interface_matrix_in_place.solve_once()

    def time_interface_cash_karp_matrix_return(self, chain_size: int) -> None:
        del chain_size
        self.interface_matrix_return.solve_once()


class TimeInterfaceSetup:
    params = (FPUT_SIZES,)
    param_names = ("chain_size",)

    def setup(self, chain_size: int) -> None:
        self.problem = parameters(chain_size)

    def time_core_cash_karp_setup(self, chain_size: int) -> None:
        del chain_size
        CoreFPUTCase(self.problem)

    def time_vector_core_cash_karp_setup(self, chain_size: int) -> None:
        del chain_size
        VectorCoreFPUTCase(
            self.problem,
            initial_fput_vector(self.problem),
            FPUTVectorDerivative(self.problem),
        )

    def time_matrix_core_cash_karp_setup(self, chain_size: int) -> None:
        del chain_size
        VectorCoreFPUTCase(
            self.problem,
            initial_fput_matrix(self.problem),
            FPUTMatrixDerivative(self.problem),
        )

    def time_interface_cash_karp_in_place_setup(self, chain_size: int) -> None:
        del chain_size
        InterfaceFPUTCase(
            self.problem,
            initial_fput_vector(self.problem),
            StarkDerivative.in_place(FPUTVectorDerivative(self.problem)),
        )

    def time_interface_cash_karp_return_setup(self, chain_size: int) -> None:
        del chain_size
        InterfaceFPUTCase(
            self.problem,
            initial_fput_vector(self.problem),
            FPUTVectorReturnDerivative(self.problem),
        )

    def time_interface_cash_karp_matrix_in_place_setup(self, chain_size: int) -> None:
        del chain_size
        InterfaceFPUTCase(
            self.problem,
            initial_fput_matrix(self.problem),
            StarkDerivative.in_place(FPUTMatrixDerivative(self.problem)),
        )

    def time_interface_cash_karp_matrix_return_setup(self, chain_size: int) -> None:
        del chain_size
        InterfaceFPUTCase(
            self.problem,
            initial_fput_matrix(self.problem),
            FPUTMatrixReturnDerivative(self.problem),
        )
