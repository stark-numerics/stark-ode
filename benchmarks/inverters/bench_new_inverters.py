from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import median
from time import perf_counter
from typing import Callable

from stark import Executor, ExecutorTolerance, Interval, IntegratorStepper
from stark.engines.accelerators import AcceleratorNumba
from stark.core.block import BlockBasis
from stark.methods.inverters.dense import InverterDense, InverterProviderDenseNative
from stark.methods.inverters.relaxation import InverterRelaxationJacobi, InverterRelaxationRichardson
from stark.methods.inverters.support import InverterBudget, InverterTolerance
from stark.diagnostics.monitor import MonitorInverter
from stark.methods.resolvents import ResolventNewton, ResolventPolicy, ResolventTolerance
from stark.methods.schemes import SchemeBackwardEuler

DIMENSION = 128
REPEATS = 5


@dataclass(slots=True)
class VectorState:
    values: list[float]

    def __init__(self, values: list[float] | None = None, *, dimension: int = DIMENSION) -> None:
        self.values = [0.0 for _ in range(dimension)] if values is None else [float(value) for value in values]


@dataclass(slots=True)
class VectorTranslation:
    values: list[float]

    def __init__(self, values: list[float] | None = None, *, dimension: int = DIMENSION) -> None:
        self.values = [0.0 for _ in range(dimension)] if values is None else [float(value) for value in values]

    def __call__(self, origin: VectorState, result: VectorState) -> None:
        result.values[:] = [
            origin_value + shift_value
            for origin_value, shift_value in zip(origin.values, self.values, strict=True)
        ]

    def norm(self) -> float:
        return sqrt(sum(value * value for value in self.values) / len(self.values))

    def __add__(self, other: VectorTranslation) -> VectorTranslation:
        return VectorTranslation([
            left + right
            for left, right in zip(self.values, other.values, strict=True)
        ])

    def __rmul__(self, scalar: float) -> VectorTranslation:
        return VectorTranslation([scalar * value for value in self.values])

    @staticmethod
    def scale(
        scalar: float,
        translation: VectorTranslation,
        output: VectorTranslation,
    ) -> VectorTranslation:
        output.values[:] = [scalar * value for value in translation.values]
        return output

    @staticmethod
    def combine2(
        left_scalar: float,
        left: VectorTranslation,
        right_scalar: float,
        right: VectorTranslation,
        output: VectorTranslation,
    ) -> VectorTranslation:
        output.values[:] = [
            left_scalar * left_value + right_scalar * right_value
            for left_value, right_value in zip(left.values, right.values, strict=True)
        ]
        return output

    @staticmethod
    def combine3(
        first_scalar: float,
        first: VectorTranslation,
        second_scalar: float,
        second: VectorTranslation,
        third_scalar: float,
        third: VectorTranslation,
        output: VectorTranslation,
    ) -> VectorTranslation:
        output.values[:] = [
            first_scalar * first_value + second_scalar * second_value + third_scalar * third_value
            for first_value, second_value, third_value in zip(
                first.values,
                second.values,
                third.values,
                strict=True,
            )
        ]
        return output

    linear_combine = [scale, combine2, combine3]


@dataclass(frozen=True, slots=True)
class VectorBasis:
    dimension: int

    def vector(self, index: int, output: VectorTranslation) -> VectorTranslation:
        output.values[:] = [0.0 for _ in range(self.dimension)]
        output.values[index] = 1.0
        return output

    def coordinate(self, index: int, translation: VectorTranslation) -> float:
        return translation.values[index]

    def coordinates(self, translation: VectorTranslation, output: list[float]) -> list[float]:
        output[:] = translation.values[:]
        return output

    def synthesize(self, coordinates: list[float], output: VectorTranslation) -> VectorTranslation:
        output.values[:] = list(coordinates)
        return output


@dataclass(slots=True)
class VectorAllocator:
    dimension: int

    def allocate_state(self) -> VectorState:
        return VectorState(dimension=self.dimension)

    def copy_state(self, source: VectorState, output: VectorState) -> None:
        output.values[:] = source.values[:]

    def allocate_translation(self) -> VectorTranslation:
        return VectorTranslation(dimension=self.dimension)


@dataclass(frozen=True, slots=True)
class VectorDerivative:
    rates: tuple[float, ...]

    def __call__(self, interval: Interval, state: VectorState, output: VectorTranslation) -> None:
        del interval
        output.values[:] = [
            -rate * state_value
            for rate, state_value in zip(self.rates, state.values, strict=True)
        ]


@dataclass(frozen=True, slots=True)
class VectorLinearizer:
    rates: tuple[float, ...]

    def __call__(self, interval: Interval, state: VectorState, operator) -> None:
        del interval, state

        def apply(translation: VectorTranslation, output: VectorTranslation) -> None:
            output.values[:] = [
                -rate * translation_value
                for rate, translation_value in zip(self.rates, translation.values, strict=True)
            ]

        operator.apply = apply


@dataclass(frozen=True, slots=True)
class DiagonalEntryInverse:
    basis: VectorBasis

    def __call__(self, operator, source: VectorTranslation, target: VectorTranslation) -> None:
        basis_vector = VectorTranslation(dimension=self.basis.dimension)
        image = VectorTranslation(dimension=self.basis.dimension)
        target.values[:] = [0.0 for _ in range(self.basis.dimension)]

        for index in range(self.basis.dimension):
            basis_vector = self.basis.vector(index, basis_vector)
            result = operator(basis_vector, image)
            if result is not None:
                image = result
            diagonal = self.basis.coordinate(index, image)
            if diagonal == 0.0:
                raise ZeroDivisionError("Jacobi benchmark diagonal entry is singular.")
            target.values[index] = source.values[index] / diagonal


def make_initial_state(dimension: int) -> VectorState:
    return VectorState(
        [(-1.0 if index % 2 else 1.0) / (1.0 + index) for index in range(dimension)]
    )


def make_dense_inverter(basis: VectorBasis, monitor: MonitorInverter) -> InverterDense[VectorTranslation]:
    return InverterDense(
        basis=BlockBasis([basis]),
        provider=InverterProviderDenseNative(accelerator=AcceleratorNumba()),
        monitor=monitor,
    )


def make_jacobi_inverter(basis: VectorBasis, monitor: MonitorInverter) -> InverterRelaxationJacobi[VectorTranslation]:
    return InverterRelaxationJacobi(
        diagonal_inverse=DiagonalEntryInverse(basis),
        damping=1.0,
        tolerance=InverterTolerance(atol=1.0e-11, rtol=0.0),
        budget=InverterBudget(maximum_steps=8),
        monitor=monitor,
    )


def make_richardson_inverter(monitor: MonitorInverter) -> InverterRelaxationRichardson[VectorTranslation]:
    return InverterRelaxationRichardson(
        damping=0.55,
        tolerance=InverterTolerance(atol=1.0e-9, rtol=0.0),
        budget=InverterBudget(maximum_steps=200),
        monitor=monitor,
    )


def prepare_solver(
    label: str,
    make_inverter: Callable[[VectorBasis, MonitorInverter], object],
    *,
    dimension: int = DIMENSION,
):
    rates = tuple(1.0 + index / dimension for index in range(dimension))
    allocator = VectorAllocator(dimension)
    derivative = VectorDerivative(rates)
    basis = VectorBasis(dimension)
    monitor = MonitorInverter()
    inverter = make_inverter(basis, monitor)
    resolvent = ResolventNewton(
        allocator,
        linearizer=VectorLinearizer(rates),
        inverter=inverter,
        ExecutorTolerance=ResolventTolerance(atol=1.0e-9, rtol=1.0e-9),
        policy=ResolventPolicy(max_iterations=12),
    )
    scheme = SchemeBackwardEuler(
        derivative,
        allocator,
        resolvent=resolvent,
    )
    stepper = IntegratorStepper(scheme, Executor(tolerance=ExecutorTolerance()))

    def solve_once():
        interval = Interval(present=0.0, step=0.01, stop=0.01)
        state = make_initial_state(dimension)
        scheme.delta[0].values[:] = [0.0 for _ in range(dimension)]
        stepper(interval, state)
        summary = monitor.summary()
        monitor.clear()
        return state, summary

    return label, solve_once


def time_solver(label: str, solve_once):
    setup_start = perf_counter()
    # Construction has already happened by the time this function is called.
    setup = perf_counter() - setup_start

    warmup_start = perf_counter()
    _state, warmup_summary = solve_once()
    warmup = perf_counter() - warmup_start

    timings = []
    summary = warmup_summary
    for _ in range(REPEATS):
        started = perf_counter()
        _state, summary = solve_once()
        timings.append(perf_counter() - started)

    return {
        "solver": label,
        "setup": setup,
        "warmup": warmup,
        "median": median(timings),
        "min": min(timings),
        "solves": summary.solve_count,
        "failures": summary.failure_count,
        "iter_median": summary.iteration_median,
        "iter_max": summary.iteration_max,
    }


def render_table(headers, rows):
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    lines = [
        " | ".join(header.ljust(width) for header, width in zip(headers, widths, strict=True)),
        "-+-".join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append(" | ".join(value.ljust(width) for value, width in zip(row, widths, strict=True)))
    return "\n".join(lines)


def main() -> None:
    rows = []
    for label, factory in (
        ("Dense Native", make_dense_inverter),
        ("Jacobi Diagonal", make_jacobi_inverter),
        ("Richardson", lambda _basis, monitor: make_richardson_inverter(monitor)),
    ):
        label, solve_once = prepare_solver(label, factory)
        rows.append(time_solver(label, solve_once))

    print("New Inverter Benchmark")
    print(f"  dimension: {DIMENSION}")
    print(f"  repeats: {REPEATS}")
    print("  scheme: Backward Euler")
    print("  resolvent: Newton")
    print()
    print(
        render_table(
            ("solver", "warmup", "median", "min", "solves", "failures", "iter med", "iter max"),
            [
                (
                    row["solver"],
                    f"{row['warmup']:.6f}s",
                    f"{row['median']:.6f}s",
                    f"{row['min']:.6f}s",
                    str(row["solves"]),
                    str(row["failures"]),
                    "-" if row["iter_median"] is None else f"{row['iter_median']:.1f}",
                    "-" if row["iter_max"] is None else str(row["iter_max"]),
                )
                for row in rows
            ],
        )
    )


if __name__ == "__main__":
    main()
