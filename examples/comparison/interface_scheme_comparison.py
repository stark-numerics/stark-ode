from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np
from scipy.integrate import solve_ivp

from stark import Interval
from stark.interface import StarkIVP
from stark.schemes import (
    SchemeCashKarp,
    SchemeDormandPrince,
    SchemeFehlberg45,
    SchemeTsitouras5,
)


SIZES = [3, 1_000, 10_000, 100_000]
RUNS_BY_SIZE = {
    3: 100,
    1_000: 50,
    10_000: 20,
    100_000: 5,
}

START = 0.0
STEP = 0.1
STOP = 10.0
RTOL = 1.0e-6
ATOL = 1.0e-6

STARK_SCHEMES = [
    ("DormandPrince", SchemeDormandPrince),
    ("CashKarp", SchemeCashKarp),
    ("Fehlberg45", SchemeFehlberg45),
    ("Tsitouras5", SchemeTsitouras5),
]


@dataclass
class RunResult:
    final_value: np.ndarray
    rhs_calls: int


@dataclass
class TimedResult:
    label: str
    size: int
    runs: int
    total_seconds: float
    per_run_seconds: float
    rhs_calls_last: int
    rhs_calls_mean: float
    time_per_rhs_seconds: float
    max_abs_error: float | None = None


class CountedReturnRHS:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, t: float, y: Any) -> Any:
        self.calls += 1
        return -0.5 * y


def make_initial(size: int) -> np.ndarray:
    return np.linspace(1.0, 2.0, size)


def run_stark_scheme(size: int, scheme_class: type) -> RunResult:
    rhs = CountedReturnRHS()
    initial = make_initial(size)

    ivp = StarkIVP(
        derivative=rhs,
        initial=initial,
        interval=Interval(present=START, step=STEP, stop=STOP),
        scheme=scheme_class,
    )

    build = ivp.build()
    final_state = None

    for _interval, state in build.integrator.live(
        build.marcher,
        build.interval,
        build.initial,
    ):
        final_state = state

    if final_state is None:
        raise RuntimeError(f"{scheme_class.__name__} produced no output.")

    return RunResult(
        final_value=np.asarray(final_state.value).copy(),
        rhs_calls=rhs.calls,
    )


def run_scipy_rk45(size: int) -> RunResult:
    rhs = CountedReturnRHS()
    initial = make_initial(size)

    result = solve_ivp(
        rhs,
        (START, STOP),
        initial,
        method="RK45",
        first_step=STEP,
        rtol=RTOL,
        atol=ATOL,
    )

    return RunResult(
        final_value=result.y[:, -1],
        rhs_calls=result.nfev,
    )


def time_runs(
    label: str,
    size: int,
    runs: int,
    function: Callable[[], RunResult],
) -> tuple[TimedResult, RunResult]:
    # Warm-up.
    function()

    started = perf_counter()

    result = None
    total_rhs_calls = 0

    for _ in range(runs):
        result = function()
        total_rhs_calls += result.rhs_calls

    elapsed = perf_counter() - started

    if result is None:
        raise RuntimeError(f"{label} produced no result.")

    return (
        TimedResult(
            label=label,
            size=size,
            runs=runs,
            total_seconds=elapsed,
            per_run_seconds=elapsed / runs,
            rhs_calls_last=result.rhs_calls,
            rhs_calls_mean=total_rhs_calls / runs,
            time_per_rhs_seconds=elapsed / total_rhs_calls,
        ),
        result,
    )


def format_seconds(value: float) -> str:
    return f"{value:.6f}"


def format_small_seconds(value: float) -> str:
    return f"{value:.9f}"


def print_table(results: list[TimedResult]) -> None:
    headers = [
        "case",
        "per run s",
        "rhs/run",
        "time/rhs s",
        "max err",
        "vs SciPy",
    ]

    scipy = next(result for result in results if result.label == "SciPy RK45")
    rows = []

    for result in results:
        ratio = result.per_run_seconds / scipy.per_run_seconds

        if result.max_abs_error is None:
            error = "-"
        else:
            error = f"{result.max_abs_error:.3e}"

        rows.append(
            [
                result.label,
                format_seconds(result.per_run_seconds),
                f"{result.rhs_calls_mean:.0f}",
                format_small_seconds(result.time_per_rhs_seconds),
                error,
                f"{ratio:.2f}x",
            ]
        )

    widths = [
        max(len(str(row[index])) for row in [headers, *rows])
        for index in range(len(headers))
    ]

    def line(values: list[str]) -> str:
        return "  ".join(
            str(value).ljust(width)
            for value, width in zip(values, widths, strict=True)
        )

    print(line(headers))
    print(line(["-" * width for width in widths]))

    for row in rows:
        print(line(row))


def benchmark_size(size: int) -> None:
    runs = RUNS_BY_SIZE[size]

    print("=" * 88)
    print(f"Vector size: {size}")
    print(f"Runs:        {runs}")
    print()

    scipy_timed, scipy_result = time_runs(
        label="SciPy RK45",
        size=size,
        runs=runs,
        function=lambda: run_scipy_rk45(size),
    )

    timed_results = [scipy_timed]

    for label, scheme_class in STARK_SCHEMES:
        timed, result = time_runs(
            label=label,
            size=size,
            runs=runs,
            function=lambda scheme_class=scheme_class: run_stark_scheme(
                size,
                scheme_class,
            ),
        )

        timed.max_abs_error = float(
            np.max(np.abs(result.final_value - scipy_result.final_value))
        )
        timed_results.append(timed)

    print_table(timed_results)
    print()


def main() -> None:
    print("STARK scheme comparison against SciPy RK45")
    print(f"Tolerance: rtol={RTOL}, atol={ATOL}")
    print(f"Interval:  present={START}, step={STEP}, stop={STOP}")
    print()

    for size in SIZES:
        benchmark_size(size)


if __name__ == "__main__":
    main()