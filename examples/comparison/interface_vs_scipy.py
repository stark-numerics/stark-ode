from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np
from scipy.integrate import solve_ivp

from stark import Interval
from stark.interface import StarkDerivative, StarkIVP


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


@dataclass
class RunResult:
    final_value: np.ndarray
    rhs_calls: int | None = None


class CountedReturnRHS:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, t: float, y: Any) -> Any:
        self.calls += 1
        return -0.5 * y


class CountedInPlaceRHS:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, t: float, y: Any, dy: Any) -> None:
        self.calls += 1
        dy[:] = -0.5 * y


def make_initial(size: int) -> np.ndarray:
    return np.linspace(1.0, 2.0, size)


def run_stark_live_return(size: int) -> RunResult:
    rhs = CountedReturnRHS()
    initial = make_initial(size)

    ivp = StarkIVP(
        derivative=rhs,
        initial=initial,
        interval=Interval(present=START, step=STEP, stop=STOP),
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
        raise RuntimeError("STARK live return integration produced no output.")

    return RunResult(
        final_value=np.asarray(final_state.value).copy(),
        rhs_calls=rhs.calls,
    )


def run_stark_live_in_place(size: int) -> RunResult:
    rhs = CountedInPlaceRHS()
    derivative = StarkDerivative.in_place(rhs)
    initial = make_initial(size)

    ivp = StarkIVP(
        derivative=derivative,
        initial=initial,
        interval=Interval(present=START, step=STEP, stop=STOP),
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
        raise RuntimeError("STARK live in-place integration produced no output.")

    return RunResult(
        final_value=np.asarray(final_state.value).copy(),
        rhs_calls=rhs.calls,
    )


def run_scipy_matched(size: int) -> RunResult:
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
    function: Callable[[int], RunResult],
    size: int,
    runs: int,
) -> RunResult:
    function(size)

    started = perf_counter()

    result = None
    total_rhs_calls = 0

    for _ in range(runs):
        result = function(size)
        if result.rhs_calls is not None:
            total_rhs_calls += result.rhs_calls

    elapsed = perf_counter() - started

    if result is None:
        raise RuntimeError(f"{label} produced no result.")

    print(f"  {label}:")
    print(f"    total:         {elapsed:.6f} s")
    print(f"    per run:       {elapsed / runs:.6f} s")

    if result.rhs_calls is not None:
        print(f"    rhs/run last:  {result.rhs_calls}")
        print(f"    rhs/run mean:  {total_rhs_calls / runs:.2f}")
        print(f"    time/rhs mean: {elapsed / total_rhs_calls:.9f} s")

    return result


def compare_to_reference(label: str, value: np.ndarray, reference: np.ndarray) -> None:
    difference = np.abs(value - reference)

    print(f"  {label} max abs error vs SciPy:")
    print(f"    {np.max(difference)}")


def benchmark_size(size: int) -> None:
    runs = RUNS_BY_SIZE[size]

    print("=" * 80)
    print(f"Vector size: {size}")
    print(f"Runs:        {runs}")
    print()

    stark_return = time_runs(
        "STARK live return-style",
        run_stark_live_return,
        size,
        runs,
    )
    print()

    stark_in_place = time_runs(
        "STARK live in-place",
        run_stark_live_in_place,
        size,
        runs,
    )
    print()

    scipy = time_runs(
        "SciPy solve_ivp RK45 matched tolerance",
        run_scipy_matched,
        size,
        runs,
    )
    print()

    compare_to_reference(
        "STARK return-style",
        stark_return.final_value,
        scipy.final_value,
    )
    compare_to_reference(
        "STARK in-place",
        stark_in_place.final_value,
        scipy.final_value,
    )
    print()


def main() -> None:
    print("Large-vector exponential decay benchmark")
    print(f"Tolerance: rtol={RTOL}, atol={ATOL}")
    print(f"Interval:  present={START}, step={STEP}, stop={STOP}")
    print()

    for size in SIZES:
        benchmark_size(size)


if __name__ == "__main__":
    main()