from __future__ import annotations

import argparse
import platform
import subprocess
from dataclasses import dataclass
from statistics import median
from time import perf_counter
from typing import Callable

import numpy as np

from stark import Executor, Integrator, Interval, Marcher, Safety, Tolerance
from stark.accelerators import Accelerator
from stark.algebraist.classic import Algebraist, AlgebraistBroadcast, AlgebraistField, AlgebraistLooped
from stark.resolvents import ResolventPicard
from stark.resolvents.support.policy import ResolventPolicy
from stark.schemes.imex_adaptive import SchemeKennedyCarpenter32


IMEX_ADAPTIVE_CASES = (
    ("kennedy-carpenter32", SchemeKennedyCarpenter32),
)


@dataclass(slots=True)
class ArrayState:
    value: np.ndarray

    def copy(self) -> "ArrayState":
        return ArrayState(self.value.copy())


@dataclass(slots=True)
class ArrayTranslation:
    value: np.ndarray
    linear_combine: tuple[Callable[..., object], ...] = ()

    def __call__(self, origin: ArrayState, result: ArrayState) -> None:
        result.value[...] = origin.value + self.value

    def norm(self) -> float:
        return float(np.sqrt(np.mean(self.value * self.value)))

    def __add__(self, other: "ArrayTranslation") -> "ArrayTranslation":
        return ArrayTranslation(self.value + other.value, self.linear_combine)

    def __rmul__(self, scalar: float) -> "ArrayTranslation":
        return ArrayTranslation(scalar * self.value, self.linear_combine)


class ArrayWorkbench:
    __slots__ = ("algebraist", "count")

    def __init__(self, count: int, algebraist: Algebraist | None = None) -> None:
        self.count = count
        self.algebraist = algebraist

    def allocate_state(self) -> ArrayState:
        return ArrayState(np.zeros(self.count, dtype=np.float64))

    def copy_state(self, dst: ArrayState, src: ArrayState) -> None:
        np.copyto(dst.value, src.value)

    def allocate_translation(self) -> ArrayTranslation:
        linear_combine = ()
        if self.algebraist is not None:
            linear_combine = self.algebraist.linear_combine
        return ArrayTranslation(
            np.zeros(self.count, dtype=np.float64),
            linear_combine,
        )


@dataclass(slots=True)
class SplitDerivative:
    explicit: object
    implicit: object


def explicit_rhs(
    interval: Interval,
    state: ArrayState,
    out: ArrayTranslation,
) -> None:
    del interval
    out.value[...] = 0.05 * state.value


def implicit_rhs(
    interval: Interval,
    state: ArrayState,
    out: ArrayTranslation,
) -> None:
    del interval
    out.value[...] = -state.value


def make_initial_state(count: int) -> ArrayState:
    grid = np.linspace(0.0, 1.0, count, dtype=np.float64)
    return ArrayState(1.0 + 0.25 * np.sin(2.0 * np.pi * grid))


def make_algebraist(policy, accelerator=None) -> Algebraist:
    return Algebraist(
        fields=(AlgebraistField("value", "value", policy=policy),),
        accelerator=accelerator,
        generate_norm="rms",
    )


def numba_accelerator():
    try:
        return Accelerator.numba(cache=False)
    except ModuleNotFoundError:
        return None


def make_executor() -> Executor:
    return Executor(
        tolerance=Tolerance(atol=1.0e-8, rtol=1.0e-6),
        safety=Safety.fast(),
    )


def make_resolvent(scheme_cls, workbench: ArrayWorkbench) -> ResolventPicard:
    return ResolventPicard(
        implicit_rhs,
        workbench,
        tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=ResolventPolicy(max_iterations=16),
        accelerator=Accelerator.none(),
        tableau=scheme_cls.tableau,
    )


class BenchmarkCase:
    __slots__ = ("integrator", "interval", "marcher", "name", "state")

    def __init__(
        self,
        *,
        name: str,
        scheme,
        state: ArrayState,
        interval: Interval,
        executor: Executor,
    ) -> None:
        self.name = name
        self.marcher = Marcher(scheme, executor)
        self.integrator = Integrator(executor=executor)
        self.state = state
        self.interval = interval

    def solve_once(self) -> ArrayState:
        state = self.state.copy()
        interval = self.interval.copy()
        for _interval, _state in self.integrator.live(self.marcher, interval, state):
            pass
        return state


@dataclass(frozen=True, slots=True)
class TimingResult:
    name: str
    category: str
    repeat: int
    best: float
    median: float
    worst: float
    rms_vs_generic: float
    max_vs_generic: float

    @property
    def best_ms(self) -> float:
        return 1000.0 * self.best

    @property
    def median_ms(self) -> float:
        return 1000.0 * self.median

    @property
    def worst_ms(self) -> float:
        return 1000.0 * self.worst


def make_case(
    *,
    label: str,
    scheme_cls,
    count: int,
    steps: int,
    step: float,
    algebraist: Algebraist | None,
) -> BenchmarkCase:
    workbench = ArrayWorkbench(count, algebraist)
    derivative = SplitDerivative(
        explicit=explicit_rhs,
        implicit=implicit_rhs,
    )
    resolvent = make_resolvent(scheme_cls, workbench)
    scheme = scheme_cls(
        derivative,
        workbench,
        resolvent=resolvent,
        algebraist=algebraist,
    )
    return BenchmarkCase(
        name=label,
        scheme=scheme,
        state=make_initial_state(count),
        interval=Interval(present=0.0, step=step, stop=steps * step),
        executor=make_executor(),
    )


def compare_states(left: ArrayState, right: ArrayState) -> tuple[float, float]:
    difference = left.value - right.value
    return (
        float(np.sqrt(np.mean(difference * difference))),
        float(np.max(np.abs(difference))),
    )


def time_case(
    case: BenchmarkCase,
    *,
    repeat: int,
    warmup: int,
    minimum_warmup: int = 0,
    reference: ArrayState,
    category: str,
) -> TimingResult:
    for _ in range(max(warmup, minimum_warmup)):
        case.solve_once()

    samples: list[float] = []
    result = reference
    for _ in range(repeat):
        start = perf_counter()
        result = case.solve_once()
        samples.append(perf_counter() - start)

    rms, max_abs = compare_states(result, reference)
    return TimingResult(
        name=case.name,
        category=category,
        repeat=repeat,
        best=min(samples),
        median=median(samples),
        worst=max(samples),
        rms_vs_generic=rms,
        max_vs_generic=max_abs,
    )


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def build_results(
    *,
    count: int,
    steps: int,
    step: float,
    repeat: int,
    warmup: int,
) -> list[TimingResult]:
    results: list[TimingResult] = []
    numba = numba_accelerator()

    for scheme_name, scheme_cls in IMEX_ADAPTIVE_CASES:
        generic = make_case(
            label=f"{scheme_name}/generic",
            scheme_cls=scheme_cls,
            count=count,
            steps=steps,
            step=step,
            algebraist=None,
        )
        reference = generic.solve_once()
        results.append(
            time_case(
                generic,
                repeat=repeat,
                warmup=warmup,
                reference=reference,
                category=scheme_name,
            )
        )

        algebraist = make_algebraist(AlgebraistBroadcast())
        plain = make_case(
            label=f"{scheme_name}/algebraist-broadcast",
            scheme_cls=scheme_cls,
            count=count,
            steps=steps,
            step=step,
            algebraist=algebraist,
        )
        results.append(
            time_case(
                plain,
                repeat=repeat,
                warmup=warmup,
                reference=reference,
                category=scheme_name,
            )
        )

        if numba is not None:
            accelerated_algebraist = make_algebraist(AlgebraistLooped(rank=1), numba)
            accelerated = make_case(
                label=f"{scheme_name}/algebraist-looped-numba",
                scheme_cls=scheme_cls,
                count=count,
                steps=steps,
                step=step,
                algebraist=accelerated_algebraist,
            )
            results.append(
                time_case(
                    accelerated,
                    repeat=repeat,
                    warmup=warmup,
                    minimum_warmup=1,
                    reference=reference,
                    category=scheme_name,
                )
            )

    return results


def print_results(
    results: list[TimingResult],
    *,
    count: int,
    steps: int,
    step: float,
    repeat: int,
    warmup: int,
) -> None:
    print("IMEX adaptive Algebraist timing comparison")
    print("Manual local timings only; do not use as a CI gate.")
    print("Numba cases run at least one untimed warmup to exclude compilation.")
    print()
    print(f"commit: {git_output('rev-parse', 'HEAD')}")
    print(f"dirty:  {bool(git_output('status', '--porcelain'))}")
    print(f"python: {platform.python_version()}")
    print()
    print(f"state size : {count}")
    print(f"target steps: {steps}")
    print(f"initial step: {step}")
    print(f"warmup     : {warmup}")
    print(f"repeat     : {repeat}")
    print()
    print(
        f"{'case':42} {'best ms':>10} {'median ms':>10} {'worst ms':>10} "
        f"{'rms':>10} {'max':>10} {'runs':>5}"
    )
    print("-" * 105)

    for result in results:
        print(
            f"{result.name:42} "
            f"{result.best_ms:10.3f} "
            f"{result.median_ms:10.3f} "
            f"{result.worst_ms:10.3f} "
            f"{result.rms_vs_generic:10.3e} "
            f"{result.max_vs_generic:10.3e} "
            f"{result.repeat:5d}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local non-gating timings for IMEX adaptive Algebraist paths."
    )
    parser.add_argument("--count", type=int, default=256, help="state vector size")
    parser.add_argument("--steps", type=int, default=8, help="target interval in initial steps")
    parser.add_argument("--step", type=float, default=1.0e-3, help="initial step size")
    parser.add_argument("--repeat", type=int, default=5, help="timed runs per case")
    parser.add_argument("--warmup", type=int, default=1, help="untimed warmup runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be at least 1")
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1")
    if args.step <= 0.0:
        raise SystemExit("--step must be positive")
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")

    results = build_results(
        count=args.count,
        steps=args.steps,
        step=args.step,
        repeat=args.repeat,
        warmup=args.warmup,
    )
    print_results(
        results,
        count=args.count,
        steps=args.steps,
        step=args.step,
        repeat=args.repeat,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    main()
