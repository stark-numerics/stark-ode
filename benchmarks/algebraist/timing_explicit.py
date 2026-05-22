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
from stark.schemes.explicit_adaptive import SchemeCashKarp, SchemeDormandPrince
from stark.schemes.explicit_fixed import SchemeEuler, SchemeRK4


EXPLICIT_CASES = (
    ("euler", SchemeEuler, "fixed"),
    ("rk4", SchemeRK4, "fixed"),
    ("cash-karp", SchemeCashKarp, "adaptive"),
    ("dormand-prince", SchemeDormandPrince, "adaptive"),
)


@dataclass(slots=True)
class PairState:
    q: np.ndarray
    p: np.ndarray

    def copy(self) -> "PairState":
        return PairState(self.q.copy(), self.p.copy())


@dataclass(slots=True)
class PairTranslation:
    dq: np.ndarray
    dp: np.ndarray
    linear_combine: tuple[Callable[..., object], ...] = ()

    def __call__(self, origin: PairState, result: PairState) -> None:
        result.q[...] = origin.q + self.dq
        result.p[...] = origin.p + self.dp

    def norm(self) -> float:
        return float(np.sqrt(np.mean(self.dq * self.dq + self.dp * self.dp)))

    def __add__(self, other: "PairTranslation") -> "PairTranslation":
        return PairTranslation(
            self.dq + other.dq,
            self.dp + other.dp,
            self.linear_combine,
        )

    def __rmul__(self, scalar: float) -> "PairTranslation":
        return PairTranslation(
            scalar * self.dq,
            scalar * self.dp,
            self.linear_combine,
        )


class PairWorkbench:
    __slots__ = ("algebraist", "count")

    def __init__(self, count: int, algebraist: Algebraist | None = None) -> None:
        self.count = count
        self.algebraist = algebraist

    def allocate_state(self) -> PairState:
        return PairState(
            np.zeros(self.count, dtype=np.float64),
            np.zeros(self.count, dtype=np.float64),
        )

    def copy_state(self, dst: PairState, src: PairState) -> None:
        np.copyto(dst.q, src.q)
        np.copyto(dst.p, src.p)

    def allocate_translation(self) -> PairTranslation:
        linear_combine = ()
        if self.algebraist is not None:
            linear_combine = self.algebraist.linear_combine
        return PairTranslation(
            np.zeros(self.count, dtype=np.float64),
            np.zeros(self.count, dtype=np.float64),
            linear_combine,
        )


class PairDerivative:
    __slots__ = ("scratch",)

    def __init__(self, count: int) -> None:
        self.scratch = np.zeros(count, dtype=np.float64)

    def __call__(
        self,
        interval: Interval,
        state: PairState,
        out: PairTranslation,
    ) -> None:
        del interval
        out.dq[...] = state.p
        np.multiply(state.q, -0.1, out.dp)
        out.dp[...] += 0.05 * state.p


def make_initial_state(count: int) -> PairState:
    grid = np.linspace(0.0, 1.0, count, dtype=np.float64)
    return PairState(
        np.sin(2.0 * np.pi * grid),
        np.cos(2.0 * np.pi * grid),
    )


def make_algebraist(policy, accelerator=None) -> Algebraist:
    return Algebraist(
        fields=(
            AlgebraistField("dq", "q", policy=policy),
            AlgebraistField("dp", "p", policy=policy),
        ),
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


class BenchmarkCase:
    __slots__ = ("integrator", "interval", "marcher", "name", "state")

    def __init__(
        self,
        *,
        name: str,
        scheme,
        state: PairState,
        interval: Interval,
        executor: Executor,
    ) -> None:
        self.name = name
        self.marcher = Marcher(scheme, executor)
        self.integrator = Integrator(executor=executor)
        self.state = state
        self.interval = interval

    def solve_once(self) -> PairState:
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
    workbench = PairWorkbench(count, algebraist)
    scheme = scheme_cls(PairDerivative(count), workbench, algebraist=algebraist)
    return BenchmarkCase(
        name=label,
        scheme=scheme,
        state=make_initial_state(count),
        interval=Interval(present=0.0, step=step, stop=steps * step),
        executor=make_executor(),
    )


def compare_states(left: PairState, right: PairState) -> tuple[float, float]:
    q_difference = left.q - right.q
    p_difference = left.p - right.p
    rms = np.sqrt(np.mean(q_difference * q_difference + p_difference * p_difference))
    max_abs = max(np.max(np.abs(q_difference)), np.max(np.abs(p_difference)))
    return float(rms), float(max_abs)


def time_case(
    case: BenchmarkCase,
    *,
    repeat: int,
    warmup: int,
    reference: PairState,
    category: str,
) -> TimingResult:
    for _ in range(warmup):
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

    for scheme_name, scheme_cls, _kind in EXPLICIT_CASES:
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

        broadcast_algebraist = make_algebraist(AlgebraistBroadcast())
        broadcast = make_case(
            label=f"{scheme_name}/algebraist-broadcast",
            scheme_cls=scheme_cls,
            count=count,
            steps=steps,
            step=step,
            algebraist=broadcast_algebraist,
        )
        results.append(
            time_case(
                broadcast,
                repeat=repeat,
                warmup=warmup,
                reference=reference,
                category=scheme_name,
            )
        )

        if numba is not None:
            numba_algebraist = make_algebraist(AlgebraistLooped(rank=1), numba)
            numba_case = make_case(
                label=f"{scheme_name}/algebraist-looped-numba",
                scheme_cls=scheme_cls,
                count=count,
                steps=steps,
                step=step,
                algebraist=numba_algebraist,
            )
            results.append(
                time_case(
                    numba_case,
                    repeat=repeat,
                    warmup=warmup,
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
    print("Explicit Algebraist timing comparison")
    print("Manual local timings only; do not use as a CI gate.")
    print()
    print(f"commit: {git_output('rev-parse', 'HEAD')}")
    print(f"dirty:  {bool(git_output('status', '--porcelain'))}")
    print(f"python: {platform.python_version()}")
    print()
    print(f"state size : {count}")
    print(f"steps      : {steps}")
    print(f"step       : {step}")
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
        description="Run local non-gating timings for explicit Algebraist paths."
    )
    parser.add_argument("--count", type=int, default=1024, help="state vector size")
    parser.add_argument("--steps", type=int, default=32, help="integration steps")
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
