from __future__ import annotations

import argparse
import platform
import subprocess
from dataclasses import dataclass
from statistics import median
from time import perf_counter
from typing import Callable

import numpy as np

from stark import Configuration, Integrator, Interval, IntegratorStepper, Tolerance
from stark.accelerators import AcceleratorNone, AcceleratorNumba
from stark.algebraist.arity import AlgebraistArity
from stark.algebraist.generator import AlgebraistGeneratorLinearCombine, AlgebraistGeneratorSpecialist
from stark.algebraist.layout import (
    AlgebraistLayout,
    AlgebraistLayoutBroadcast,
    AlgebraistLayoutField,
    AlgebraistLayoutLooped,
)
from stark.methods.resolvents import ResolventCoupledPicard, ResolventPicard
from stark.methods.schemes.implicit.fixed.backward_euler import SchemeBackwardEuler
from stark.methods.schemes.implicit.fixed.crank_nicolson import SchemeCrankNicolson
from stark.methods.schemes.implicit.fixed.crouzeix_dirk3 import SchemeCrouzeixDIRK3
from stark.methods.schemes.implicit.fixed.gauss_legendre4 import SchemeGaussLegendre4
from stark.methods.schemes.implicit.fixed.implicit_midpoint import SchemeImplicitMidpoint
from stark.methods.schemes.implicit.fixed.lobatto_iiic4 import SchemeLobattoIIIC4
from stark.methods.schemes.implicit.fixed.radau_iia5 import SchemeRadauIIA5


GENERATED_CASES = (
    ("crouzeix-dirk3", SchemeCrouzeixDIRK3, "picard"),
    ("crank-nicolson", SchemeCrankNicolson, "picard"),
    ("gauss-legendre4", SchemeGaussLegendre4, "coupled-picard"),
    ("implicit-midpoint", SchemeImplicitMidpoint, "picard"),
)
NO_OP_CASES = (
    ("backward-euler", SchemeBackwardEuler, "picard"),
    ("lobatto-iiic4", SchemeLobattoIIIC4, "coupled-picard"),
    ("radau-iia5", SchemeRadauIIA5, "coupled-picard"),
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


class ArrayAllocator:
    __slots__ = ("count", "linear_combine")

    def __init__(self, count: int) -> None:
        self.count = count
        self.linear_combine: tuple[Callable[..., object], ...] = ()

    def allocate_state(self) -> ArrayState:
        return ArrayState(np.zeros(self.count, dtype=np.float64))

    def copy_state(self, source: ArrayState, out: ArrayState) -> None:
        np.copyto(out.value, source.value)

    def allocate_translation(self) -> ArrayTranslation:
        return ArrayTranslation(
            np.zeros(self.count, dtype=np.float64),
            self.linear_combine,
        )


def decay_rhs(
    interval: Interval,
    state: ArrayState,
    out: ArrayTranslation,
) -> None:
    del interval
    out.value[...] = -state.value


def make_initial_state(count: int) -> ArrayState:
    grid = np.linspace(0.0, 1.0, count, dtype=np.float64)
    return ArrayState(1.0 + 0.25 * np.sin(2.0 * np.pi * grid))


@dataclass(frozen=True, slots=True)
class ArrayAlgebraist:
    linear_combine: tuple[Callable[..., object], ...]
    specialist: AlgebraistGeneratorSpecialist


def make_algebraist(policy, allocator: ArrayAllocator, accelerator=None) -> ArrayAlgebraist:
    active_accelerator = accelerator if accelerator is not None else AcceleratorNone()
    layout = AlgebraistLayout(
        fields=(AlgebraistLayoutField("value", "value", policy=policy),),
    )
    general = AlgebraistGeneratorLinearCombine(
        translation=allocator.allocate_translation(),
        allocator=allocator,
        layout=layout,
        accelerator=active_accelerator,
    )
    linear_combine = tuple(general.provide(AlgebraistArity(arity)) for arity in range(1, 13))
    allocator.linear_combine = linear_combine
    specialist = AlgebraistGeneratorSpecialist(
        translation=allocator.allocate_translation(),
        allocator=allocator,
        layout=layout,
        accelerator=active_accelerator,
    )
    return ArrayAlgebraist(linear_combine=linear_combine, specialist=specialist)


def numba_accelerator():
    try:
        return AcceleratorNumba(cache=False)
    except ModuleNotFoundError:
        return None


def make_resolvent(scheme_cls, allocator: ArrayAllocator, kind: str):
    kwargs = dict(
        configuration=Configuration(
            resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12),
            resolvent_maximum_steps=16,
        ),
        accelerator=AcceleratorNone(),
        tableau=scheme_cls.tableau,
    )
    if kind == "picard":
        return ResolventPicard(allocator, **kwargs)
    if kind == "coupled-picard":
        return ResolventCoupledPicard(allocator, **kwargs)
    raise ValueError(f"Unknown resolvent kind {kind!r}.")


class BenchmarkCase:
    __slots__ = ("integrator", "interval", "stepper", "name", "state")

    def __init__(
        self,
        *,
        name: str,
        scheme,
        state: ArrayState,
        interval: Interval,
    ) -> None:
        self.name = name
        self.stepper = IntegratorStepper(scheme)
        self.integrator = Integrator(configuration=Configuration(check_progress=False))
        self.state = state
        self.interval = interval

    def solve_once(self) -> ArrayState:
        state = self.state.copy()
        interval = self.interval.copy()
        for _interval, _state in self.integrator.mutating_trajectory(self.stepper, interval, state):
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
    resolvent_kind: str,
    count: int,
    steps: int,
    step: float,
    policy=None,
    accelerator=None,
) -> BenchmarkCase:
    allocator = ArrayAllocator(count)
    algebraist = make_algebraist(policy, allocator, accelerator) if policy is not None else None
    specialist = algebraist.specialist if algebraist is not None else None
    resolvent = make_resolvent(scheme_cls, allocator, resolvent_kind)
    scheme = scheme_cls(
        decay_rhs,
        allocator,
        resolvent=resolvent,
        specialist=specialist,
    )
    return BenchmarkCase(
        name=label,
        scheme=scheme,
        state=make_initial_state(count),
        interval=Interval(present=0.0, step=step, stop=steps * step),
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
    reference: ArrayState,
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

    for scheme_name, scheme_cls, resolvent_kind in (*GENERATED_CASES, *NO_OP_CASES):
        generic = make_case(
            label=f"{scheme_name}/generic",
            scheme_cls=scheme_cls,
            resolvent_kind=resolvent_kind,
            count=count,
            steps=steps,
            step=step,
            policy=None,
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

        plain = make_case(
            label=f"{scheme_name}/algebraist-broadcast",
            scheme_cls=scheme_cls,
            resolvent_kind=resolvent_kind,
            count=count,
            steps=steps,
            step=step,
            policy=AlgebraistLayoutBroadcast(),
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
            accelerated = make_case(
                label=f"{scheme_name}/algebraist-looped-numba",
                scheme_cls=scheme_cls,
                resolvent_kind=resolvent_kind,
                count=count,
                steps=steps,
                step=step,
                policy=AlgebraistLayoutLooped(rank=1),
                accelerator=numba,
            )
            results.append(
                time_case(
                    accelerated,
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
    print("Implicit fixed Algebraist timing comparison")
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
        description="Run local non-gating timings for implicit fixed Algebraist paths."
    )
    parser.add_argument("--count", type=int, default=256, help="state vector size")
    parser.add_argument("--steps", type=int, default=8, help="fixed implicit steps")
    parser.add_argument("--step", type=float, default=1.0e-2, help="step size")
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
