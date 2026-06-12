from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import perf_counter

from stark import Interval
from stark.engines.accelerators import AcceleratorNone
from stark.block import Block
from stark.executor.safety import ExecutorSafety
from stark.methods.inverters import InverterGMRES, InverterPolicy, InverterTolerance
from stark.methods.resolvents import (
    ResolventAnderson,
    ResolventBroyden,
    ResolventCoupledNewton,
    ResolventCoupledPicard,
    ResolventNewton,
    ResolventPicard,
    ResolventPolicy,
    ResolventTolerance,
)
from stark.methods.schemes.implicit.fixed import GAUSS_LEGENDRE4_TABLEAU


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_DIR = ROOT / "benchmarks" / "resolvents" / "results"

ALPHA = 0.05
RATE = -1.0
ATOL = 1.0e-10
RTOL = 1.0e-10
MAX_ITERATIONS = 32


@dataclass(slots=True)
class ScalarState:
    value: float = 0.0


@dataclass(slots=True)
class ScalarTranslation:
    value: float = 0.0

    def __call__(self, origin: ScalarState, result: ScalarState) -> None:
        result.value = origin.value + self.value

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: ScalarTranslation) -> ScalarTranslation:
        return ScalarTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> ScalarTranslation:
        return ScalarTranslation(scalar * self.value)

    @staticmethod
    def scale(a: float, x: ScalarTranslation, out: ScalarTranslation) -> ScalarTranslation:
        out.value = a * x.value
        return out

    @staticmethod
    def combine2(
        a0: float,
        x0: ScalarTranslation,
        a1: float,
        x1: ScalarTranslation,
        out: ScalarTranslation,
    ) -> ScalarTranslation:
        out.value = a0 * x0.value + a1 * x1.value
        return out

    linear_combine = [scale, combine2]


class ScalarAllocator:
    def allocate_state(self) -> ScalarState:
        return ScalarState()

    def copy_state(self, source: ScalarState, out: ScalarState) -> None:
        out.value = source.value

    def allocate_translation(self) -> ScalarTranslation:
        return ScalarTranslation()


class ScalarDerivative:
    def __init__(self, rate: float) -> None:
        self.rate = rate

    def __call__(self, interval: Interval, state: ScalarState, out: ScalarTranslation) -> None:
        del interval
        out.value = self.rate * state.value


class ScalarLinearizer:
    def __init__(self, rate: float) -> None:
        self.rate = rate

    def __call__(self, interval: Interval, state: ScalarState, out) -> None:
        del interval, state

        def apply(translation: ScalarTranslation, result: ScalarTranslation) -> None:
            result.value = self.rate * translation.value

        out.apply = apply


class ResolventSmokeCase:
    """Reusable configured resolvent case for manual timing."""

    __slots__ = ("alpha", "interval", "name", "out", "residual_buffer", "resolvent", "state")

    def __init__(
        self,
        name: str,
        resolvent,
        stage_count: int,
        *,
        alpha: float = ALPHA,
    ) -> None:
        self.name = name
        self.resolvent = resolvent
        self.alpha = alpha
        self.interval = Interval(present=0.0, step=alpha, stop=alpha)
        self.state = ScalarState(1.0)
        self.out = Block([ScalarTranslation() for _ in range(stage_count)])
        self.residual_buffer = Block([ScalarTranslation() for _ in range(stage_count)])
        self.resolvent.bind(self.interval, self.state)
        self.validate()

    def solve_once(self) -> None:
        self.resolvent(self.alpha, None, self.out)

    def solve_many(self, count: int) -> None:
        for _ in range(count):
            self.solve_once()

    def validate(self) -> None:
        self.solve_once()
        self.resolvent.residual(self.out, self.residual_buffer)
        error = self.residual_buffer.norm()
        if error > 1.0e-8:
            raise RuntimeError(f"{self.name} failed benchmark validation with residual {error:g}.")


@dataclass(frozen=True, slots=True)
class TimingResult:
    name: str
    repeat: int
    solves_per_sample: int
    best: float
    median: float
    worst: float

    @property
    def best_us_per_solve(self) -> float:
        return 1_000_000.0 * self.best / self.solves_per_sample

    @property
    def median_us_per_solve(self) -> float:
        return 1_000_000.0 * self.median / self.solves_per_sample

    @property
    def worst_us_per_solve(self) -> float:
        return 1_000_000.0 * self.worst / self.solves_per_sample


@dataclass(frozen=True, slots=True)
class BenchmarkMetadata:
    timestamp_utc: str
    python: str
    platform: str
    git_commit: str
    git_dirty: bool
    repeat: int
    warmup: int
    solves_per_sample: int
    alpha: float
    rate: float
    atol: float
    rtol: float
    max_iterations: int


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    metadata: BenchmarkMetadata
    results: list[TimingResult]


def scalar_inner_product(left: ScalarTranslation, right: ScalarTranslation) -> float:
    return left.value * right.value


def make_policy(max_iterations: int = MAX_ITERATIONS) -> ResolventPolicy:
    return ResolventPolicy(max_iterations=max_iterations)


def make_tolerance() -> ResolventTolerance:
    return ResolventTolerance(atol=ATOL, rtol=RTOL)


def make_inverter(allocator: ScalarAllocator) -> InverterGMRES:
    return InverterGMRES(
        allocator,
        scalar_inner_product,
        ExecutorTolerance=InverterTolerance(atol=ATOL, rtol=RTOL),
        policy=InverterPolicy(max_iterations=8, restart=4),
        safety=ExecutorSafety(block_sizes=False),
        accelerator=AcceleratorNone(),
    )


def build_cases() -> list[ResolventSmokeCase]:
    derivative = ScalarDerivative(RATE)
    linearizer = ScalarLinearizer(RATE)

    picard_allocator = ScalarAllocator()
    anderson_allocator = ScalarAllocator()
    broyden_allocator = ScalarAllocator()
    newton_allocator = ScalarAllocator()
    coupled_picard_allocator = ScalarAllocator()
    coupled_newton_allocator = ScalarAllocator()

    return [
        ResolventSmokeCase(
            "Picard / one-stage",
            ResolventPicard(
                picard_allocator,
                ExecutorTolerance=make_tolerance(),
                policy=make_policy(),
                safety=ExecutorSafety(block_sizes=False),
                accelerator=AcceleratorNone(),
            ),
            1,
        ),
        ResolventSmokeCase(
            "Anderson / one-stage",
            ResolventAnderson(
                anderson_allocator,
                scalar_inner_product,
                ExecutorTolerance=make_tolerance(),
                policy=make_policy(),
                depth=4,
                safety=ExecutorSafety(block_sizes=False),
                accelerator=AcceleratorNone(),
            ),
            1,
        ),
        ResolventSmokeCase(
            "Broyden / one-stage",
            ResolventBroyden(
                broyden_allocator,
                scalar_inner_product,
                ExecutorTolerance=make_tolerance(),
                policy=make_policy(),
                depth=4,
                safety=ExecutorSafety(block_sizes=False),
                accelerator=AcceleratorNone(),
            ),
            1,
        ),
        ResolventSmokeCase(
            "Newton / one-stage",
            ResolventNewton(
                newton_allocator,
                linearizer,
                make_inverter(newton_allocator),
                ExecutorTolerance=make_tolerance(),
                policy=make_policy(8),
                safety=ExecutorSafety(block_sizes=False),
                accelerator=AcceleratorNone(),
            ),
            1,
        ),
        ResolventSmokeCase(
            "Picard / coupled stages",
            ResolventCoupledPicard(
                coupled_picard_allocator,
                GAUSS_LEGENDRE4_TABLEAU,
                ExecutorTolerance=make_tolerance(),
                policy=make_policy(),
                safety=ExecutorSafety(block_sizes=False),
                accelerator=AcceleratorNone(),
            ),
            len(GAUSS_LEGENDRE4_TABLEAU.c),
        ),
        ResolventSmokeCase(
            "Newton / coupled stages",
            ResolventCoupledNewton(
                coupled_newton_allocator,
                GAUSS_LEGENDRE4_TABLEAU,
                linearizer,
                make_inverter(coupled_newton_allocator),
                ExecutorTolerance=make_tolerance(),
                policy=make_policy(8),
                safety=ExecutorSafety(block_sizes=False),
                accelerator=AcceleratorNone(),
            ),
            len(GAUSS_LEGENDRE4_TABLEAU.c),
        ),
    ]


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def current_metadata(*, repeat: int, warmup: int, solves_per_sample: int) -> BenchmarkMetadata:
    status = git_output("status", "--porcelain")
    return BenchmarkMetadata(
        timestamp_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        python=platform.python_version(),
        platform=platform.platform(),
        git_commit=git_output("rev-parse", "HEAD"),
        git_dirty=bool(status),
        repeat=repeat,
        warmup=warmup,
        solves_per_sample=solves_per_sample,
        alpha=ALPHA,
        rate=RATE,
        atol=ATOL,
        rtol=RTOL,
        max_iterations=MAX_ITERATIONS,
    )


def time_case(
    case: ResolventSmokeCase,
    *,
    repeat: int,
    warmup: int,
    solves_per_sample: int,
) -> TimingResult:
    for _ in range(warmup):
        case.solve_many(solves_per_sample)

    samples: list[float] = []
    for _ in range(repeat):
        start = perf_counter()
        case.solve_many(solves_per_sample)
        samples.append(perf_counter() - start)

    return TimingResult(
        name=case.name,
        repeat=repeat,
        solves_per_sample=solves_per_sample,
        best=min(samples),
        median=median(samples),
        worst=max(samples),
    )


def run_benchmark(*, repeat: int, warmup: int, solves_per_sample: int) -> BenchmarkRun:
    return BenchmarkRun(
        metadata=current_metadata(
            repeat=repeat,
            warmup=warmup,
            solves_per_sample=solves_per_sample,
        ),
        results=[
            time_case(
                case,
                repeat=repeat,
                warmup=warmup,
                solves_per_sample=solves_per_sample,
            )
            for case in build_cases()
        ],
    )


def baseline_path(name: str, baseline_dir: Path = DEFAULT_BASELINE_DIR) -> Path:
    safe_name = name.strip().replace(" ", "_")
    if not safe_name:
        raise ValueError("Baseline name must not be empty.")
    return baseline_dir / f"resolvent_refactor_{safe_name}.json"


def save_run(run: BenchmarkRun, name: str) -> Path:
    path = baseline_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(run_to_json(run), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_run(name: str) -> BenchmarkRun:
    path = baseline_path(name)
    data = json.loads(path.read_text(encoding="utf-8"))
    return run_from_json(data)


def run_to_json(run: BenchmarkRun) -> dict:
    return {
        "metadata": asdict(run.metadata),
        "results": [asdict(result) for result in run.results],
    }


def run_from_json(data: dict) -> BenchmarkRun:
    return BenchmarkRun(
        metadata=BenchmarkMetadata(**data["metadata"]),
        results=[
            TimingResult(**result)
            for result in data["results"]
        ],
    )


def print_current_results(run: BenchmarkRun) -> None:
    print("STARK resolvent refactor smoke benchmark")
    print("Manual local timings only; do not use as a CI gate.")
    print()
    print(f"commit: {run.metadata.git_commit}")
    print(f"dirty:  {run.metadata.git_dirty}")
    print(f"python: {run.metadata.python}")
    print(f"solves per timed sample: {run.metadata.solves_per_sample}")
    print()
    print(f"{'case':32} {'best us':>12} {'median us':>12} {'worst us':>12} {'runs':>6}")
    print("-" * 78)

    for result in run.results:
        print(
            f"{result.name:32} "
            f"{result.best_us_per_solve:12.3f} "
            f"{result.median_us_per_solve:12.3f} "
            f"{result.worst_us_per_solve:12.3f} "
            f"{result.repeat:6d}"
        )


def print_comparison(current: BenchmarkRun, baseline: BenchmarkRun) -> None:
    baseline_by_name = {result.name: result for result in baseline.results}

    print("STARK resolvent refactor smoke benchmark comparison")
    print("Ratios use median timing per solve. Values above 1.00x are slower than baseline.")
    print()
    print(f"baseline commit: {baseline.metadata.git_commit}")
    print(f"current  commit: {current.metadata.git_commit}")
    print()
    print(
        f"{'case':32} "
        f"{'baseline us':>12} "
        f"{'current us':>12} "
        f"{'ratio':>9} "
        f"{'delta us':>10}"
    )
    print("-" * 78)

    for result in current.results:
        old = baseline_by_name.get(result.name)
        if old is None:
            print(
                f"{result.name:32} "
                f"{'missing':>12} "
                f"{result.median_us_per_solve:12.3f} "
                f"{'n/a':>9} "
                f"{'n/a':>10}"
            )
            continue

        ratio = result.median_us_per_solve / old.median_us_per_solve if old.median else float("inf")
        delta_us = result.median_us_per_solve - old.median_us_per_solve

        print(
            f"{result.name:32} "
            f"{old.median_us_per_solve:12.3f} "
            f"{result.median_us_per_solve:12.3f} "
            f"{ratio:8.2f}x "
            f"{delta_us:10.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local non-gating smoke timings for representative resolvent paths."
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=7,
        help="number of timed samples per case",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="number of untimed warmup samples per case",
    )
    parser.add_argument(
        "--solves-per-sample",
        type=int,
        default=200,
        help="number of repeated solves inside one timed sample",
    )
    parser.add_argument(
        "--save-baseline",
        metavar="NAME",
        help="save current results as a named local baseline",
    )
    parser.add_argument(
        "--compare-baseline",
        metavar="NAME",
        help="compare current results against a named local baseline",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if args.solves_per_sample < 1:
        raise SystemExit("--solves-per-sample must be at least 1")

    current = run_benchmark(
        repeat=args.repeat,
        warmup=args.warmup,
        solves_per_sample=args.solves_per_sample,
    )
    print_current_results(current)

    if args.save_baseline:
        path = save_run(current, args.save_baseline)
        print()
        print(f"Saved baseline: {path}")

    if args.compare_baseline:
        baseline = load_run(args.compare_baseline)
        print()
        print_comparison(current, baseline)


if __name__ == "__main__":
    main()
