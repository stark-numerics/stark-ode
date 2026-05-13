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
from typing import Callable

from stark import Executor, Integrator, Interval, Marcher, Tolerance
from stark.accelerators import Accelerator
from stark.resolvents import ResolventPicard
from stark.resolvents.policy import ResolventPolicy
from stark.schemes.explicit_adaptive.bogacki_shampine import SchemeBogackiShampine
from stark.schemes.explicit_fixed.rk4 import SchemeRK4
from stark.schemes.imex_adaptive.kennedy_carpenter32 import SchemeKennedyCarpenter32
from stark.schemes.implicit_adaptive.kvaerno3 import SchemeKvaerno3
from stark.schemes.implicit_fixed.backward_euler import SchemeBackwardEuler


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_DIR = ROOT / "benchmarks" / "schemes" / "results"

RTOL = 1.0e-7
ATOL = 1.0e-9
INITIAL_STEP = 1.0e-3
STOP = 0.1


@dataclass(slots=True)
class ScalarState:
    value: float = 0.0

    def copy(self) -> ScalarState:
        return ScalarState(self.value)


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


class ScalarWorkbench:
    def allocate_state(self) -> ScalarState:
        return ScalarState()

    def copy_state(self, dst: ScalarState, src: ScalarState) -> None:
        dst.value = src.value

    def allocate_translation(self) -> ScalarTranslation:
        return ScalarTranslation()


@dataclass(slots=True)
class SplitDerivative:
    explicit: object
    implicit: object


def negative_decay(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval
    out.value = -state.value


def zero_rhs(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval, state
    out.value = 0.0


def make_executor() -> Executor:
    return Executor(
        tolerance=Tolerance(atol=ATOL, rtol=RTOL),
        accelerator=Accelerator.none(),
    )


class SchemeSmokeCase:
    """Reusable local solve case for manual timing.

    This benchmark is a local smoke check, not a CI gate. Each timed sample
    gets a fresh interval and state while reusing the configured scheme,
    marcher, and integrator.
    """

    __slots__ = ("integrator", "interval", "marcher", "name", "state")

    def __init__(
        self,
        name: str,
        scheme,
        interval: Interval,
        state: ScalarState,
        run_executor: Executor,
    ) -> None:
        self.name = name
        self.marcher = Marcher(scheme, run_executor)
        self.integrator = Integrator()
        self.interval = interval
        self.state = state

    def solve_once(self) -> None:
        interval = self.interval.copy()
        state = self.state.copy()

        for _snapshot_interval, _snapshot_state in self.integrator.live(
            self.marcher,
            interval,
            state,
        ):
            pass


@dataclass(frozen=True, slots=True)
class TimingResult:
    name: str
    repeat: int
    best: float
    median: float
    worst: float

    @property
    def best_ms(self) -> float:
        return 1000.0 * self.best

    @property
    def median_ms(self) -> float:
        return 1000.0 * self.median

    @property
    def worst_ms(self) -> float:
        return 1000.0 * self.worst


@dataclass(frozen=True, slots=True)
class BenchmarkMetadata:
    timestamp_utc: str
    python: str
    platform: str
    git_commit: str
    git_dirty: bool
    repeat: int
    warmup: int
    initial_step: float
    stop: float
    atol: float
    rtol: float


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    metadata: BenchmarkMetadata
    results: list[TimingResult]


def make_interval() -> Interval:
    return Interval(present=0.0, step=INITIAL_STEP, stop=STOP)


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


def current_metadata(*, repeat: int, warmup: int) -> BenchmarkMetadata:
    status = git_output("status", "--porcelain")
    return BenchmarkMetadata(
        timestamp_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        python=platform.python_version(),
        platform=platform.platform(),
        git_commit=git_output("rev-parse", "HEAD"),
        git_dirty=bool(status),
        repeat=repeat,
        warmup=warmup,
        initial_step=INITIAL_STEP,
        stop=STOP,
        atol=ATOL,
        rtol=RTOL,
    )


def make_resolvent(
    derivative: Callable[[Interval, ScalarState, ScalarTranslation], None],
    workbench: ScalarWorkbench,
    tableau,
) -> ResolventPicard:
    return ResolventPicard(
        derivative,
        workbench,
        tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=ResolventPolicy(max_iterations=16),
        accelerator=Accelerator.none(),
        tableau=tableau,
    )


def make_rk4_case() -> SchemeSmokeCase:
    workbench = ScalarWorkbench()
    scheme = SchemeRK4(negative_decay, workbench)

    return SchemeSmokeCase(
        "fixed explicit / RK4",
        scheme,
        make_interval(),
        ScalarState(1.0),
        make_executor(),
    )


def make_explicit_adaptive_case() -> SchemeSmokeCase:
    workbench = ScalarWorkbench()
    scheme = SchemeBogackiShampine(negative_decay, workbench)

    return SchemeSmokeCase(
        "adaptive explicit / Bogacki-Shampine",
        scheme,
        make_interval(),
        ScalarState(1.0),
        make_executor(),
    )


def make_implicit_fixed_case() -> SchemeSmokeCase:
    workbench = ScalarWorkbench()
    resolvent = make_resolvent(
        negative_decay,
        workbench,
        SchemeBackwardEuler.tableau,
    )
    scheme = SchemeBackwardEuler(
        negative_decay,
        workbench,
        resolvent=resolvent,
    )

    return SchemeSmokeCase(
        "fixed implicit / Backward Euler",
        scheme,
        make_interval(),
        ScalarState(1.0),
        make_executor(),
    )


def make_implicit_adaptive_case() -> SchemeSmokeCase:
    workbench = ScalarWorkbench()
    resolvent = make_resolvent(
        negative_decay,
        workbench,
        SchemeKvaerno3.tableau,
    )
    scheme = SchemeKvaerno3(
        negative_decay,
        workbench,
        resolvent=resolvent,
    )

    return SchemeSmokeCase(
        "adaptive implicit / Kvaerno3",
        scheme,
        make_interval(),
        ScalarState(1.0),
        make_executor(),
    )


def make_imex_adaptive_case() -> SchemeSmokeCase:
    workbench = ScalarWorkbench()
    derivative = SplitDerivative(
        explicit=zero_rhs,
        implicit=negative_decay,
    )
    resolvent = make_resolvent(
        negative_decay,
        workbench,
        SchemeKennedyCarpenter32.tableau,
    )
    scheme = SchemeKennedyCarpenter32(
        derivative,
        workbench,
        resolvent=resolvent,
    )

    return SchemeSmokeCase(
        "adaptive IMEX / Kennedy-Carpenter32",
        scheme,
        make_interval(),
        ScalarState(1.0),
        make_executor(),
    )


def build_cases() -> list[SchemeSmokeCase]:
    return [
        make_rk4_case(),
        make_explicit_adaptive_case(),
        make_implicit_fixed_case(),
        make_implicit_adaptive_case(),
        make_imex_adaptive_case(),
    ]


def time_case(
    case: SchemeSmokeCase,
    *,
    repeat: int,
    warmup: int,
) -> TimingResult:
    for _ in range(warmup):
        case.solve_once()

    samples: list[float] = []
    for _ in range(repeat):
        start = perf_counter()
        case.solve_once()
        samples.append(perf_counter() - start)

    return TimingResult(
        name=case.name,
        repeat=repeat,
        best=min(samples),
        median=median(samples),
        worst=max(samples),
    )


def run_benchmark(*, repeat: int, warmup: int) -> BenchmarkRun:
    return BenchmarkRun(
        metadata=current_metadata(repeat=repeat, warmup=warmup),
        results=[
            time_case(case, repeat=repeat, warmup=warmup)
            for case in build_cases()
        ],
    )


def baseline_path(name: str, baseline_dir: Path = DEFAULT_BASELINE_DIR) -> Path:
    safe_name = name.strip().replace(" ", "_")
    if not safe_name:
        raise ValueError("Baseline name must not be empty.")
    return baseline_dir / f"scheme_refactor_{safe_name}.json"


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
    print("STARK scheme refactor smoke benchmark")
    print("Manual local timings only; do not use as a CI gate.")
    print()
    print(f"commit: {run.metadata.git_commit}")
    print(f"dirty:  {run.metadata.git_dirty}")
    print(f"python: {run.metadata.python}")
    print()
    print(f"{'case':45} {'best ms':>12} {'median ms':>12} {'worst ms':>12} {'runs':>6}")
    print("-" * 93)

    for result in run.results:
        print(
            f"{result.name:45} "
            f"{result.best_ms:12.3f} "
            f"{result.median_ms:12.3f} "
            f"{result.worst_ms:12.3f} "
            f"{result.repeat:6d}"
        )


def print_comparison(current: BenchmarkRun, baseline: BenchmarkRun) -> None:
    baseline_by_name = {result.name: result for result in baseline.results}

    print("STARK scheme refactor smoke benchmark comparison")
    print("Ratios use median timing. Values above 1.00x are slower than baseline.")
    print()
    print(f"baseline commit: {baseline.metadata.git_commit}")
    print(f"current  commit: {current.metadata.git_commit}")
    print()
    print(
        f"{'case':45} "
        f"{'baseline ms':>12} "
        f"{'current ms':>12} "
        f"{'ratio':>9} "
        f"{'delta ms':>10}"
    )
    print("-" * 93)

    for result in current.results:
        old = baseline_by_name.get(result.name)
        if old is None:
            print(
                f"{result.name:45} "
                f"{'missing':>12} "
                f"{result.median_ms:12.3f} "
                f"{'n/a':>9} "
                f"{'n/a':>10}"
            )
            continue

        ratio = result.median / old.median if old.median else float("inf")
        delta_ms = result.median_ms - old.median_ms

        print(
            f"{result.name:45} "
            f"{old.median_ms:12.3f} "
            f"{result.median_ms:12.3f} "
            f"{ratio:8.2f}x "
            f"{delta_ms:10.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run local non-gating smoke timings for representative scheme "
            "families after the scheme-owned-call refactor."
        )
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=7,
        help="number of timed runs per case",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="number of untimed warmup solves per case",
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

    current = run_benchmark(repeat=args.repeat, warmup=args.warmup)
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