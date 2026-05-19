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

from stark.accelerators import Accelerator
from stark.block.operator import BlockOperator
from stark.contracts import Block
from stark.execution.safety import Safety
from stark.inverters import InverterBiCGStab, InverterFGMRES, InverterGMRES, InverterPolicy, InverterTolerance


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_DIR = ROOT / "benchmarks" / "inverters" / "results"

ATOL = 1.0e-11
RTOL = 1.0e-11
MAX_ITERATIONS = 32
RESTART = 8
COUPLED_SIZE = 8


@dataclass(slots=True)
class ScalarTranslation:
    value: float = 0.0

    def __call__(self, origin: object, result: object) -> None:
        del origin, result

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: ScalarTranslation) -> ScalarTranslation:
        return ScalarTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> ScalarTranslation:
        return ScalarTranslation(scalar * self.value)


class ScalarWorkbench:
    def allocate_state(self) -> object:
        return object()

    def copy_state(self, dst: object, src: object) -> None:
        del dst, src

    def allocate_translation(self) -> ScalarTranslation:
        return ScalarTranslation()


class DenseBlockOperator(BlockOperator):
    """Small dense block operator expressed through STARK's block contract."""

    __slots__ = ("matrix",)

    def __init__(self, matrix: tuple[tuple[float, ...], ...]) -> None:
        super().__init__([], check_sizes=False)
        self.matrix = matrix

    def __call__(self, block: Block, out: Block) -> None:
        for row_index, row in enumerate(self.matrix):
            total = 0.0
            for column_index, coefficient in enumerate(row):
                total += coefficient * block[column_index].value
            out[row_index].value = total


class DiagonalBlockOperator(BlockOperator):
    """One-item scalar block operator for call-overhead-sensitive rows."""

    __slots__ = ("scale",)

    def __init__(self, scale: float) -> None:
        super().__init__([], check_sizes=False)
        self.scale = scale

    def __call__(self, block: Block, out: Block) -> None:
        out[0].value = self.scale * block[0].value


class InverterSmokeCase:
    """Reusable configured inverter case for manual timing."""

    __slots__ = ("inverter", "name", "rhs", "solution")

    def __init__(self, name: str, inverter, rhs: Block) -> None:
        self.name = name
        self.inverter = inverter
        self.rhs = rhs
        self.solution = Block([ScalarTranslation() for _ in range(len(rhs))])
        self.validate()

    def validate(self) -> None:
        self.solve_once()
        residual = residual_norm(self.inverter.operator, self.solution, self.rhs)
        if residual > 1.0e-8:
            raise RuntimeError(f"{self.name} failed benchmark validation with residual {residual:g}.")

    def zero_solution(self) -> None:
        for item in self.solution:
            item.value = 0.0

    def solve_once(self) -> None:
        self.zero_solution()
        self.inverter(self.rhs, self.solution)

    def solve_many(self, count: int) -> None:
        for _ in range(count):
            self.solve_once()


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
    coupled_size: int
    atol: float
    rtol: float
    max_iterations: int
    restart: int


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    metadata: BenchmarkMetadata
    results: list[TimingResult]


def scalar_inner_product(left: ScalarTranslation, right: ScalarTranslation) -> float:
    return left.value * right.value


def make_policy() -> InverterPolicy:
    return InverterPolicy(max_iterations=MAX_ITERATIONS, restart=RESTART)


def make_tolerance() -> InverterTolerance:
    return InverterTolerance(atol=ATOL, rtol=RTOL)


def make_inverter(inverter_type):
    return inverter_type(
        ScalarWorkbench(),
        scalar_inner_product,
        tolerance=make_tolerance(),
        policy=make_policy(),
        safety=Safety(block_sizes=False),
        accelerator=Accelerator.none(),
    )


def make_scalar_rhs() -> Block:
    return Block([ScalarTranslation(2.0)])


def make_coupled_rhs(size: int = COUPLED_SIZE) -> Block:
    return Block([ScalarTranslation(1.0 + index / size) for index in range(size)])


def make_coupled_matrix(size: int = COUPLED_SIZE) -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    for row_index in range(size):
        row = [0.0] * size
        row[row_index] = 2.0 + 0.02 * row_index
        if row_index > 0:
            row[row_index - 1] = -0.11
        if row_index + 1 < size:
            row[row_index + 1] = 0.07
        if row_index + 2 < size:
            row[row_index + 2] = -0.015
        rows.append(tuple(row))
    return tuple(rows)


def residual_norm(operator, solution: Block, rhs: Block) -> float:
    applied = Block([ScalarTranslation() for _ in range(len(rhs))])
    operator(solution, applied)
    total = 0.0
    for applied_item, rhs_item in zip(applied, rhs, strict=True):
        error = rhs_item.value - applied_item.value
        total += error * error
    return total**0.5


def build_cases() -> list[InverterSmokeCase]:
    cases: list[InverterSmokeCase] = []
    inverter_types = (
        ("GMRES", InverterGMRES),
        ("FGMRES", InverterFGMRES),
        ("BiCGStab", InverterBiCGStab),
    )

    for label, inverter_type in inverter_types:
        inverter = make_inverter(inverter_type)
        inverter.bind(DiagonalBlockOperator(2.0))
        cases.append(InverterSmokeCase(f"{label} / scalar block", inverter, make_scalar_rhs()))

    matrix = make_coupled_matrix()
    rhs = make_coupled_rhs()
    for label, inverter_type in inverter_types:
        inverter = make_inverter(inverter_type)
        inverter.bind(DenseBlockOperator(matrix))
        cases.append(InverterSmokeCase(f"{label} / coupled block", inverter, rhs))

    return cases


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
        coupled_size=COUPLED_SIZE,
        atol=ATOL,
        rtol=RTOL,
        max_iterations=MAX_ITERATIONS,
        restart=RESTART,
    )


def time_case(
    case: InverterSmokeCase,
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
    return baseline_dir / f"inverter_refactor_{safe_name}.json"


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
    print("STARK inverter refactor smoke benchmark")
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

    print("STARK inverter refactor smoke benchmark comparison")
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
        description="Run local non-gating smoke timings for representative inverter paths."
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
