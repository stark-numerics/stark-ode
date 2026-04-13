from __future__ import annotations

from statistics import median
from time import perf_counter

from benchmarks.fput import common, diffrax, scipy, stark


def prewarm_runner(prepare_runner, problem, tolerances, initial_conditions, reference, allow_failure=False) -> None:
    try:
        solve_once = prepare_runner(problem, tolerances, initial_conditions, reference)
        solve_once()
    except Exception:
        if not allow_failure:
            raise


def timed_runner(
    library,
    solver,
    prepare_runner,
    repeats,
    problem,
    tolerances,
    initial_conditions,
    reference,
    allow_failure=False,
):
    setup_elapsed = None
    warmup_elapsed = None
    try:
        started = perf_counter()
        solve_once = prepare_runner(problem, tolerances, initial_conditions, reference)
        setup_elapsed = perf_counter() - started

        started = perf_counter()
        solve_once()
        warmup_elapsed = perf_counter() - started

        durations = []
        result = None
        for _ in range(repeats):
            started = perf_counter()
            result = solve_once()
            durations.append(perf_counter() - started)
    except Exception as exc:  # pragma: no cover - optional solver stacks vary locally
        if not allow_failure:
            raise
        return {
            "library": library,
            "solver": solver,
            "error": None,
            "steps": None,
            "setup": setup_elapsed,
            "warmup": warmup_elapsed,
            "preparation": None if setup_elapsed is None else setup_elapsed + (warmup_elapsed or 0.0),
            "median": None,
            "min": None,
            "note": f"{type(exc).__name__}: {exc}",
        }

    return {
        "library": result["library"],
        "solver": result["solver"],
        "error": result["error"],
        "steps": result["steps"],
        "setup": float(setup_elapsed),
        "warmup": float(warmup_elapsed),
        "preparation": float(setup_elapsed + warmup_elapsed),
        "median": float(median(durations)),
        "min": float(min(durations)),
        "note": "",
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


def describe_problem(problem, tolerances, reference_tolerances, reference, reference_elapsed):
    print("FPUT-Beta Benchmark")
    print()
    print("This benchmark solves the Fermi-Pasta-Ulam-Tsingou beta lattice with fixed endpoints:")
    print("  dq_i/dt = p_i")
    print("  dp_i/dt = q_{i+1} - 2 q_i + q_{i-1} + beta[(q_{i+1} - q_i)^3 - (q_i - q_{i-1})^3]")
    print("  q_0 = q_{N+1} = 0")
    print()
    print("Problem parameters:")
    print(f"  chain size: {problem['chain_size']}")
    print(f"  time interval: [{problem['t0']}, {problem['t1']}]")
    print(f"  beta={problem['beta']}")
    print(f"  amplitude={problem['amplitude']}")
    print()
    print("Initial condition:")
    print("  q_i = amplitude * sin(pi i / (N + 1))")
    print("  p_i = 0")
    print()
    print("Reference solution:")
    print("  generated with SciPy solve_ivp using DOP853")
    print(f"  rtol={reference_tolerances['rtol']:.0e}, atol={reference_tolerances['atol']:.0e}")
    print(f"  reference steps: {reference['steps']}")
    print(f"  reference generation time: {reference_elapsed:.6f}s")
    print()
    print("Compared runs:")
    print(f"  rtol={tolerances['rtol']:.0e}, atol={tolerances['atol']:.0e}")
    print(f"  STARK initial step: {tolerances['initial_step']:.0e}")
    print(
        "  STARK acceleration: "
        + ("Numba-jitted RHS and fused translation kernels" if stark.NUMBA_AVAILABLE else "NumPy fallback kernels")
    )
    print("  all compared solver stacks are prewarmed once before timed rows")
    print("  each method performs setup once, then one complete untimed warmup solve")
    print("  preparation timing excludes one-time cross-row JIT or tracing costs")
    print("  reference generation is excluded from the timing table")
    print("  timings are CPU wall-clock timings for repeated solves of one medium-sized problem")
    print("  equal tolerance numbers do not imply identical solver error norms")
    print()


def print_summary(rows):
    completed = [row for row in rows if row["error"] is not None and row["median"] is not None]
    if not completed:
        print("Summary: no compared method completed.")
        return

    most_accurate = min(completed, key=lambda row: row["error"])
    lowest_preparation = min(completed, key=lambda row: row["preparation"])
    fastest_median = min(completed, key=lambda row: row["median"])
    fastest_min = min(completed, key=lambda row: row["min"])

    print("Summary:")
    print(
        f"  Best accuracy: {most_accurate['library']} {most_accurate['solver']} "
        f"with error {most_accurate['error']:.6e}."
    )
    print(
        f"  Lowest preparation time: {lowest_preparation['library']} {lowest_preparation['solver']} "
        f"at {lowest_preparation['preparation']:.6f}s."
    )
    print(
        f"  Fastest median time: {fastest_median['library']} {fastest_median['solver']} "
        f"at {fastest_median['median']:.6f}s."
    )
    print(
        f"  Fastest single run: {fastest_min['library']} {fastest_min['solver']} "
        f"at {fastest_min['min']:.6f}s."
    )


def main() -> None:
    problem = common.PROBLEM_PARAMETERS
    tolerances = common.TOLERANCE_PARAMETERS
    reference_tolerances = common.REFERENCE_TOLERANCE_PARAMETERS
    initial_conditions = common.INITIAL_CONDITIONS
    repeats = common.BENCHMARK_PARAMETERS["repeats"]

    started = perf_counter()
    reference = scipy.run_reference(problem, reference_tolerances, initial_conditions)
    reference_elapsed = perf_counter() - started

    prewarm_runner(stark.prepare_rkck, problem, tolerances, initial_conditions, reference)
    prewarm_runner(stark.prepare_rkdp, problem, tolerances, initial_conditions, reference)
    prewarm_runner(scipy.prepare_rk45, problem, tolerances, initial_conditions, reference)
    prewarm_runner(scipy.prepare_dop853, problem, tolerances, initial_conditions, reference)
    prewarm_runner(diffrax.prepare_tsit5, problem, tolerances, initial_conditions, reference, allow_failure=True)
    prewarm_runner(diffrax.prepare_dopri5, problem, tolerances, initial_conditions, reference, allow_failure=True)

    rows = [
        timed_runner("STARK", "RKCK", stark.prepare_rkck, repeats, problem, tolerances, initial_conditions, reference),
        timed_runner("STARK", "RKDP", stark.prepare_rkdp, repeats, problem, tolerances, initial_conditions, reference),
        timed_runner("SciPy", "RK45", scipy.prepare_rk45, repeats, problem, tolerances, initial_conditions, reference),
        timed_runner("SciPy", "DOP853", scipy.prepare_dop853, repeats, problem, tolerances, initial_conditions, reference),
        timed_runner(
            "Diffrax",
            "Tsit5",
            diffrax.prepare_tsit5,
            repeats,
            problem,
            tolerances,
            initial_conditions,
            reference,
            allow_failure=True,
        ),
        timed_runner(
            "Diffrax",
            "Dopri5",
            diffrax.prepare_dopri5,
            repeats,
            problem,
            tolerances,
            initial_conditions,
            reference,
            allow_failure=True,
        ),
    ]

    describe_problem(problem, tolerances, reference_tolerances, reference, reference_elapsed)

    print("Error Table")
    print(
        render_table(
            ("library", "solver", "steps", "error", "note"),
            [
                (
                    row["library"],
                    row["solver"],
                    "-" if row["steps"] is None else str(row["steps"]),
                    "-" if row["error"] is None else f"{row['error']:.6e}",
                    row["note"],
                )
                for row in rows
            ],
        )
    )
    print()

    print("Preparation Timing Table")
    print(
        render_table(
            ("library", "solver", "setup", "warmup", "total", "note"),
            [
                (
                    row["library"],
                    row["solver"],
                    "-" if row["setup"] is None else f"{row['setup']:.6f}s",
                    "-" if row["warmup"] is None else f"{row['warmup']:.6f}s",
                    "-" if row["preparation"] is None else f"{row['preparation']:.6f}s",
                    row["note"],
                )
                for row in rows
            ],
        )
    )
    print()

    print("Run Timing Table")
    print(
        render_table(
            ("library", "solver", "median", "min", "repeats", "note"),
            [
                (
                    row["library"],
                    row["solver"],
                    "-" if row["median"] is None else f"{row['median']:.6f}s",
                    "-" if row["min"] is None else f"{row['min']:.6f}s",
                    str(repeats),
                    row["note"],
                )
                for row in rows
            ],
        )
    )
    print()
    print_summary(rows)


if __name__ == "__main__":
    main()
