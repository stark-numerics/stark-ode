from __future__ import annotations

from statistics import median
from time import perf_counter

from examples.comparison.robertson import common, diffrax, scipy, stark


def prewarm_runner(prepare_runner, problem, parameters, initial_conditions, reference) -> None:
    solve_once = prepare_runner(problem, parameters, initial_conditions, reference)
    solve_once()


def timed_runner(
    library,
    solver,
    prepare_runner,
    repeats,
    problem,
    parameters,
    initial_conditions,
    reference,
):
    started = perf_counter()
    solve_once = prepare_runner(problem, parameters, initial_conditions, reference)
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


def describe_problem(problem, tolerances, stark_parameters, reference_tolerances, reference, reference_elapsed):
    print("Robertson Benchmark")
    print()
    print("This benchmark solves the stiff Robertson chemical kinetics system:")
    print("  dy1/dt = -0.04 y1 + 1e4 y2 y3")
    print("  dy2/dt = 0.04 y1 - 1e4 y2 y3 - 3e7 y2^2")
    print("  dy3/dt = 3e7 y2^2")
    print()
    print("Problem parameters:")
    print(f"  time interval: [{problem['t0']}, {problem['t1']}]")
    print()
    print("Initial condition:")
    print("  y(0) = [1, 0, 0]")
    print()
    print("Reference solution:")
    print("  generated with SciPy solve_ivp using Radau")
    print(f"  rtol={reference_tolerances['rtol']:.0e}, atol={reference_tolerances['atol']:.0e}")
    print(f"  reference steps: {reference['steps']}")
    print(f"  reference generation time: {reference_elapsed:.6f}s")
    print()
    print("Compared runs:")
    print(f"  SciPy compared tolerances: rtol={tolerances['rtol']:.0e}, atol={tolerances['atol']:.0e}")
    print(
        "  STARK adaptive tolerance: "
        f"rtol={stark_parameters['tolerance_rtol']:.0e}, "
        f"atol={stark_parameters['tolerance_atol']:.0e}"
    )
    print(f"  STARK initial step: {stark_parameters['step']:.0e}")
    print(
        "  STARK resolver tolerance/policy: "
        f"atol={stark_parameters['resolution_atol']:.0e}, "
        f"rtol={stark_parameters['resolution_rtol']:.0e}, "
        f"max_iterations={stark_parameters['resolution_max_iterations']}"
    )
    print(
        "  STARK acceleration: "
        + (
            "selected accelerator: numba, with compiled RHS/Jacobian kernels and fused translation kernels active"
            if stark.USE_NUMBA_ACCELERATION
            else "selected accelerator: numba, but it is unavailable here so the benchmark is using pure Python kernels"
        )
    )
    print("  STARK Robertson uses a fully implicit Kvaerno4 solve with a custom exact cubic resolvent")
    print("  the whole Robertson right-hand side is treated implicitly in that resolvent")
    print("  Diffrax uses Kvaerno5, an adaptive stiffly accurate ESDIRK method")
    print("  all compared solver stacks are prewarmed once before timed rows")
    print("  each method performs setup once, then one complete untimed warmup solve")
    print("  preparation timing excludes one-time cross-row JIT or tracing costs")
    print("  reference generation is excluded from the timing table")
    print("  timings are CPU wall-clock timings for repeated solves of one small stiff problem")
    print("  the solver rows do not use identical step-control or nonlinear-solve strategies")
    print()


def print_summary(rows):
    most_accurate = min(rows, key=lambda row: row["error"])
    lowest_preparation = min(rows, key=lambda row: row["preparation"])
    fastest_median = min(rows, key=lambda row: row["median"])
    fastest_min = min(rows, key=lambda row: row["min"])

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
    stark_parameters = common.STARK_PARAMETERS
    diffrax_parameters = common.DIFFRAX_PARAMETERS
    initial_conditions = common.INITIAL_CONDITIONS
    repeats = common.BENCHMARK_PARAMETERS["repeats"]

    started = perf_counter()
    reference = scipy.run_reference(problem, reference_tolerances, initial_conditions)
    reference_elapsed = perf_counter() - started

    prewarm_runner(stark.prepare_kvaerno4_full_custom, problem, stark_parameters, initial_conditions, reference)
    prewarm_runner(scipy.prepare_radau, problem, tolerances, initial_conditions, reference)
    prewarm_runner(scipy.prepare_bdf, problem, tolerances, initial_conditions, reference)
    if diffrax.DIFFRAX_AVAILABLE:
        prewarm_runner(
            lambda problem, tolerances, initial_conditions, reference: diffrax.prepare_kvaerno5(
                problem,
                tolerances,
                diffrax_parameters,
                initial_conditions,
                reference,
            ),
            problem,
            tolerances,
            initial_conditions,
            reference,
        )

    rows = [
        timed_runner("STARK", "Kvaerno4 Full Cubic", stark.prepare_kvaerno4_full_custom, repeats, problem, stark_parameters, initial_conditions, reference),
        timed_runner("SciPy", "Radau", scipy.prepare_radau, repeats, problem, tolerances, initial_conditions, reference),
        timed_runner("SciPy", "BDF", scipy.prepare_bdf, repeats, problem, tolerances, initial_conditions, reference),
    ]

    if diffrax.DIFFRAX_AVAILABLE:
        rows.append(
            timed_runner(
                "Diffrax",
                "Kvaerno5",
                lambda problem, tolerances, initial_conditions, reference: diffrax.prepare_kvaerno5(
                    problem,
                    tolerances,
                    diffrax_parameters,
                    initial_conditions,
                    reference,
                ),
                repeats,
                problem,
                tolerances,
                initial_conditions,
                reference,
            )
        )

    describe_problem(problem, tolerances, stark_parameters, reference_tolerances, reference, reference_elapsed)

    print("Error Table")
    print(
        render_table(
            ("library", "solver", "steps", "error"),
            [
                (
                    row["library"],
                    row["solver"],
                    str(row["steps"]),
                    f"{row['error']:.6e}",
                )
                for row in rows
            ],
        )
    )
    print()

    print("Preparation Timing Table")
    print(
        render_table(
            ("library", "solver", "setup", "warmup", "total"),
            [
                (
                    row["library"],
                    row["solver"],
                    f"{row['setup']:.6f}s",
                    f"{row['warmup']:.6f}s",
                    f"{row['preparation']:.6f}s",
                )
                for row in rows
            ],
        )
    )
    print()

    print("Run Timing Table")
    print(
        render_table(
            ("library", "solver", "median", "min", "repeats"),
            [
                (
                    row["library"],
                    row["solver"],
                    f"{row['median']:.6f}s",
                    f"{row['min']:.6f}s",
                    str(repeats),
                )
                for row in rows
            ],
        )
    )
    print()
    print_summary(rows)


if __name__ == "__main__":
    main()









