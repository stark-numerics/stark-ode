from __future__ import annotations

from statistics import median
from time import perf_counter

from benchmarks.fitzhugh_nagumo_1d import common, diffrax, scipy, stark


def announce(message: str) -> None:
    print(message, flush=True)


def prewarm_runner(prepare_runner, problem, parameters, initial_conditions, reference):
    solve_once = prepare_runner(problem, parameters, initial_conditions, reference)
    started = perf_counter()
    result = solve_once()
    elapsed = perf_counter() - started
    return result, elapsed


def timed_runner(library, solver, prepare_runner, repeats, problem, parameters, initial_conditions, reference):
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

    row = {
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
    announce(
        f"Completed {row['library']} {row['solver']}: "
        f"steps={row['steps']}, "
        f"error={row['error']:.6e}, "
        f"setup={row['setup']:.3f}s, "
        f"warmup={row['warmup']:.3f}s, "
        f"median={row['median']:.3f}s, "
        f"min={row['min']:.3f}s"
    )
    return row


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
    print("FitzHugh-Nagumo 1D Benchmark")
    print()
    print("This benchmark solves a periodic one-dimensional reaction-diffusion FitzHugh-Nagumo system.")
    print()
    print("Problem parameters:")
    print(f"  grid size: {problem['grid_size']}")
    print(f"  domain length: {problem['length']}")
    print(f"  time interval: [{problem['t0']}, {problem['t1']}]")
    print(
        f"  diffusivity_u={problem['diffusivity_u']}, epsilon={problem['epsilon']}, a={problem['a']}, b={problem['b']}"
    )
    print()
    print("Reference solution:")
    print("  generated with SciPy solve_ivp using Radau")
    print(f"  rtol={reference_tolerances['rtol']:.0e}, atol={reference_tolerances['atol']:.0e}")
    print(f"  reference steps: {reference['steps']}")
    print(f"  reference generation time: {reference_elapsed:.6f}s")
    print()
    print("Compared runs:")
    print(f"  SciPy/Diffrax tolerances: rtol={tolerances['rtol']:.0e}, atol={tolerances['atol']:.0e}")
    print(
        "  STARK adaptive tolerance: "
        f"rtol={stark_parameters['tolerance_rtol']:.0e}, atol={stark_parameters['tolerance_atol']:.0e}"
    )
    print(
        "  STARK resolver tolerance/policy: "
        f"atol={stark_parameters['resolution_atol']:.0e}, "
        f"rtol={stark_parameters['resolution_rtol']:.0e}, "
        f"max_iterations={stark_parameters['resolution_max_iterations']}"
    )
    print(
        "  STARK inverter tolerance/policy: "
        f"atol={stark_parameters['inversion_atol']:.0e}, "
        f"rtol={stark_parameters['inversion_rtol']:.0e}, "
        f"max_iterations={stark_parameters['inversion_max_iterations']}, "
        f"restart={stark_parameters['inversion_restart']}"
    )
    print(
        "  STARK acceleration: "
        + (
            "Numba-jitted stencil, derivative, Jacobian, and fused translation kernels"
            if stark.NUMBA_AVAILABLE
            else "Python kernels with branchless hot-path translation calls"
        )
    )
    print("  STARK shows the current best local combination: Kvaerno3 with Anderson acceleration")
    print("  Diffrax uses Kvaerno5, an adaptive stiffly accurate ESDIRK method")
    print("  all compared solver stacks are prewarmed once before timed rows")
    print("  each method performs setup once, then one complete untimed warmup solve")
    print("  preparation timing excludes one-time cross-row JIT or tracing costs")
    print("  reference generation is excluded from the timing table")
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

    announce("Generating SciPy Radau reference solution...")
    started = perf_counter()
    reference = scipy.run_reference(problem, reference_tolerances, initial_conditions)
    reference_elapsed = perf_counter() - started
    announce(
        f"Reference ready: steps={reference['steps']}, "
        f"elapsed={reference_elapsed:.3f}s"
    )

    stark_runners = [
        ("STARK", "Kvaerno3 Anderson", stark.prepare_kvaerno3_anderson, stark_parameters),
    ]
    external_runners = [
        ("SciPy", "Radau", scipy.prepare_radau, tolerances),
        ("SciPy", "BDF", scipy.prepare_bdf, tolerances),
    ]

    for library, solver, prepare_runner, parameters in [*stark_runners, *external_runners]:
        announce(f"Prewarming {library} {solver}...")
        result, elapsed = prewarm_runner(prepare_runner, problem, parameters, initial_conditions, reference)
        announce(
            f"Prewarm complete: {result['solver']} "
            f"steps={result['steps']}, error={result['error']:.6e}, elapsed={elapsed:.3f}s"
        )

    if diffrax.DIFFRAX_AVAILABLE:
        announce("Prewarming Diffrax Kvaerno5...")
        result, elapsed = prewarm_runner(
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
        announce(f"Prewarm complete: {result['solver']} steps={result['steps']}, error={result['error']:.6e}, elapsed={elapsed:.3f}s")

    rows = []

    for library, solver, prepare_runner, parameters in [*stark_runners, *external_runners]:
        announce(f"Timing {library} {solver}...")
        rows.append(timed_runner(library, solver, prepare_runner, repeats, problem, parameters, initial_conditions, reference))

    if diffrax.DIFFRAX_AVAILABLE:
        announce("Timing Diffrax Kvaerno5...")
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
            [(row["library"], row["solver"], str(row["steps"]), f"{row['error']:.6e}") for row in rows],
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
