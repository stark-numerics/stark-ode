from __future__ import annotations

from time import perf_counter

from competition.robertson import common, diffrax, scipy, stark
from competition.runner import CompetitionData, CompetitionEntry, CompetitionRunner, render_table


def announce(message: str) -> None:
    print(message, flush=True)


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
        "  STARK adaptive Tolerance: "
        f"rtol={stark_parameters['tolerance_rtol']:.0e}, "
        f"atol={stark_parameters['tolerance_atol']:.0e}"
    )
    print(f"  STARK initial step: {stark_parameters['step']:.0e}")
    print(
        "  STARK resolver Tolerance/policy: "
        f"atol={stark_parameters['resolution_atol']:.0e}, "
        f"rtol={stark_parameters['resolution_rtol']:.0e}, "
        f"max_iterations={stark_parameters['resolution_max_iterations']}"
    )
    print(
        "  STARK acceleration: "
        "StarkEngineNumpy selects Numba when it is installed and otherwise uses unaccelerated callables"
    )
    print("  STARK Robertson prepares the carrier, allocator, derivative, and integrator explicitly")
    print("  Kvaerno5 Exact Cubic uses a custom analytic resolvent for this problem")
    print("  STARK Robertson also includes Kvaerno5 Newton, Chord, and VeryChord rows using the new inverter request path")
    print("  dense rows use the provider-free dense inverter and one-block nucleus path")
    print("  Diffrax uses Kvaerno5, an adaptive stiffly accurate ESDIRK method")
    print("  all compared solver stacks are prewarmed once before timed rows")
    print("  each method performs setup once, then one complete untimed warmup solve")
    print("  preparation timing excludes one-time cross-row JIT or tracing costs")
    print("  reference generation is excluded from the timing table")
    print("  timings are CPU wall-clock timings for repeated solves of one small stiff problem")
    print("  the solver rows do not use identical step-control or nonlinear-solve strategies")
    print()


def print_error_table(rows):
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


def print_preparation_table(rows):
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


def print_run_table(rows, repeats):
    print("Warm Run Timing Table")
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


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6e}"


def _format_optional_iteration(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def print_total_table(rows, repeats):
    print("Total Timing Table")
    print(
        render_table(
            ("library", "solver", "total median", "total min", "repeats", "note"),
            [
                (
                    row["library"],
                    row["solver"],
                    "-" if row["total_median"] is None else f"{row['total_median']:.6f}s",
                    "-" if row["total_min"] is None else f"{row['total_min']:.6f}s",
                    str(repeats),
                    row["note"],
                )
                for row in rows
            ],
        )
    )
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
    fastest_total_median = min(completed, key=lambda row: row["total_median"])
    fastest_total_min = min(completed, key=lambda row: row["total_min"])

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
        f"  Fastest warm median time: {fastest_median['library']} {fastest_median['solver']} "
        f"at {fastest_median['median']:.6f}s."
    )
    print(
        f"  Fastest warm single run: {fastest_min['library']} {fastest_min['solver']} "
        f"at {fastest_min['min']:.6f}s."
    )
    print(
        f"  Fastest total median time: {fastest_total_median['library']} {fastest_total_median['solver']} "
        f"at {fastest_total_median['total_median']:.6f}s."
    )
    print(
        f"  Fastest total single run: {fastest_total_min['library']} {fastest_total_min['solver']} "
        f"at {fastest_total_min['total_min']:.6f}s."
    )


def main() -> None:
    problem = common.PROBLEM_PARAMETERS
    tolerances = common.TOLERANCE_PARAMETERS
    reference_tolerances = common.REFERENCE_TOLERANCE_PARAMETERS
    stark_parameters = common.STARK_PARAMETERS
    initial_conditions = common.INITIAL_CONDITIONS
    repeats = common.BENCHMARK_PARAMETERS["repeats"]

    announce("Generating Robertson reference solution...")
    started = perf_counter()
    reference = scipy.run_reference(problem, reference_tolerances, initial_conditions)
    reference_elapsed = perf_counter() - started
    announce(f"Reference complete in {reference_elapsed:.6f}s.")

    entries = [
        CompetitionEntry(
            "STARK",
            "Kvaerno5 Exact Cubic",
            stark.prepare_kvaerno5_cubic,
            stark_parameters,
        ),
        CompetitionEntry(
            "STARK",
            "Kvaerno5 Newton Dense",
            stark.prepare_kvaerno5_newton_dense,
            stark_parameters,
        ),
        CompetitionEntry(
            "STARK",
            "Kvaerno5 Chord Dense",
            stark.prepare_kvaerno5_chord_dense_small,
            stark_parameters,
        ),
        CompetitionEntry(
            "STARK",
            "Kvaerno5 VeryChord Dense",
            stark.prepare_kvaerno5_very_chord_dense_small,
            stark_parameters,
        ),
        CompetitionEntry("SciPy", "Radau", scipy.prepare_radau, tolerances),
        CompetitionEntry("SciPy", "BDF", scipy.prepare_bdf, tolerances),
        CompetitionEntry("Diffrax", "Kvaerno5", diffrax.prepare_kvaerno5, tolerances, optional=True),
    ]
    rows = CompetitionRunner(
        CompetitionData(problem, initial_conditions, reference),
        entries,
        repeats,
        announce=announce,
    ).time_all()

    describe_problem(problem, tolerances, stark_parameters, reference_tolerances, reference, reference_elapsed)
    print_error_table(rows)
    print_preparation_table(rows)
    print_run_table(rows, repeats)
    print_total_table(rows, repeats)
    print_summary(rows)


if __name__ == "__main__":
    main()
