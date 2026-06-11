from __future__ import annotations

from time import perf_counter

from competition.robertson import common, diffrax, scipy, stark
from competition.runner import CompetitionData, CompetitionEntry, CompetitionRunner, render_table


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
    print("  STARK Robertson includes a fully implicit Kvaerno4 solve with a custom exact cubic resolvent")
    print("  STARK Robertson also includes Kvaerno4 Newton rows using the new inverter request path")
    print("  one Newton row uses Jacobi relaxation with a materialized local 3x3 entry inverse")
    print("  one Newton row uses the generic dense inverter with the native dense provider")
    print("  the Newton/Jacobi relaxation update is supplied through the allocator BlockSpecialist")
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


def print_inverter_diagnostics(rows) -> None:
    diagnostic_rows = [row for row in rows if "inverter_solve_count" in row]
    if not diagnostic_rows:
        return

    print("Inverter Diagnostics Table")
    print(
        render_table(
            (
                "solver",
                "solves",
                "failures",
                "iter min",
                "iter med",
                "iter max",
                "initial med",
                "final med",
            ),
            [
                (
                    row["solver"],
                    str(row["inverter_solve_count"]),
                    str(row["inverter_failure_count"]),
                    _format_optional_iteration(row["inverter_iteration_min"]),
                    _format_optional_iteration(row["inverter_iteration_median"]),
                    _format_optional_iteration(row["inverter_iteration_max"]),
                    _format_optional_float(row["inverter_initial_residual_median"]),
                    _format_optional_float(row["inverter_final_residual_median"]),
                )
                for row in diagnostic_rows
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
    initial_conditions = common.INITIAL_CONDITIONS
    repeats = common.BENCHMARK_PARAMETERS["repeats"]

    started = perf_counter()
    reference = scipy.run_reference(problem, reference_tolerances, initial_conditions)
    reference_elapsed = perf_counter() - started

    entries = [
        CompetitionEntry("STARK", "Kvaerno4 Full Cubic", stark.prepare_kvaerno4_full_custom, stark_parameters),
        CompetitionEntry("STARK", "Kvaerno4 Full Newton Jacobi", stark.prepare_kvaerno4_full_newton, stark_parameters),
        CompetitionEntry("STARK", "Kvaerno4 Full Newton Dense", stark.prepare_kvaerno4_full_newton_dense, stark_parameters),
        CompetitionEntry("SciPy", "Radau", scipy.prepare_radau, tolerances),
        CompetitionEntry("SciPy", "BDF", scipy.prepare_bdf, tolerances),
        CompetitionEntry("Diffrax", "Kvaerno5", diffrax.prepare_kvaerno5, tolerances, optional=True),
    ]
    rows = CompetitionRunner(
        CompetitionData(problem, initial_conditions, reference),
        entries,
        repeats,
    ).time_all()

    describe_problem(problem, tolerances, stark_parameters, reference_tolerances, reference, reference_elapsed)
    print_error_table(rows)
    print_preparation_table(rows)
    print_run_table(rows, repeats)
    print_inverter_diagnostics(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
