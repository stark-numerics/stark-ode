from __future__ import annotations

from time import perf_counter

from competition.allen_cahn_1d import common, diffrax, scipy, stark
from competition.runner import CompetitionData, CompetitionEntry, CompetitionRunner, render_table


def announce(message: str) -> None:
    print(message, flush=True)


def describe_problem(problem, tolerances, stark_parameters, reference_tolerances, reference, reference_elapsed):
    print("Allen-Cahn 1D Krylov Benchmark")
    print()
    print("This benchmark solves the periodic one-dimensional Allen-Cahn equation:")
    print("  u_t = D u_xx + u - u^3")
    print()
    print("Problem parameters:")
    print(f"  grid size: {problem['grid_size']}")
    print(f"  domain length: {problem['length']}")
    print(f"  time interval: [{problem['t0']}, {problem['t1']}]")
    print(f"  diffusivity: {problem['diffusivity']}")
    print()
    print("Reference solution:")
    print("  generated with SciPy solve_ivp using Radau and a sparse periodic Jacobian")
    print(f"  rtol={reference_tolerances['rtol']:.0e}, atol={reference_tolerances['atol']:.0e}")
    print(f"  reference steps: {reference['steps']}")
    print(f"  reference generation time: {reference_elapsed:.6f}s")
    print()
    print("Compared runs:")
    print(f"  SciPy/Diffrax tolerances: rtol={tolerances['rtol']:.0e}, atol={tolerances['atol']:.0e}")
    print(
        "  STARK adaptive Tolerance: "
        f"rtol={stark_parameters['tolerance_rtol']:.0e}, atol={stark_parameters['tolerance_atol']:.0e}"
    )
    print(
        "  STARK Newton/Krylov tolerances: "
        f"resolvent=({stark_parameters['resolution_atol']:.0e}, {stark_parameters['resolution_rtol']:.0e}), "
        f"inverter=({stark_parameters['inversion_atol']:.0e}, {stark_parameters['inversion_rtol']:.0e}), "
        f"restart={stark_parameters['inversion_restart']}"
    )
    print("  STARK row is matrix-free; SciPy rows use sparse Jacobians.")
    print("  Diffrax is optional and shows the JAX/JIT comparison when installed.")
    print("  all compared solver stacks are prewarmed once before timed rows")
    print("  warm run timings exclude setup and warmup; total timings include both")
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
    fastest_warm = min(completed, key=lambda row: row["median"])
    fastest_total = min(completed, key=lambda row: row["total_median"])

    print("Summary:")
    print(
        f"  Best accuracy: {most_accurate['library']} {most_accurate['solver']} "
        f"with error {most_accurate['error']:.6e}."
    )
    print(
        f"  Fastest warm median time: {fastest_warm['library']} {fastest_warm['solver']} "
        f"at {fastest_warm['median']:.6f}s."
    )
    print(
        f"  Fastest total median time: {fastest_total['library']} {fastest_total['solver']} "
        f"at {fastest_total['total_median']:.6f}s."
    )


def main() -> None:
    problem = common.PROBLEM_PARAMETERS
    tolerances = common.TOLERANCE_PARAMETERS
    stark_parameters = common.STARK_PARAMETERS
    reference_tolerances = common.REFERENCE_TOLERANCE_PARAMETERS
    initial_conditions = common.INITIAL_CONDITIONS
    repeats = common.BENCHMARK_PARAMETERS["repeats"]

    announce("Generating SciPy Radau reference solution...")
    started = perf_counter()
    reference = scipy.run_reference(problem, reference_tolerances, initial_conditions)
    reference_elapsed = perf_counter() - started
    announce(f"Reference ready: steps={reference['steps']}, elapsed={reference_elapsed:.3f}s")

    entries = [
        CompetitionEntry("STARK", "SDIRK21 Newton Krylov", stark.prepare_sdirk21_newton_krylov, stark_parameters),
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
