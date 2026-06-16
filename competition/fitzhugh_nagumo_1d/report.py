from __future__ import annotations

from time import perf_counter

from competition.fitzhugh_nagumo_1d import common, diffrax, scipy, stark
from competition.runner import CompetitionData, CompetitionEntry, CompetitionRunner, render_table


def announce(message: str) -> None:
    print(message, flush=True)


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
        "  STARK adaptive Tolerance: "
        f"rtol={stark_parameters['tolerance_rtol']:.0e}, atol={stark_parameters['tolerance_atol']:.0e}"
    )
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
    print("  STARK compares the current best generic local combination with an IMEX spectral custom resolvent")
    print("  STARK generic row: Kvaerno3 with Anderson acceleration")
    print("  STARK IMEX split: implicit diffusion in u, explicit reaction and recovery terms")
    print("  STARK IMEX custom resolvent: exact spectral solve of the periodic discrete diffusion stage")
    print("  Diffrax uses Kvaerno5, an adaptive stiffly accurate ESDIRK method")
    print("  all compared solver stacks are prewarmed once before timed rows")
    print("  each method performs setup once, then one complete untimed warmup solve")
    print("  preparation timing excludes one-time cross-row JIT or tracing costs")
    print("  reference generation is excluded from the timing table")
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

    announce("Generating SciPy Radau reference solution...")
    started = perf_counter()
    reference = scipy.run_reference(problem, reference_tolerances, initial_conditions)
    reference_elapsed = perf_counter() - started
    announce(
        f"Reference ready: steps={reference['steps']}, "
        f"elapsed={reference_elapsed:.3f}s"
    )

    entries = [
        CompetitionEntry("STARK", "Kvaerno3 Anderson", stark.prepare_kvaerno3_anderson, stark_parameters),
        CompetitionEntry("STARK", "KC43_7 IMEX Spectral", stark.prepare_kc43_imex_spectral, stark_parameters),
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
    print_total_table(rows, repeats)
    print_summary(rows)


if __name__ == "__main__":
    main()
