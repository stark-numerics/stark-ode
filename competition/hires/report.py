from __future__ import annotations

from time import perf_counter

from competition.hires import common, diffrax, scipy, stark
from competition.runner import CompetitionData, CompetitionEntry, CompetitionRunner, render_table


def announce(message: str) -> None:
    print(message, flush=True)


def describe_problem(problem, tolerances, stark_parameters, reference_tolerances, reference, reference_elapsed):
    print("HIRES Stiff Kinetics Benchmark")
    print()
    print("This benchmark solves the classic 8-variable HIRES stiff chemical kinetics system.")
    print(f"  interval: [{problem['t0']}, {problem['t1']}]")
    print(f"  initial state dimension: {len(common.INITIAL_CONDITIONS['y'])}")
    print(f"  comparison tolerances: rtol={tolerances['rtol']}, atol={tolerances['atol']}")
    print(
        "  STARK tolerances: "
        f"scheme=({stark_parameters['tolerance_rtol']}, {stark_parameters['tolerance_atol']}), "
        f"resolvent=({stark_parameters['resolution_rtol']}, {stark_parameters['resolution_atol']}), "
        f"inverter=({stark_parameters['inversion_rtol']}, {stark_parameters['inversion_atol']})"
    )
    print(
        f"  reference: SciPy Radau, rtol={reference_tolerances['rtol']}, "
        f"atol={reference_tolerances['atol']}, steps={reference['steps']}, "
        f"generated in {reference_elapsed:.6f}s"
    )
    print()
    print("Interpretation notes:")
    print("  HIRES is intentionally bigger than Robertson but still a small dense implicit problem")
    print("  STARK rows use the same dense inverter/nucleus path as Robertson, now at dimension 8")
    print("  there are no problem-specific exact resolvents in this report")
    print("  all compared solver stacks are prewarmed once before timed rows")
    print("  reference generation is excluded from the timing table")
    print("  equal tolerance numbers do not imply identical internal error norms")
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

    announce("Generating HIRES reference solution...")
    started = perf_counter()
    reference = scipy.run_reference(problem, reference_tolerances, initial_conditions)
    reference_elapsed = perf_counter() - started
    announce(f"Reference complete in {reference_elapsed:.6f}s.")

    entries = [
        CompetitionEntry(
            "STARK",
            "Kvaerno5 Newton Dense",
            stark.prepare_kvaerno5_newton_dense,
            stark_parameters,
        ),
        CompetitionEntry(
            "STARK",
            "Kvaerno5 Chord Dense",
            stark.prepare_kvaerno5_chord_dense,
            stark_parameters,
        ),
        CompetitionEntry(
            "STARK",
            "Kvaerno5 VeryChord Dense",
            stark.prepare_kvaerno5_very_chord_dense,
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
