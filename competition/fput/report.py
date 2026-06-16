from __future__ import annotations

from time import perf_counter

from competition.fput import common, diffrax, scipy, stark
from competition.runner import CompetitionData, CompetitionEntry, CompetitionRunner, render_table


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
        "StarkEngineNumpy selects Numba when it is installed and otherwise uses unaccelerated callables"
    )
    print("  all compared solver stacks are prewarmed once before timed rows")
    print("  each method performs setup once, then one complete untimed warmup solve")
    print("  preparation timing excludes one-time cross-row JIT or tracing costs")
    print("  reference generation is excluded from the timing table")
    print("  timings are CPU wall-clock timings for repeated solves of one medium-sized problem")
    print("  equal Tolerance numbers do not imply identical solver error norms")
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
    initial_conditions = common.INITIAL_CONDITIONS
    repeats = common.BENCHMARK_PARAMETERS["repeats"]

    started = perf_counter()
    reference = scipy.run_reference(problem, reference_tolerances, initial_conditions)
    reference_elapsed = perf_counter() - started

    stark_problem = stark.FPUTStarkProblem(problem)
    entries = [
        CompetitionEntry("STARK", "RKCK", stark_problem.prepare_rkck, tolerances),
        CompetitionEntry("STARK", "RKDP", stark_problem.prepare_rkdp, tolerances),
        CompetitionEntry("SciPy", "RK45", scipy.prepare_rk45, tolerances),
        CompetitionEntry("SciPy", "DOP853", scipy.prepare_dop853, tolerances),
        CompetitionEntry("Diffrax", "Tsit5", diffrax.prepare_tsit5, tolerances, optional=True),
        CompetitionEntry("Diffrax", "Dopri5", diffrax.prepare_dopri5, tolerances, optional=True),
    ]
    rows = CompetitionRunner(
        CompetitionData(problem, initial_conditions, reference),
        entries,
        repeats,
    ).time_all()

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
