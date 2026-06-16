from __future__ import annotations

import argparse
from statistics import median
from time import perf_counter

from competition.robertson import common, scipy, stark
from competition.runner import render_table


def prepare_solvers(reference):
    problem = common.PROBLEM_PARAMETERS
    parameters = common.STARK_PARAMETERS
    initial = common.INITIAL_CONDITIONS
    return {
        "Newton Dense Small": stark.prepare_kvaerno5_newton_dense_small(
            problem,
            parameters,
            initial,
            reference,
        ),
        "Chord Dense Small": stark.prepare_kvaerno5_chord_dense_small(
            problem,
            parameters,
            initial,
            reference,
        ),
        "VeryChord Dense Small": stark.prepare_kvaerno5_very_chord_dense_small(
            problem,
            parameters,
            initial,
            reference,
        ),
    }


def timed_solve(solve_once):
    started = perf_counter()
    row = solve_once()
    elapsed = perf_counter() - started
    return elapsed, row


def format_seconds(value: float) -> str:
    return f"{value:.6f}s"


def run_warmups(solvers, warmups: int) -> None:
    for _index in range(warmups):
        for solve_once in solvers.values():
            solve_once()


def run_timing_table(solvers, repeats: int):
    rows = []
    for run_index in range(1, repeats + 1):
        names = tuple(solvers)
        if run_index % 2 == 0:
            names = tuple(reversed(names))

        timings = {}
        results = {}
        for name in names:
            elapsed, row = timed_solve(solvers[name])
            timings[name] = elapsed
            results[name] = row

        newton = results["Newton Dense Small"]
        chord = results["Chord Dense Small"]
        very_chord = results["VeryChord Dense Small"]
        rows.append(
            {
                "run": run_index,
                "order": " -> ".join(names),
                "newton_time": timings["Newton Dense Small"],
                "newton_steps": newton["steps"],
                "newton_error": newton["error"],
                "chord_time": timings["Chord Dense Small"],
                "chord_steps": chord["steps"],
                "chord_error": chord["error"],
                "very_chord_time": timings["VeryChord Dense Small"],
                "very_chord_steps": very_chord["steps"],
                "very_chord_error": very_chord["error"],
            }
        )
    return rows


def print_timing_table(rows) -> None:
    print("Robertson Kvaerno5 Dense Timing Drift")
    print(
        render_table(
            (
                "run",
                "order",
                "Newton time",
                "Newton steps",
                "Newton error",
                "Chord time",
                "Chord steps",
                "Chord error",
                "VeryChord time",
                "VeryChord steps",
                "VeryChord error",
            ),
            [
                (
                    str(row["run"]),
                    row["order"],
                    format_seconds(row["newton_time"]),
                    str(row["newton_steps"]),
                    f"{row['newton_error']:.6e}",
                    format_seconds(row["chord_time"]),
                    str(row["chord_steps"]),
                    f"{row['chord_error']:.6e}",
                    format_seconds(row["very_chord_time"]),
                    str(row["very_chord_steps"]),
                    f"{row['very_chord_error']:.6e}",
                )
                for row in rows
            ],
        )
    )


def print_summary(rows) -> None:
    newton_times = [row["newton_time"] for row in rows]
    chord_times = [row["chord_time"] for row in rows]
    very_chord_times = [row["very_chord_time"] for row in rows]
    print()
    print("Summary")
    print(
        render_table(
            ("solver", "first", "last", "min", "median", "max", "last / first"),
            [
                (
                    "Newton Dense Small",
                    format_seconds(newton_times[0]),
                    format_seconds(newton_times[-1]),
                    format_seconds(min(newton_times)),
                    format_seconds(median(newton_times)),
                    format_seconds(max(newton_times)),
                    f"{newton_times[-1] / newton_times[0]:.3f}x",
                ),
                (
                    "Chord Dense Small",
                    format_seconds(chord_times[0]),
                    format_seconds(chord_times[-1]),
                    format_seconds(min(chord_times)),
                    format_seconds(median(chord_times)),
                    format_seconds(max(chord_times)),
                    f"{chord_times[-1] / chord_times[0]:.3f}x",
                ),
                (
                    "VeryChord Dense Small",
                    format_seconds(very_chord_times[0]),
                    format_seconds(very_chord_times[-1]),
                    format_seconds(min(very_chord_times)),
                    format_seconds(median(very_chord_times)),
                    format_seconds(max(very_chord_times)),
                    f"{very_chord_times[-1] / very_chord_times[0]:.3f}x",
                ),
            ],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run many Robertson Kvaerno5 Newton/Chord dense-small solves and "
            "print one timing row per repeat. This is a drift probe, not a "
            "competition report."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=50,
        help="Number of timed solve pairs.",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="Untimed solves for each solver before timing starts.",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative.")

    print("Generating Robertson reference solution...", flush=True)
    reference = scipy.run_reference(
        common.PROBLEM_PARAMETERS,
        common.REFERENCE_TOLERANCE_PARAMETERS,
        common.INITIAL_CONDITIONS,
    )
    print("Preparing STARK solvers...", flush=True)
    solvers = prepare_solvers(reference)
    if args.warmups:
        print(f"Running {args.warmups} warmup solve(s) per solver...", flush=True)
        run_warmups(solvers, args.warmups)
    print(f"Timing {args.repeats} solve pair(s)...", flush=True)
    rows = run_timing_table(solvers, args.repeats)
    print()
    print_timing_table(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
