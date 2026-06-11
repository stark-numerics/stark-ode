from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import median
from time import perf_counter
from typing import Any


SolveOnce = Callable[[], Mapping[str, Any]]
PrepareSolve = Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], SolveOnce]
Announce = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class CompetitionData:
    """
    Shared inputs supplied to every solver row in a competition report.

    The reference solution is generated before timed solver rows. It is shared
    with every entry so that all solvers are judged against the same final
    state. This object deliberately does not contain per-entry tolerances or
    solver options; those live on `CompetitionEntry` so each row declares the
    parameters it is actually timed with.
    """

    problem: Mapping[str, Any]
    initial_conditions: Mapping[str, Any]
    reference: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CompetitionEntry:
    """
    One solver row in a competition report.

    `prepare` must build and return a zero-argument `solve_once` callable using
    the entry parameters and shared `CompetitionData`. The competition runner
    times that preparation as row setup, then times one complete warmup solve,
    then times repeated calls to the same prepared solve. Entry construction is
    not timed, which lets reports assemble clear row declarations without
    charging one library for report bookkeeping.

    Optional entries are for locally unavailable stacks such as Diffrax/JAX.
    Optional failures are reported as table rows with a note; required entry
    failures still raise.
    """

    library: str
    solver: str
    prepare: PrepareSolve
    parameters: Mapping[str, Any]
    optional: bool = False

    def prepare_solve(self, data: CompetitionData) -> SolveOnce:
        return self.prepare(
            data.problem,
            self.parameters,
            data.initial_conditions,
            data.reference,
        )


class CompetitionRunner:
    """
    Fair timing harness for competition report entries.

    The timing policy is intentionally explicit and identical for every entry:

    1. Optional prewarming can run before timed rows and is excluded from all
       timing tables. This absorbs cross-row one-time costs such as Numba
       compilation or JAX tracing when a report chooses to prewarm.
    2. Timed setup measures one call to `CompetitionEntry.prepare_solve(...)`.
    3. Timed warmup measures one complete call to the prepared solver.
    4. Timed runs measure repeated calls to the same prepared solver.

    The prepared callable must return a mapping containing at least `library`,
    `solver`, `error`, and `steps`. Extra keys, such as inverter diagnostics,
    are preserved on the timed row.
    """

    __slots__ = ("announce", "data", "entries", "repeats")

    def __init__(
        self,
        data: CompetitionData,
        entries: Sequence[CompetitionEntry],
        repeats: int,
        *,
        announce: Announce | None = None,
    ) -> None:
        if repeats < 1:
            raise ValueError("CompetitionRunner repeats must be at least 1.")
        self.data = data
        self.entries = tuple(entries)
        self.repeats = repeats
        self.announce = announce

    def prewarm_all(self) -> None:
        """
        Run one untimed preparation and solve for each entry.

        This phase is deliberately outside the timing table. It is used by the
        reports to avoid charging the first timed row for one-time compilation,
        tracing, or cache setup that later rows from the same backend reuse.
        """

        for entry in self.entries:
            self.prewarm(entry)

    def prewarm(self, entry: CompetitionEntry) -> None:
        if self.announce is not None:
            self.announce(f"Prewarming {entry.library} {entry.solver}...")
        try:
            solve_once = entry.prepare_solve(self.data)
            result = solve_once()
        except Exception:
            if not entry.optional:
                raise
            if self.announce is not None:
                self.announce(f"Prewarm skipped: {entry.library} {entry.solver} is unavailable.")
            return
        if self.announce is not None:
            self.announce(
                f"Prewarm complete: {result['solver']} "
                f"steps={result['steps']}, error={result['error']:.6e}"
            )

    def time_all(self, *, prewarm: bool = True) -> list[dict[str, Any]]:
        """
        Return timed rows for every entry.

        When `prewarm` is true, all entries are prewarmed before the first timed
        row. Timed rows still include their own setup and warmup measurements,
        so the preparation table remains comparable across libraries.
        """

        if prewarm:
            self.prewarm_all()
        rows = []
        for entry in self.entries:
            if self.announce is not None:
                self.announce(f"Timing {entry.library} {entry.solver}...")
            rows.append(self.time(entry))
        return rows

    def time(self, entry: CompetitionEntry) -> dict[str, Any]:
        """
        Time setup, warmup, and repeated solves for one entry.

        Required entry failures raise immediately. Optional entry failures are
        returned as incomplete rows with a human-readable note so the report
        shows exactly which local dependency was unavailable.
        """

        setup_elapsed = None
        warmup_elapsed = None
        try:
            started = perf_counter()
            solve_once = entry.prepare_solve(self.data)
            setup_elapsed = perf_counter() - started

            started = perf_counter()
            solve_once()
            warmup_elapsed = perf_counter() - started

            durations = []
            result = None
            for _repeat in range(self.repeats):
                started = perf_counter()
                result = solve_once()
                durations.append(perf_counter() - started)
        except Exception as exc:
            if not entry.optional:
                raise
            return self._optional_failure_row(entry, setup_elapsed, warmup_elapsed, exc)

        if result is None:
            raise RuntimeError("Competition entry produced no timed result.")
        return self._timed_row(result, setup_elapsed, warmup_elapsed, durations)

    def _timed_row(
        self,
        result: Mapping[str, Any],
        setup_elapsed: float,
        warmup_elapsed: float,
        durations: Sequence[float],
    ) -> dict[str, Any]:
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
            "note": "",
        }
        for key, value in result.items():
            if key not in row:
                row[key] = value
        return row

    def _optional_failure_row(
        self,
        entry: CompetitionEntry,
        setup_elapsed: float | None,
        warmup_elapsed: float | None,
        exc: Exception,
    ) -> dict[str, Any]:
        return {
            "library": entry.library,
            "solver": entry.solver,
            "error": None,
            "steps": None,
            "setup": setup_elapsed,
            "warmup": warmup_elapsed,
            "preparation": None if setup_elapsed is None else setup_elapsed + (warmup_elapsed or 0.0),
            "median": None,
            "min": None,
            "note": f"{type(exc).__name__}: {exc}",
        }


def render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """
    Render a small fixed-width text table for competition reports.

    The reports build plain strings before calling this helper so formatting
    decisions remain visible at each table call site.
    """

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
