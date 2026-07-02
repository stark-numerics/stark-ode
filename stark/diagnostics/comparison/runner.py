"""High-level runner for fair, repeatable comparison reports."""

from __future__ import annotations

from collections.abc import Iterable
from math import sqrt
from typing import Any

from stark.diagnostics.comparison.models import (
    ComparisonEntry,
    ComparisonProblem,
    ComparisonReport,
    Comparison,
    ComparisonResult,
)
from stark.diagnostics.comparison.runtime import ComparisonEntryEvaluation, ComparisonEntryEvaluator


class ComparisonRunner:
    """Run a problem through multiple entries and return a structured report."""

    __slots__ = ("announce", "entries", "entry_evaluator", "prewarm_builders", "problem", "repeats")

    def __init__(
        self,
        problem: ComparisonProblem,
        entries: Iterable[ComparisonEntry],
        repeats: int = 5,
        prewarm_builders: bool = True,
        announce: Any | None = None,
    ) -> None:
        self.problem = problem
        self.entries = list(entries)
        if len(self.entries) < 2:
            raise ValueError("ComparisonRunner needs at least two entries.")
        if repeats < 1:
            raise ValueError("ComparisonRunner repeats must be at least 1.")
        self.repeats = repeats
        self.prewarm_builders = prewarm_builders
        self.announce = announce
        self.entry_evaluator = ComparisonEntryEvaluator(problem, repeats, announce)

    def __repr__(self) -> str:
        return (
            "ComparisonRunner("
            f"problem={self.problem.name!r}, "
            f"entries={len(self.entries)!r}, "
            f"repeats={self.repeats!r}, "
            f"prewarm_builders={self.prewarm_builders!r})"
        )

    def __str__(self) -> str:
        return f"ComparisonRunner {self.problem.name} with {len(self.entries)} entries"

    def __call__(self) -> ComparisonReport:
        if self.prewarm_builders:
            self.prewarm_entry_builders()
        baked = [self.entry_evaluator(entry) for entry in self.entries]
        results = [
            ComparisonResult(
                name=entry.name,
                steps=row.steps,
                timing=row.timing,
                diagnostics=row.diagnostics,
                profile=row.profile,
                metadata={} if entry.metadata is None else dict(entry.metadata),
                monitor_summary=row.monitor_summary,
            )
            for entry, row in zip(self.entries, baked, strict=True)
        ]
        return ComparisonReport(
            problem_name=self.problem.name,
            repeats=self.repeats,
            description=self.problem.description,
            prewarmed_builders=self.prewarm_builders,
            results=results,
            final_differences=Comparison(
                labels=[result.name for result in results],
                values=self.pairwise_final_differences(baked),
            ),
            trajectory_differences=self.trajectory_comparison(results, baked),
        )

    def prewarm_entry_builders(self) -> None:
        self.announce_message("Prewarming entry builders...")
        for entry in self.entries:
            entry.make_stepper(self.problem.ivp)
            entry.make_integrator(self.problem.ivp)

    def pairwise_final_differences(self, baked: list[ComparisonEntryEvaluation]) -> list[list[float]]:
        return [[self.problem.difference(left.state, right.state) for right in baked] for left in baked]

    def pairwise_trajectory_differences(self, baked: list[ComparisonEntryEvaluation]) -> list[list[float]] | None:
        if self.problem.checkpoints is None:
            return None
        difference = (
            self.problem.trajectory_difference
            if self.problem.trajectory_difference is not None
            else self.default_trajectory_difference
        )
        return [[difference(left.checkpoints, right.checkpoints) for right in baked] for left in baked]

    def trajectory_comparison(self, results: list[ComparisonResult], baked: list[ComparisonEntryEvaluation]) -> Comparison | None:
        values = self.pairwise_trajectory_differences(baked)
        if values is None:
            return None
        return Comparison(
            labels=[result.name for result in results],
            values=values,
            note=self.trajectory_difference_note(),
        )

    def trajectory_difference_note(self) -> str | None:
        if self.problem.checkpoints is None:
            return None
        if self.problem.trajectory_difference is not None:
            return "Trajectory differences use the problem-supplied ComparisonProblem.trajectory_difference(...)."
        return (
            "Trajectory differences use the default RMS of checkpoint-wise state differences, "
            "computed from ComparisonProblem.difference(...)."
        )

    def default_trajectory_difference(self, left: list[Any], right: list[Any]) -> float:
        if len(left) != len(right):
            raise ValueError("ComparisonRunner trajectory checkpoints must line up entry-to-entry.")
        if not left:
            return 0.0
        total = 0.0
        for left_state, right_state in zip(left, right, strict=True):
            delta = self.problem.difference(left_state, right_state)
            total += delta * delta
        return sqrt(total / len(left))

    def announce_message(self, message: str) -> None:
        if self.announce is not None:
            self.announce(message)


__all__ = ["ComparisonRunner"]


