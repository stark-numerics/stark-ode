from __future__ import annotations

from collections.abc import Iterable
from math import sqrt
from typing import Any

from stark.comparison.models import (
    ComparatorEntry,
    ComparatorProblem,
    ComparatorReport,
    Comparison,
    ComparisonResult,
)
from stark.comparison.runtime import BakedEntry, Comparer


class Comparator:
    __slots__ = ("announce", "entries", "prewarm_builders", "problem", "repeats", "comparer")

    def __init__(
        self,
        problem: ComparatorProblem,
        entries: Iterable[ComparatorEntry],
        repeats: int = 5,
        prewarm_builders: bool = True,
        announce: Any | None = None,
    ) -> None:
        self.problem = problem
        self.entries = list(entries)
        if len(self.entries) < 2:
            raise ValueError("Comparator needs at least two entries.")
        if repeats < 1:
            raise ValueError("Comparator repeats must be at least 1.")
        self.repeats = repeats
        self.prewarm_builders = prewarm_builders
        self.announce = announce
        self.comparer = Comparer(problem, repeats, announce)

    def __repr__(self) -> str:
        return (
            "Comparator("
            f"problem={self.problem.name!r}, "
            f"entries={len(self.entries)!r}, "
            f"repeats={self.repeats!r}, "
            f"prewarm_builders={self.prewarm_builders!r})"
        )

    def __str__(self) -> str:
        return f"Comparator {self.problem.name} with {len(self.entries)} entries"

    def __call__(self) -> ComparatorReport:
        if self.prewarm_builders:
            self._prewarm_builders()
        baked = [self.comparer(entry) for entry in self.entries]
        results = [
            ComparisonResult(
                name=entry.name,
                steps=row.steps,
                timing=row.timing,
                diagnostics=row.diagnostics,
                profile=row.profile,
                metadata={} if entry.metadata is None else dict(entry.metadata),
            )
            for entry, row in zip(self.entries, baked, strict=True)
        ]
        return ComparatorReport(
            problem_name=self.problem.name,
            repeats=self.repeats,
            description=self.problem.description,
            prewarmed_builders=self.prewarm_builders,
            results=results,
            final_differences=Comparison(
                labels=[result.name for result in results],
                values=self._pairwise_final_differences(baked),
            ),
            trajectory_differences=self._trajectory_comparison(results, baked),
        )

    def _prewarm_builders(self) -> None:
        self._announce("Prewarming entry builders...")
        for entry in self.entries:
            entry.make_marcher()
            if entry.build_integrator is not None:
                entry.build_integrator()

    def _pairwise_final_differences(self, baked: list[BakedEntry]) -> list[list[float]]:
        return [[self.problem.difference(left.state, right.state) for right in baked] for left in baked]

    def _pairwise_trajectory_differences(self, baked: list[BakedEntry]) -> list[list[float]] | None:
        if self.problem.checkpoints is None:
            return None
        difference = (
            self.problem.trajectory_difference
            if self.problem.trajectory_difference is not None
            else self._default_trajectory_difference
        )
        return [[difference(left.checkpoints, right.checkpoints) for right in baked] for left in baked]

    def _trajectory_comparison(self, results: list[ComparisonResult], baked: list[BakedEntry]) -> Comparison | None:
        values = self._pairwise_trajectory_differences(baked)
        if values is None:
            return None
        return Comparison(
            labels=[result.name for result in results],
            values=values,
            note=self._trajectory_difference_note(),
        )

    def _trajectory_difference_note(self) -> str | None:
        if self.problem.checkpoints is None:
            return None
        if self.problem.trajectory_difference is not None:
            return "Trajectory differences use the problem-supplied ComparatorProblem.trajectory_difference(...)."
        return (
            "Trajectory differences use the default RMS of checkpoint-wise state differences, "
            "computed from ComparatorProblem.difference(...)."
        )

    def _default_trajectory_difference(self, left: list[Any], right: list[Any]) -> float:
        if len(left) != len(right):
            raise ValueError("Comparator trajectory checkpoints must line up entry-to-entry.")
        if not left:
            return 0.0
        total = 0.0
        for left_state, right_state in zip(left, right, strict=True):
            delta = self.problem.difference(left_state, right_state)
            total += delta * delta
        return sqrt(total / len(left))

    def _announce(self, message: str) -> None:
        if self.announce is not None:
            self.announce(message)


__all__ = ["Comparator"]


