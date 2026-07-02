"""Reusable benchmark problem definitions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from stark import Configuration, Interval, Method, System
from stark.problem.system.system import EngineFactory, SystemIVP


BenchmarkProblemInitialFactory = Callable[[], Mapping[str, object]]
BenchmarkProblemIntervalFactory = Callable[[], Interval]
BenchmarkProblemReferenceFactory = Callable[[], Mapping[str, object]]
BenchmarkProblemError = Callable[[object, Mapping[str, object]], float]
BenchmarkProblemSystemFactory = Callable[[], System]


@dataclass(frozen=True, slots=True)
class BenchmarkProblemDefinition:
    """Reusable problem recipe for benchmarks.

    A benchmark problem owns the parts that should not be repeated across ASV
    classes: the `System`, initial values, interval, and optional final-state
    reference. It deliberately does not know about timings. Benchmark runners
    combine this object with a method stack and an engine.
    """

    name: str
    summary: str
    system_factory: BenchmarkProblemSystemFactory
    initial_factory: BenchmarkProblemInitialFactory
    interval_factory: BenchmarkProblemIntervalFactory
    reference_factory: BenchmarkProblemReferenceFactory | None = None
    final_error: BenchmarkProblemError | None = None

    def system(self) -> System:
        """Build a fresh `System` for this problem."""

        return self.system_factory()

    def initial(self) -> Mapping[str, object]:
        """Build fresh initial values for this problem."""

        return self.initial_factory()

    def interval(self) -> Interval:
        """Build a fresh interval for this problem."""

        return self.interval_factory()

    def reference(self) -> Mapping[str, object] | None:
        """Build the final-state reference when one is available."""

        if self.reference_factory is None:
            return None
        return self.reference_factory()

    def ivp(
        self,
        *,
        method: Method,
        engine: EngineFactory,
        configuration: Configuration | None = None,
    ) -> SystemIVP:
        """Build a reusable IVP for one method and engine choice."""

        return self.system().ivp(
            initial=self.initial(),
            interval=self.interval(),
            method=method,
            engine=engine,
            configuration=configuration,
        )

    def error(self, state: object) -> float | None:
        """Return final-state error when this problem has a reference."""

        reference = self.reference()
        if reference is None or self.final_error is None:
            return None
        return self.final_error(state, reference)


__all__ = [
    "BenchmarkProblemDefinition",
    "BenchmarkProblemError",
    "BenchmarkProblemInitialFactory",
    "BenchmarkProblemIntervalFactory",
    "BenchmarkProblemReferenceFactory",
    "BenchmarkProblemSystemFactory",
]

