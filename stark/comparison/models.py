from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any


Checkpoints = int | Iterable[float]
StateBuilder = Callable[[], Any]
IntervalBuilder = Callable[[], Any]
MarcherBuilder = Callable[[], Any]
Difference = Callable[[Any, Any], float]
Diagnostics = Callable[[Any], Any]
TrajectoryDifference = Callable[[list[Any], list[Any]], float]
ProfileCategory = Callable[[str, int, str], str | None]


def _normalize_marcher_builder(source: Any) -> MarcherBuilder:
    if _accepts_zero_arguments(source):
        return source
    return lambda: source


def _accepts_zero_arguments(candidate: Any) -> bool:
    if not callable(candidate):
        return False

    try:
        signature = inspect.signature(candidate)
    except (TypeError, ValueError):
        return False

    for parameter in signature.parameters.values():
        if parameter.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}:
            if parameter.default is inspect.Parameter.empty:
                return False
    return True


@dataclass(slots=True)
class ComparatorProblem:
    name: str
    build_state: StateBuilder
    build_interval: IntervalBuilder
    difference: Difference
    diagnostics: Diagnostics | None = None
    description: str | None = None
    checkpoints: Checkpoints | None = None
    trajectory_difference: TrajectoryDifference | None = None


@dataclass(slots=True, init=False)
class ComparatorEntry:
    name: str
    build_marcher: MarcherBuilder = field(repr=False)
    build_integrator: Callable[[], Any] | None = None
    profile_category: ProfileCategory | None = None
    metadata: dict[str, Any] | None = None

    def __init__(
        self,
        name: str,
        marcher: Any,
        build_integrator: Callable[[], Any] | None = None,
        profile_category: ProfileCategory | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.build_marcher = _normalize_marcher_builder(marcher)
        self.build_integrator = build_integrator
        self.profile_category = profile_category
        self.metadata = metadata

    def make_marcher(self) -> Any:
        source = self.build_marcher
        if _accepts_zero_arguments(source):
            return source()
        return source


@dataclass(slots=True)
class ComparisonBreakdown:
    profiled: float
    problem: float
    method: float
    resolvent: float
    inverter: float
    framework: float
    other: float

    def __str__(self) -> str:
        return (
            f"profiled={self.profiled:.6f}s, "
            f"problem={self.problem:.6f}s, "
            f"method={self.method:.6f}s, "
            f"resolvent={self.resolvent:.6f}s, "
            f"inverter={self.inverter:.6f}s, "
            f"framework={self.framework:.6f}s, "
            f"other={self.other:.6f}s"
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "profiled": self.profiled,
            "problem": self.problem,
            "method": self.method,
            "resolvent": self.resolvent,
            "inverter": self.inverter,
            "framework": self.framework,
            "other": self.other,
        }

    @property
    def scheme(self) -> float:
        return self.method

    @property
    def resolver(self) -> float:
        return self.resolvent


@dataclass(slots=True)
class ComparisonHotspot:
    location: str
    self_time: float
    cumulative_time: float
    calls: int

    def __str__(self) -> str:
        return (
            f"{self.location}: self={self.self_time:.6f}s, "
            f"cumulative={self.cumulative_time:.6f}s, calls={self.calls}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "location": self.location,
            "self_time": self.self_time,
            "cumulative_time": self.cumulative_time,
            "calls": self.calls,
        }


@dataclass(slots=True)
class ComparisonTiming:
    setup: float
    warmup: float
    median: float
    minimum: float

    def __str__(self) -> str:
        return (
            f"setup={self.setup:.6f}s, "
            f"warmup={self.warmup:.6f}s, "
            f"median={self.median:.6f}s, "
            f"min={self.minimum:.6f}s"
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "setup": self.setup,
            "warmup": self.warmup,
            "median": self.median,
            "minimum": self.minimum,
        }


@dataclass(slots=True)
class ComparisonDiagnostics:
    values: dict[str, Any]

    def __bool__(self) -> bool:
        return bool(self.values)

    def __iter__(self):
        return iter(self.values.items())

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)

    def names(self) -> list[str]:
        return list(self.values)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.values)

    def __str__(self) -> str:
        from stark.comparison.writers import ComparisonDiagnosticsWriter

        return ComparisonDiagnosticsWriter()(self)

    def __repr__(self) -> str:
        return f"ComparisonDiagnostics(names={list(self.values)!r})"

    @classmethod
    def coerce(cls, diagnostics: "ComparisonDiagnostics | dict[str, Any] | None") -> "ComparisonDiagnostics":
        if diagnostics is None:
            return cls({})
        if isinstance(diagnostics, cls):
            return diagnostics
        return cls(dict(diagnostics))


@dataclass(slots=True)
class Comparison:
    labels: list[str]
    values: list[list[float]]
    note: str | None = None

    def rows(self) -> list[tuple[str, list[float]]]:
        return list(zip(self.labels, self.values, strict=True))

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {
            row_name: {column_name: value for column_name, value in zip(self.labels, row_values, strict=True)}
            for row_name, row_values in self.rows()
        }

    def __str__(self) -> str:
        from stark.comparison.writers import ComparisonWriter

        return ComparisonWriter()(self)

    def __repr__(self) -> str:
        return f"Comparison(labels={self.labels!r})"


@dataclass(slots=True)
class ComparisonProfile:
    breakdown: ComparisonBreakdown
    note: str | None
    custom_hotspots: list[ComparisonHotspot]

    def __str__(self) -> str:
        from stark.comparison.writers import ComparisonProfileWriter

        return ComparisonProfileWriter()(self)

    def as_dict(self) -> dict[str, Any]:
        return {
            "breakdown": self.breakdown.as_dict(),
            "note": self.note,
            "custom_hotspots": [hotspot.as_dict() for hotspot in self.custom_hotspots],
        }


@dataclass(slots=True)
class ComparisonResult:
    name: str
    steps: int
    timing: ComparisonTiming
    diagnostics: ComparisonDiagnostics
    profile: ComparisonProfile
    metadata: dict[str, Any]

    def __str__(self) -> str:
        from stark.comparison.writers import ComparisonResultWriter

        return ComparisonResultWriter()(self)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "steps": self.steps,
            "timing": self.timing.as_dict(),
            "diagnostics": self.diagnostics.as_dict(),
            "profile": self.profile.as_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ComparatorReport:
    problem_name: str
    repeats: int
    description: str | None
    prewarmed_builders: bool
    results: list[ComparisonResult]
    final_differences: Comparison
    trajectory_differences: Comparison | None

    def __str__(self) -> str:
        return self.render()

    def render(self) -> str:
        from stark.comparison.writers import ComparatorReportWriter

        return ComparatorReportWriter()(self)

    def results_by_name(self) -> dict[str, ComparisonResult]:
        return {result.name: result for result in self.results}

    def timings_by_name(self) -> dict[str, ComparisonTiming]:
        return {result.name: result.timing for result in self.results}

    def diagnostics_by_name(self) -> dict[str, dict[str, Any]]:
        return {result.name: result.diagnostics.as_dict() for result in self.results}

    def profiles_by_name(self) -> dict[str, ComparisonProfile]:
        return {result.name: result.profile for result in self.results}

    def metadata_by_name(self) -> dict[str, dict[str, Any]]:
        return {result.name: dict(result.metadata) for result in self.results}

    def final_difference_map(self) -> dict[str, dict[str, float]]:
        return self.final_differences.as_dict()

    def trajectory_difference_map(self) -> dict[str, dict[str, float]] | None:
        if self.trajectory_differences is None:
            return None
        return self.trajectory_differences.as_dict()

    def as_dict(self) -> dict[str, Any]:
        return {
            "problem_name": self.problem_name,
            "repeats": self.repeats,
            "description": self.description,
            "prewarmed_builders": self.prewarmed_builders,
            "results": [result.as_dict() for result in self.results],
            "final_differences": self.final_differences.as_dict(),
            "trajectory_differences": None if self.trajectory_differences is None else self.trajectory_differences.as_dict(),
        }


__all__ = [
    "ComparatorEntry",
    "ComparatorProblem",
    "ComparatorReport",
    "Comparison",
    "ComparisonBreakdown",
    "ComparisonDiagnostics",
    "ComparisonHotspot",
    "ComparisonProfile",
    "ComparisonResult",
    "ComparisonTiming",
    "Difference",
    "Diagnostics",
    "IntervalBuilder",
    "MarcherBuilder",
    "ProfileCategory",
    "StateBuilder",
    "TrajectoryDifference",
]


