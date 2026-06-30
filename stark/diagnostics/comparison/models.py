"""Data models used by the comparison diagnostics runner.

The comparison layer is deliberately more structured than a timing script. It
records how each entry was prepared, timed, diagnosed, and profiled so reports
can show whether a comparison is fair rather than only printing a stopwatch
result.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from math import sqrt
from typing import Any, SupportsFloat, cast

from stark.diagnostics.monitor import MonitorSummary


Checkpoints = int | Iterable[float]
StateBuilder = Callable[[], Any]
IntervalBuilder = Callable[[], Any]
StepperBuilder = Callable[[], Any]
ObservedStepperBuilder = Callable[[Any], Any]
Difference = Callable[[Any, Any], float]
Diagnostics = Callable[[Any], Any]
TrajectoryDifference = Callable[[list[Any], list[Any]], float]
ProfileCategory = Callable[[str, int, str], str | None]


def _normalize_stepper_builder(source: Any) -> StepperBuilder:
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


@dataclass(slots=True, init=False)
class ComparisonProblem:
    """Comparison problem built from a prepared `SystemIVP`."""

    name: str
    ivp: Any
    build_state: StateBuilder
    build_interval: IntervalBuilder
    difference: Difference
    diagnostics: Diagnostics | None = None
    description: str | None = None
    checkpoints: Checkpoints | None = None
    trajectory_difference: TrajectoryDifference | None = None

    def __init__(
        self,
        name: str,
        ivp: Any,
        *,
        difference: Difference | None = None,
        diagnostics: Diagnostics | None = None,
        description: str | None = None,
        checkpoints: Checkpoints | None = None,
        trajectory_difference: TrajectoryDifference | None = None,
    ) -> None:
        self.name = name
        self.ivp = ivp
        self.build_state = lambda: _copy_ivp_initial(ivp)
        self.build_interval = lambda: _copy_ivp_interval(ivp)
        self.difference = difference if difference is not None else _ivp_state_difference(ivp)
        self.diagnostics = diagnostics
        self.description = description
        self.checkpoints = checkpoints
        self.trajectory_difference = trajectory_difference


@dataclass(slots=True, init=False)
class ComparisonProblemManual:
    """Low-level comparison problem with user-supplied state and interval builders."""

    name: str
    ivp: Any | None
    build_state: StateBuilder
    build_interval: IntervalBuilder
    difference: Difference
    diagnostics: Diagnostics | None = None
    description: str | None = None
    checkpoints: Checkpoints | None = None
    trajectory_difference: TrajectoryDifference | None = None

    def __init__(
        self,
        name: str,
        *,
        build_state: StateBuilder,
        build_interval: IntervalBuilder,
        difference: Difference,
        diagnostics: Diagnostics | None = None,
        description: str | None = None,
        checkpoints: Checkpoints | None = None,
        trajectory_difference: TrajectoryDifference | None = None,
    ) -> None:
        self.name = name
        self.ivp = None
        self.build_state = build_state
        self.build_interval = build_interval
        self.difference = difference
        self.diagnostics = diagnostics
        self.description = description
        self.checkpoints = checkpoints
        self.trajectory_difference = trajectory_difference


@dataclass(slots=True, init=False)
class ComparisonEntry:
    """Comparison entry selected by a `Method` for the problem IVP."""

    name: str
    build_stepper: Callable[[Any], Any] = field(repr=False)
    build_observed_stepper: Callable[[Any, Any], Any] | None = field(default=None, repr=False)
    build_integrator: Callable[[Any], Any] | None = None
    profile_category: ProfileCategory | None = None
    metadata: dict[str, Any] | None = None

    def __init__(
        self,
        name: str,
        method: Any,
        *,
        configuration: Any | None = None,
        profile_category: ProfileCategory | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.build_stepper = lambda ivp: _ivp_method_stepper(ivp, method, configuration=configuration)
        self.build_observed_stepper = lambda monitor, ivp: _ivp_method_stepper(
            ivp,
            _method_with_scheme_monitor(method, monitor),
            configuration=configuration,
        )
        self.build_integrator = lambda ivp: ivp.integrator
        self.profile_category = profile_category
        self.metadata = metadata

    def make_stepper(self, ivp: Any | None = None) -> Any:
        if ivp is None:
            raise TypeError("ComparisonEntry requires a ComparisonProblem built from a SystemIVP.")
        return self.build_stepper(ivp)

    def make_observed_stepper(self, monitor: Any, ivp: Any | None = None) -> Any:
        if ivp is None:
            raise TypeError("ComparisonEntry requires a ComparisonProblem built from a SystemIVP.")
        source = self.build_observed_stepper
        if source is None:
            return self.make_stepper(ivp)
        return source(monitor, ivp)

    def make_integrator(self, ivp: Any | None = None) -> Any | None:
        if self.build_integrator is None:
            return None
        if ivp is None:
            raise TypeError("ComparisonEntry requires a ComparisonProblem built from a SystemIVP.")
        return self.build_integrator(ivp)


@dataclass(slots=True, init=False)
class ComparisonEntryStepper:
    """Low-level comparison entry backed by a user-supplied stepper."""

    name: str
    build_stepper: StepperBuilder = field(repr=False)
    build_observed_stepper: ObservedStepperBuilder | None = field(default=None, repr=False)
    build_integrator: Callable[[], Any] | None = None
    profile_category: ProfileCategory | None = None
    metadata: dict[str, Any] | None = None

    def __init__(
        self,
        name: str,
        stepper: Any,
        build_observed_stepper: ObservedStepperBuilder | None = None,
        build_integrator: Callable[[], Any] | None = None,
        metadata: dict[str, Any] | None = None,
        profile_category: ProfileCategory | None = None,
    ) -> None:
        self.name = name
        self.build_stepper = _normalize_stepper_builder(stepper)
        self.build_observed_stepper = build_observed_stepper
        self.build_integrator = build_integrator
        self.profile_category = profile_category
        self.metadata = metadata

    def make_stepper(self, ivp: Any | None = None) -> Any:
        del ivp
        source = self.build_stepper
        if _accepts_zero_arguments(source):
            return source()
        return source

    def make_observed_stepper(self, monitor: Any, ivp: Any | None = None) -> Any:
        del ivp
        source = self.build_observed_stepper
        if source is None:
            return self.make_stepper()
        return source(monitor)

    def make_integrator(self, ivp: Any | None = None) -> Any | None:
        del ivp
        if self.build_integrator is None:
            return None
        return self.build_integrator()


ComparisonProblemLike = ComparisonProblem | ComparisonProblemManual
ComparisonEntryLike = ComparisonEntry | ComparisonEntryStepper


@dataclass(slots=True)
class ComparisonBreakdown:
    """Profiled self-time grouped into STARK comparison buckets."""

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
    """One non-STARK profile hotspot reported for a custom comparison entry."""

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
    """Setup, warmup, and repeated-run timings for one comparison entry."""

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
    """Named final-state diagnostics supplied by a comparison problem."""

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
        from stark.diagnostics.comparison.writers import ComparisonDiagnosticsWriter

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
    """Labelled pairwise matrix, usually final-state or trajectory errors."""

    labels: list[str]
    values: list[list[float]]
    note: str | None = None

    @classmethod
    def fieldwise_rms_error(
        cls,
        observed: Any,
        reference: Any,
        fields: Iterable[str],
        *,
        sample_count: int | None = None,
    ) -> float:
        """
        Return an RMS error over named state fields.

        `observed` and `reference` may expose fields as attributes or mapping
        keys. By default the RMS is normalised by the total number of values in
        the selected fields. Pass `sample_count` when the fields are coupled
        components of the same physical sample and the desired scale is
        per-sample rather than per-stored-value.
        """

        total = 0.0
        value_count = 0
        for field in fields:
            observed_value = _field_value(observed, field)
            reference_value = _field_value(reference, field)
            delta = observed_value - reference_value
            total += _squared_norm(delta)
            value_count += _value_count(delta)

        denominator = value_count if sample_count is None else sample_count
        if denominator == 0:
            return 0.0
        return sqrt(total / denominator)

    def rows(self) -> list[tuple[str, list[float]]]:
        return list(zip(self.labels, self.values, strict=True))

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {
            row_name: {column_name: value for column_name, value in zip(self.labels, row_values, strict=True)}
            for row_name, row_values in self.rows()
        }

    def __str__(self) -> str:
        from stark.diagnostics.comparison.writers import ComparisonWriter

        return ComparisonWriter()(self)

    def __repr__(self) -> str:
        return f"Comparison(labels={self.labels!r})"


@dataclass(slots=True)
class ComparisonProfile:
    """Profiler summary for one comparison entry."""

    breakdown: ComparisonBreakdown
    note: str | None
    custom_hotspots: list[ComparisonHotspot]

    def __str__(self) -> str:
        from stark.diagnostics.comparison.writers import ComparisonProfileWriter

        return ComparisonProfileWriter()(self)

    def as_dict(self) -> dict[str, Any]:
        return {
            "breakdown": self.breakdown.as_dict(),
            "note": self.note,
            "custom_hotspots": [hotspot.as_dict() for hotspot in self.custom_hotspots],
        }


@dataclass(slots=True)
class ComparisonResult:
    """Complete observed result for one comparison entry."""

    name: str
    steps: int
    timing: ComparisonTiming
    diagnostics: ComparisonDiagnostics
    profile: ComparisonProfile
    metadata: dict[str, Any]
    monitor_summary: MonitorSummary | None = None

    def __str__(self) -> str:
        from stark.diagnostics.comparison.writers import ComparisonResultWriter

        return ComparisonResultWriter()(self)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "steps": self.steps,
            "timing": self.timing.as_dict(),
            "diagnostics": self.diagnostics.as_dict(),
            "profile": self.profile.as_dict(),
            "metadata": dict(self.metadata),
            "monitor_summary": None if self.monitor_summary is None else asdict(self.monitor_summary),
        }


@dataclass(slots=True)
class ComparisonReport:
    """Complete comparison report returned by `ComparisonRunner`."""

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
        from stark.diagnostics.comparison.writers import ComparisonReportWriter

        return ComparisonReportWriter()(self)

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

    def monitor_summaries_by_name(self) -> dict[str, MonitorSummary]:
        return {
            result.name: result.monitor_summary
            for result in self.results
            if result.monitor_summary is not None
        }

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


def _copy_ivp_initial(ivp: Any) -> Any:
    state = ivp.engine.allocator.allocate_state()
    ivp.engine.allocator.copy_state(ivp.initial, state)
    return state


def _copy_ivp_interval(ivp: Any) -> Any:
    copy = getattr(ivp.interval, "copy", None)
    if callable(copy):
        return copy()
    return ivp.interval


def _ivp_state_difference(ivp: Any) -> Difference:
    def difference(left: Any, right: Any) -> float:
        delta = ivp.engine.allocator.allocate_translation()
        for field, carrier in zip(ivp.engine.algebraist_frame.fields, ivp.engine.carriers, strict=True):
            carrier.arithmetic.combine2(
                1.0,
                field.state_path(right),
                -1.0,
                field.state_path(left),
                field.translation_path(delta),
            )
        return delta.norm()

    return difference


def _ivp_method_stepper(
    ivp: Any,
    method: Any,
    *,
    configuration: Any | None = None,
) -> Any:
    from stark.core.integrator.stepper import IntegratorStepper

    scheme = ivp.system.prepare_scheme(
        method,
        ivp.engine,
        ivp.scheme.derivative,
        ivp.configuration if configuration is None else configuration,
    )
    return IntegratorStepper(scheme)


def _method_with_scheme_monitor(method: Any, monitor: Any) -> Any:
    from stark.methods import Method

    scheme_options = dict(method.scheme_options)
    scheme_options["monitor"] = monitor.scheme
    return Method(
        scheme=method.scheme,
        resolvent=method.resolvent,
        inverter=method.inverter,
        scheme_options=scheme_options,
        resolvent_options=method.resolvent_options,
        inverter_options=method.inverter_options,
    )


def _field_value(source: Any, field: str) -> Any:
    if isinstance(source, Mapping):
        return source[field]
    return getattr(source, field)


def _squared_norm(value: Any) -> float:
    flat = _flatten_value(value)
    dot = getattr(flat, "dot", None)
    if callable(dot):
        return float(cast(SupportsFloat, dot(flat)))
    try:
        return float(cast(SupportsFloat, flat * flat))
    except TypeError:
        return float(sum(item * item for item in flat))


def _value_count(value: Any) -> int:
    flat = _flatten_value(value)
    size = getattr(flat, "size", None)
    if size is not None:
        return int(size)
    try:
        return len(flat)
    except TypeError:
        return 1


def _flatten_value(value: Any) -> Any:
    ravel = getattr(value, "ravel", None)
    if callable(ravel):
        return ravel()
    return value


__all__ = [
    "ComparisonEntry",
    "ComparisonEntryLike",
    "ComparisonEntryStepper",
    "ComparisonProblem",
    "ComparisonProblemLike",
    "ComparisonProblemManual",
    "ComparisonReport",
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
    "StepperBuilder",
    "ObservedStepperBuilder",
    "ProfileCategory",
    "StateBuilder",
    "TrajectoryDifference",
]


