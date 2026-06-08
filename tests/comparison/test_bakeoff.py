import pytest

from stark import Interval, IntegratorStepper, Tolerance
from stark.comparison import ComparisonRunner, ComparisonEntryStepper, ComparisonProblemManual
from stark.schemes import SchemeCashKarp, SchemeEuler


class DummyInterval:
    def __init__(self, present: float, step: float, stop: float) -> None:
        self.present = present
        self.step = step
        self.stop = stop

    def copy(self) -> "DummyInterval":
        return DummyInterval(self.present, self.step, self.stop)

    def increment(self, dt: float) -> None:
        self.present += dt


class WeightedStepper:
    def __init__(self, weight: float) -> None:
        self.weight = weight

    def __call__(self, interval: DummyInterval, state: dict[str, float]) -> None:
        state["value"] += self.weight * interval.step
        interval.increment(interval.step)

    def snapshot_state(self, state: dict[str, float]) -> dict[str, float]:
        return dict(state)


class MonitorableWeightedStepper(WeightedStepper):
    def __init__(self, weight: float, monitor) -> None:
        super().__init__(weight)
        self.monitor = monitor

    def __call__(self, interval: DummyInterval, state: dict[str, float]) -> None:
        t_start = interval.present
        super().__call__(interval, state)
        if self.monitor is not None:
            self.monitor.scheme.record_fixed_step("Weighted", t_start, interval.step)
            self.monitor.inverter.record_solve("Direct", True, None, None, None, None)


class ScalarState:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value


class ScalarTranslation:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self, origin: ScalarState, result: ScalarState) -> None:
        result.value = origin.value + self.value

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: "ScalarTranslation") -> "ScalarTranslation":
        return ScalarTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> "ScalarTranslation":
        return ScalarTranslation(scalar * self.value)


class ScalarAllocator:
    def allocate_state(self) -> ScalarState:
        return ScalarState()

    def copy_state(self, source: ScalarState, out: ScalarState) -> None:
        out.value = source.value

    def allocate_translation(self) -> ScalarTranslation:
        return ScalarTranslation()


def unit_rhs(interval: Interval, state: ScalarState, out: ScalarTranslation) -> None:
    del interval, state
    out.value = 1.0


def test_bakeoff_reports_pairwise_differences() -> None:
    problem = ComparisonProblemManual(
        name="Dummy",
        build_state=lambda: {"value": 0.0},
        build_interval=lambda: DummyInterval(0.0, 0.25, 1.0),
        difference=lambda left, right: abs(left["value"] - right["value"]),
        diagnostics=lambda state: {"value": state["value"]},
    )
    entries = [
        ComparisonEntryStepper("baseline", lambda: WeightedStepper(1.0), metadata={"variant": "baseline", "accelerator": "none"}),
        ComparisonEntryStepper("slow", lambda: WeightedStepper(0.5), metadata={"variant": "slow", "accelerator": "none"}),
    ]

    report = ComparisonRunner(problem, entries, repeats=2)()

    assert len(report.results) == 2
    assert report.results[0].steps == 4
    assert report.results_by_name()["baseline"].timing.median >= 0.0
    assert report.diagnostics_by_name()["baseline"]["value"] == 1.0
    assert report.metadata_by_name()["baseline"]["variant"] == "baseline"
    assert report.final_difference_map()["baseline"]["baseline"] == 0.0
    assert report.final_difference_map()["baseline"]["slow"] > 0.0
    assert report.final_difference_map()["slow"]["baseline"] > 0.0
    assert "Pairwise final-state differences" in report.render()
    assert "Configuration Table" in report.render()
    assert "Diagnostics Table" in report.render()
    assert "Custom entry: the profile buckets are approximate" in report.render()
    assert "Custom entry hotspots" in report.render()
    assert "test_bakeoff.py" in report.render()
    assert "baseline" in report.render()
    assert "slow" in report.render()


def test_bakeoff_accepts_direct_steppers() -> None:
    problem = ComparisonProblemManual(
        name="Dummy",
        build_state=lambda: {"value": 0.0},
        build_interval=lambda: DummyInterval(0.0, 0.25, 1.0),
        difference=lambda left, right: abs(left["value"] - right["value"]),
    )
    entries = [
        ComparisonEntryStepper("baseline", WeightedStepper(1.0)),
        ComparisonEntryStepper("slow", lambda: WeightedStepper(0.5)),
    ]

    report = ComparisonRunner(problem, entries, repeats=1)()

    assert report.final_difference_map()["baseline"]["baseline"] == 0.0
    assert report.final_difference_map()["baseline"]["slow"] > 0.0


def test_bakeoff_reports_pairwise_trajectory_differences() -> None:
    problem = ComparisonProblemManual(
        name="Dummy",
        build_state=lambda: {"value": 0.0},
        build_interval=lambda: DummyInterval(0.0, 0.25, 1.0),
        difference=lambda left, right: abs(left["value"] - right["value"]),
        checkpoints=2,
    )
    entries = [
        ComparisonEntryStepper("baseline", lambda: WeightedStepper(1.0), metadata={"variant": "baseline"}),
        ComparisonEntryStepper("slow", lambda: WeightedStepper(0.5), metadata={"variant": "slow"}),
    ]

    report = ComparisonRunner(problem, entries, repeats=1)()

    assert report.trajectory_differences is not None
    assert report.trajectory_difference_map() is not None
    assert report.trajectory_difference_map()["baseline"]["baseline"] == 0.0
    assert report.trajectory_difference_map()["baseline"]["slow"] > 0.0
    assert "Pairwise trajectory differences" in report.render()
    assert "default RMS of checkpoint-wise state differences" in report.render()


def test_bakeoff_skips_diagnostics_table_when_problem_has_none() -> None:
    problem = ComparisonProblemManual(
        name="Dummy",
        build_state=lambda: {"value": 0.0},
        build_interval=lambda: DummyInterval(0.0, 0.25, 1.0),
        difference=lambda left, right: abs(left["value"] - right["value"]),
    )
    entries = [
        ComparisonEntryStepper("baseline", lambda: WeightedStepper(1.0)),
        ComparisonEntryStepper("slow", lambda: WeightedStepper(0.5)),
    ]

    report = ComparisonRunner(problem, entries, repeats=1)()

    assert "Diagnostics Table" not in report.render()


def test_bakeoff_uses_monitored_observation_without_monitoring_timed_repeats() -> None:
    problem = ComparisonProblemManual(
        name="Dummy",
        build_state=lambda: {"value": 0.0},
        build_interval=lambda: DummyInterval(0.0, 0.25, 1.0),
        difference=lambda left, right: abs(left["value"] - right["value"]),
    )
    entries = [
        ComparisonEntryStepper(
            "baseline",
            lambda: WeightedStepper(1.0),
            build_observed_stepper=lambda monitor: MonitorableWeightedStepper(1.0, monitor),
        ),
        ComparisonEntryStepper("slow", lambda: WeightedStepper(0.5)),
    ]

    report = ComparisonRunner(problem, entries, repeats=2)()

    summaries = report.monitor_summaries_by_name()
    assert "baseline" in summaries
    assert "slow" not in summaries
    assert summaries["baseline"].scheme.step_count == 4
    assert summaries["baseline"].inverter.solve_count == 4
    assert report.results_by_name()["baseline"].as_dict()["monitor_summary"] is not None
    rendered = report.render()
    assert "Scheme behaviour" in rendered
    assert "Inverter behaviour" in rendered
    assert "monitored observation pass, not the timed repeats" in rendered


def test_bakeoff_requires_two_entries() -> None:
    problem = ComparisonProblemManual(
        name="Dummy",
        build_state=lambda: {"value": 0.0},
        build_interval=lambda: DummyInterval(0.0, 0.25, 1.0),
        difference=lambda left, right: abs(left["value"] - right["value"]),
    )

    try:
        ComparisonRunner(problem, [ComparisonEntryStepper("only", lambda: WeightedStepper(1.0))])
    except ValueError as exc:
        assert "at least two entries" in str(exc)
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("Expected bakeoff to require at least two entries.")











