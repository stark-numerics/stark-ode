import numpy as np
import pytest

from stark import Frame, Interval, Method, System
from stark.diagnostics.comparison import Comparison, ComparisonEntry, ComparisonProblem, ComparisonRunner
from stark.engines import EngineNumpy
from stark.methods import SchemeEuler, SchemeRK4


def decay_rhs(t: float, state, out) -> None:
    del t
    out.dy[0] = -0.5 * state.y[0]


class CustomForwardEuler:
    """Minimal user-defined scheme for comparison tests."""

    def __init__(self, derivative, allocator) -> None:
        self.derivative = derivative
        self.allocator = allocator
        self.delta = allocator.allocate_translation()

    def __call__(self, interval, state) -> float:
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        dt = interval.step if interval.step <= remaining else remaining
        self.derivative(interval, state, self.delta)
        (dt * self.delta)(state, state)
        return dt

    def snapshot_state(self, state):
        snapshot = self.allocator.allocate_state()
        self.allocator.copy_state(state, snapshot)
        return snapshot


def build_problem(
    *,
    diagnostics: bool = True,
    checkpoints: int | None = None,
) -> ComparisonProblem:
    system = System(
        derivative=decay_rhs,
        frame=Frame.scalar("y", translation="dy"),
    )
    template = system.ivp(
        initial={"y": np.array([2.0])},
        interval=Interval(present=0.0, step=0.1, stop=1.0),
        method=Method(SchemeRK4),
        engine=EngineNumpy,
    )
    return ComparisonProblem(
        "decay",
        template,
        diagnostics=(lambda state: {"y": float(state.y[0])}) if diagnostics else None,
        checkpoints=checkpoints,
    )


def test_comparison_reports_pairwise_differences() -> None:
    problem = build_problem()
    entries = [
        ComparisonEntry("Euler", Method(SchemeEuler), metadata={"variant": "baseline"}),
        ComparisonEntry("RK4", Method(SchemeRK4), metadata={"variant": "candidate"}),
    ]

    report = ComparisonRunner(problem, entries, repeats=2)()

    assert len(report.results) == 2
    assert report.results_by_name()["Euler"].timing.median >= 0.0
    assert report.diagnostics_by_name()["Euler"]["y"] > 0.0
    assert report.metadata_by_name()["Euler"]["variant"] == "baseline"
    assert report.final_difference_map()["Euler"]["Euler"] == 0.0
    assert report.final_difference_map()["Euler"]["RK4"] > 0.0
    rendered = report.render()
    assert "Pairwise final-state differences" in rendered
    assert "Configuration Table" in rendered
    assert "Diagnostics Table" in rendered
    assert "Euler" in rendered
    assert "RK4" in rendered


def test_comparison_accepts_user_defined_scheme_through_method() -> None:
    problem = build_problem()
    entries = [
        ComparisonEntry("custom Euler", Method(CustomForwardEuler)),
        ComparisonEntry("built-in RK4", Method(SchemeRK4)),
    ]

    report = ComparisonRunner(problem, entries, repeats=1)()

    assert report.final_difference_map()["custom Euler"]["built-in RK4"] > 0.0
    assert "Custom scheme CustomForwardEuler" in report.render()


def test_comparison_fieldwise_rms_error_supports_value_and_sample_scaling() -> None:
    observed = {
        "q": np.array([3.0, 4.0]),
        "p": np.array([0.0, 12.0]),
    }
    reference = {
        "q": np.array([0.0, 0.0]),
        "p": np.array([0.0, 0.0]),
    }

    assert Comparison.fieldwise_rms_error(observed, reference, ("q", "p")) == 6.5
    assert Comparison.fieldwise_rms_error(
        observed,
        reference,
        ("q", "p"),
        sample_count=2,
    ) == pytest.approx((169.0 / 2.0) ** 0.5)


def test_comparison_reports_pairwise_trajectory_differences() -> None:
    problem = build_problem(checkpoints=2)
    entries = [
        ComparisonEntry("Euler", Method(SchemeEuler), metadata={"variant": "baseline"}),
        ComparisonEntry("RK4", Method(SchemeRK4), metadata={"variant": "candidate"}),
    ]

    report = ComparisonRunner(problem, entries, repeats=1)()

    assert report.trajectory_differences is not None
    assert report.trajectory_difference_map() is not None
    assert report.trajectory_difference_map()["Euler"]["Euler"] == 0.0
    assert report.trajectory_difference_map()["Euler"]["RK4"] > 0.0
    assert "Pairwise trajectory differences" in report.render()
    assert "default RMS of checkpoint-wise state differences" in report.render()


def test_comparison_skips_diagnostics_table_when_problem_has_none() -> None:
    problem = build_problem(diagnostics=False)
    entries = [
        ComparisonEntry("Euler", Method(SchemeEuler)),
        ComparisonEntry("RK4", Method(SchemeRK4)),
    ]

    report = ComparisonRunner(problem, entries, repeats=1)()

    assert "Diagnostics Table" not in report.render()


def test_comparison_uses_monitored_observation_without_monitoring_timed_repeats() -> None:
    problem = build_problem()
    entries = [
        ComparisonEntry("Euler", Method(SchemeEuler)),
        ComparisonEntry("RK4", Method(SchemeRK4)),
    ]

    report = ComparisonRunner(problem, entries, repeats=2)()

    summaries = report.monitor_summaries_by_name()
    assert set(summaries) == {"Euler", "RK4"}
    assert summaries["Euler"].scheme.step_count == report.results_by_name()["Euler"].steps
    assert report.results_by_name()["Euler"].as_dict()["monitor_summary"] is not None
    rendered = report.render()
    assert "Scheme behaviour" in rendered
    assert "monitored observation pass, not the timed repeats" in rendered


def test_comparison_requires_two_entries() -> None:
    problem = build_problem()

    with pytest.raises(ValueError, match="at least two entries"):
        ComparisonRunner(problem, [ComparisonEntry("only", Method(SchemeEuler))])
