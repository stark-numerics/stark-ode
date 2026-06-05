from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark import Configuration, Interval, Tolerance
from stark.schemes.execution.step_control import (
    SchemeStepAdaptiveAdvanceReport,
    SchemeStepControl,
)
from stark.schemes.explicit.adaptive.bogacki_shampine import SchemeBogackiShampine


@dataclass(slots=True)
class ScalarState:
    value: float = 0.0


@dataclass(slots=True)
class ScalarTranslation:
    value: float = 0.0

    def __call__(self, origin: ScalarState, result: ScalarState) -> None:
        result.value = origin.value + self.value

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: ScalarTranslation) -> ScalarTranslation:
        return ScalarTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> ScalarTranslation:
        return ScalarTranslation(scalar * self.value)


class ScalarAllocator:
    def allocate_state(self) -> ScalarState:
        return ScalarState()

    def copy_state(self, source: ScalarState, out: ScalarState) -> None:
        out.value = source.value

    def allocate_translation(self) -> ScalarTranslation:
        return ScalarTranslation()


def zero_rhs(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval, state
    out.value = 0.0


def test_adaptive_support_owns_configuration_and_report_state() -> None:
    configuration = Configuration(
        scheme_tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9),
        adaptive_scheme_safety=0.7,
        adaptive_scheme_min_factor=0.2,
        adaptive_scheme_max_factor=3.0,
        adaptive_scheme_error_exponent=0.25,
    )

    support = SchemeStepControl(configuration)
    report = support.report()

    assert support.configuration is configuration
    assert support.ratio(1.0e-10, 1.0) < 1.0
    assert support.bound(1.0) == pytest.approx(configuration.scheme_tolerance.bound(1.0))
    assert isinstance(report, SchemeStepAdaptiveAdvanceReport)
    assert report.accepted_dt == pytest.approx(0.0)
    assert report.t_start == pytest.approx(0.0)
    assert report.t_end == pytest.approx(0.0)
    assert report.proposed_dt == pytest.approx(0.0)
    assert report.next_dt == pytest.approx(0.0)
    assert report.error_ratio == pytest.approx(0.0)
    assert report.rejection_count == 0


def test_adaptive_support_computes_proposed_step() -> None:
    support = SchemeStepControl(Configuration())
    interval = Interval(present=0.2, step=0.5, stop=0.3)

    proposal = support.propose_step(interval)

    assert proposal.remaining == pytest.approx(0.1)
    assert proposal.dt == pytest.approx(0.1)
    assert proposal.proposed_dt == pytest.approx(0.1)
    assert proposal.t_start == pytest.approx(0.2)


def test_adaptive_support_records_stopped_report() -> None:
    support = SchemeStepControl(Configuration())
    interval = Interval(present=1.0, step=0.1, stop=1.0)

    report = support.record_stopped(interval)

    assert report is support.report()
    assert report.accepted_dt == pytest.approx(0.0)
    assert report.t_start == pytest.approx(1.0)
    assert report.t_end == pytest.approx(1.0)
    assert report.proposed_dt == pytest.approx(0.0)
    assert report.next_dt == pytest.approx(0.0)
    assert report.error_ratio == pytest.approx(0.0)
    assert report.rejection_count == 0


def test_adaptive_support_delegates_rejected_and_accepted_step_calculations() -> None:
    support = SchemeStepControl(
        Configuration(
            adaptive_scheme_safety=0.8,
            adaptive_scheme_min_factor=0.1,
            adaptive_scheme_max_factor=5.0,
            adaptive_scheme_error_exponent=0.2,
        )
    )

    rejected_dt = support.rejected_step(
        dt=0.1,
        error_ratio=32.0,
        remaining=1.0,
        label="test",
    )
    next_dt = support.accepted_next_step(
        accepted_dt=0.1,
        error_ratio=0.25,
        remaining_after=0.3,
    )

    assert rejected_dt > 0.0
    assert rejected_dt < 0.1
    assert next_dt > 0.1
    assert next_dt <= 0.3


def test_adaptive_support_records_accepted_report() -> None:
    support = SchemeStepControl(Configuration())

    report = support.record_accepted(
        accepted_dt=0.1,
        t_start=0.2,
        proposed_dt=0.15,
        next_dt=0.3,
        error_ratio=0.4,
        rejection_count=2,
    )

    assert isinstance(report, SchemeStepAdaptiveAdvanceReport)
    assert report is support.report()
    assert report.accepted_dt == pytest.approx(0.1)
    assert report.t_start == pytest.approx(0.2)
    assert report.t_end == pytest.approx(0.3)
    assert report.proposed_dt == pytest.approx(0.15)
    assert report.next_dt == pytest.approx(0.3)
    assert report.error_ratio == pytest.approx(0.4)
    assert report.rejection_count == 2


def test_adaptive_scheme_exposes_step_control_without_legacy_report() -> None:
    scheme = SchemeBogackiShampine(zero_rhs, ScalarAllocator())

    assert isinstance(scheme.step_control, SchemeStepControl)
    assert not hasattr(scheme, "advance_report")


def test_existing_adaptive_scheme_still_runs_after_support_cleanup() -> None:
    configuration = Configuration(scheme_tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))
    scheme = SchemeBogackiShampine(zero_rhs, ScalarAllocator(), configuration=configuration)
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(2.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.1)
    assert state.value == pytest.approx(2.0)
    assert interval.step == pytest.approx(0.2)

    report = scheme.step_control.report()
    assert report.accepted_dt == pytest.approx(0.1)
    assert report.t_start == pytest.approx(0.0)
    assert report.t_end == pytest.approx(0.1)
    assert report.proposed_dt == pytest.approx(0.1)
    assert report.error_ratio == pytest.approx(0.0)
    assert report.rejection_count == 0
