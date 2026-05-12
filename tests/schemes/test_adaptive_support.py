from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark import Executor, Interval, Tolerance
from stark.execution.adaptive_controller import AdaptiveController
from stark.execution.regulator import Regulator
from stark.schemes.base import (
    _ADVANCE_ACCEPTED_DT,
    _ADVANCE_ERROR_RATIO,
    _ADVANCE_NEXT_DT,
    _ADVANCE_PROPOSED_DT,
    _ADVANCE_REJECTION_COUNT,
    _ADVANCE_T_START,
)
from stark.schemes.explicit_adaptive.bogacki_shampine import SchemeBogackiShampine
from stark.schemes.support.adaptive import (
    ReportAdaptiveAdvance,
    SchemeSupportAdaptive,
)


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


class ScalarWorkbench:
    def allocate_state(self) -> ScalarState:
        return ScalarState()

    def copy_state(self, dst: ScalarState, src: ScalarState) -> None:
        dst.value = src.value

    def allocate_translation(self) -> ScalarTranslation:
        return ScalarTranslation()


def zero_rhs(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval, state
    out.value = 0.0


def test_adaptive_support_owns_regulator_controller_and_advance_report() -> None:
    regulator = Regulator(
        safety=0.7,
        min_factor=0.2,
        max_factor=3.0,
        error_exponent=0.25,
    )

    support = SchemeSupportAdaptive(regulator)

    assert support.regulator is regulator
    assert isinstance(support.controller, AdaptiveController)
    assert support.advance_report == [0.0, 0.0, 0.0, 0.0, 0.0, 0]

    assert support.runtime_bound is False
    assert support.active_controller is None
    assert support.ratio is None
    assert support.bound is None


def test_adaptive_support_binds_and_unbinds_executor_runtime() -> None:
    support = SchemeSupportAdaptive()
    executor = Executor(tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))

    support.assign_executor(executor)

    assert support.runtime_bound is True
    assert support.active_controller is not None
    assert support.ratio is not None
    assert support.bound is not None

    support.unassign_executor()

    assert support.runtime_bound is False
    assert support.active_controller is None
    assert support.ratio is None
    assert support.bound is None


def test_adaptive_support_computes_proposed_step() -> None:
    support = SchemeSupportAdaptive()
    interval = Interval(present=0.2, step=0.5, stop=0.3)

    proposal = support.propose_step(interval)

    assert proposal.remaining == pytest.approx(0.1)
    assert proposal.dt == pytest.approx(0.1)
    assert proposal.proposed_dt == pytest.approx(0.1)
    assert proposal.t_start == pytest.approx(0.2)


def test_adaptive_support_delegates_rejected_and_accepted_step_calculations() -> None:
    support = SchemeSupportAdaptive(
        Regulator(
            safety=0.8,
            min_factor=0.1,
            max_factor=5.0,
            error_exponent=0.2,
        )
    )
    support.assign_executor(Executor())

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


def test_adaptive_support_records_stopped_advance() -> None:
    support = SchemeSupportAdaptive()
    interval = Interval(present=1.0, step=0.1, stop=1.0)

    support.record_stopped(interval)

    assert support.advance_report[_ADVANCE_ACCEPTED_DT] == pytest.approx(0.0)
    assert support.advance_report[_ADVANCE_T_START] == pytest.approx(1.0)
    assert support.advance_report[_ADVANCE_PROPOSED_DT] == pytest.approx(0.0)
    assert support.advance_report[_ADVANCE_NEXT_DT] == pytest.approx(0.0)
    assert support.advance_report[_ADVANCE_ERROR_RATIO] == pytest.approx(0.0)
    assert support.advance_report[_ADVANCE_REJECTION_COUNT] == 0


def test_adaptive_support_records_accepted_advance_and_exposes_report_object() -> None:
    support = SchemeSupportAdaptive()

    report = support.record_accepted(
        accepted_dt=0.1,
        t_start=0.2,
        proposed_dt=0.15,
        next_dt=0.3,
        error_ratio=0.4,
        rejection_count=2,
    )

    assert isinstance(report, ReportAdaptiveAdvance)
    assert report.accepted_dt == pytest.approx(0.1)
    assert report.t_start == pytest.approx(0.2)
    assert report.t_end == pytest.approx(0.3)
    assert report.proposed_dt == pytest.approx(0.15)
    assert report.next_dt == pytest.approx(0.3)
    assert report.error_ratio == pytest.approx(0.4)
    assert report.rejection_count == 2

    assert support.report() == report


def test_adaptive_scheme_base_exposes_existing_runtime_attributes_through_support() -> None:
    scheme = SchemeBogackiShampine(zero_rhs, ScalarWorkbench())

    assert isinstance(scheme.adaptive, SchemeSupportAdaptive)
    assert scheme.advance_report is scheme.adaptive.advance_report
    assert scheme.regulator is scheme.adaptive.regulator
    assert scheme.controller is scheme.adaptive.controller

    assert scheme._runtime_bound is False
    assert scheme._controller is None
    assert scheme._ratio is None
    assert scheme._bound is None


def test_existing_adaptive_scheme_still_runs_after_support_extraction() -> None:
    scheme = SchemeBogackiShampine(zero_rhs, ScalarWorkbench())
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(2.0)
    executor = Executor(tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))

    accepted_dt = scheme(interval, state, executor)

    assert accepted_dt == pytest.approx(0.1)
    assert state.value == pytest.approx(2.0)
    assert interval.step == pytest.approx(0.2)