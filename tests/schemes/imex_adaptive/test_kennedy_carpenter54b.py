from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark import Executor, Interval, Tolerance
from stark.accelerators import Accelerator
from stark.monitor import Monitor
from stark.resolvents import ResolventPicard
from stark.resolvents.policy import ResolventPolicy
from stark.schemes.imex_adaptive.kennedy_carpenter54b import (
    SchemeKennedyCarpenter54b,
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


@dataclass(slots=True)
class SplitDerivative:
    explicit: object
    implicit: object


def zero_rhs(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval, state
    out.value = 0.0


def make_scheme() -> SchemeKennedyCarpenter54b:
    workbench = ScalarWorkbench()
    derivative = SplitDerivative(
        explicit=zero_rhs,
        implicit=zero_rhs,
    )
    resolvent = ResolventPicard(
        zero_rhs,
        workbench,
        tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=ResolventPolicy(max_iterations=8),
        accelerator=Accelerator.none(),
        tableau=SchemeKennedyCarpenter54b.tableau,
    )
    return SchemeKennedyCarpenter54b(
        derivative,
        workbench,
        resolvent=resolvent,
    )


def test_kennedy_carpenter54b_owns_converted_call_surface() -> None:
    assert "__call__" in SchemeKennedyCarpenter54b.__dict__
    assert "call_bind" in SchemeKennedyCarpenter54b.__dict__
    assert "call_generic" in SchemeKennedyCarpenter54b.__dict__
    assert "call_monitored" in SchemeKennedyCarpenter54b.__dict__
    assert "refresh_call" in SchemeKennedyCarpenter54b.__dict__

    scheme = make_scheme()

    assert scheme.redirect_call.__func__ is scheme.call_bind.__func__
    assert scheme.call_pure.__func__ is scheme.call_generic.__func__


def test_kennedy_carpenter54b_accepts_zero_split_step() -> None:
    scheme = make_scheme()
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(2.0)
    executor = Executor(tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))

    accepted_dt = scheme(interval, state, executor)

    assert accepted_dt == pytest.approx(0.1)
    assert state.value == pytest.approx(2.0)
    assert scheme.redirect_call.__func__ is scheme.call_pure.__func__

    report = scheme.adaptive.report()
    assert report.accepted_dt == pytest.approx(0.1)
    assert report.proposed_dt == pytest.approx(0.1)
    assert report.error_ratio == pytest.approx(0.0)
    assert report.rejection_count == 0


def test_kennedy_carpenter54b_clips_to_remaining_interval() -> None:
    scheme = make_scheme()
    interval = Interval(present=0.25, step=0.1, stop=0.3)
    state = ScalarState(2.0)
    executor = Executor(tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))

    accepted_dt = scheme(interval, state, executor)

    assert accepted_dt == pytest.approx(0.05)
    assert state.value == pytest.approx(2.0)

    report = scheme.adaptive.report()
    assert report.accepted_dt == pytest.approx(0.05)
    assert report.proposed_dt == pytest.approx(0.05)


def test_kennedy_carpenter54b_monitoring_uses_scheme_owned_boundary() -> None:
    scheme = make_scheme()
    monitor = Monitor()
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(2.0)
    executor = Executor(tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))

    scheme.assign_executor(executor)
    scheme.assign_monitor(monitor)

    assert scheme.redirect_call.__func__ is scheme.call_monitored.__func__

    accepted_dt = scheme(interval, state, executor)

    assert accepted_dt == pytest.approx(0.1)
    assert len(monitor.steps) == 1

    record = monitor.steps[0]
    assert record.scheme == scheme.short_name
    assert record.t_start == pytest.approx(0.0)
    assert record.t_end == pytest.approx(0.1)
    assert record.proposed_dt == pytest.approx(0.1)
    assert record.accepted_dt == pytest.approx(0.1)
    assert record.error_ratio == pytest.approx(0.0)
    assert record.rejection_count == 0