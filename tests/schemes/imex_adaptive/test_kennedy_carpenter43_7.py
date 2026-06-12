from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark import Interval, Tolerance
from stark.engines.accelerators import AcceleratorNone
from stark.monitor import Monitor
from stark.methods.resolvents import ResolventPicard
from stark import Configuration
from stark.methods.schemes.imex.adaptive.kennedy_carpenter43_7 import (
    SchemeKennedyCarpenter43_7,
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


class ScalarAllocator:
    def allocate_state(self) -> ScalarState:
        return ScalarState()

    def copy_state(self, source: ScalarState, out: ScalarState) -> None:
        out.value = source.value

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


def make_scheme(*, monitor=None) -> SchemeKennedyCarpenter43_7:
    allocator = ScalarAllocator()
    derivative = SplitDerivative(
        explicit=zero_rhs,
        implicit=zero_rhs,
    )
    resolvent = ResolventPicard(
        allocator,
        configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12), resolvent_maximum_steps=8),
        accelerator=AcceleratorNone(),
        tableau=SchemeKennedyCarpenter43_7.tableau,
    )
    return SchemeKennedyCarpenter43_7(
        derivative,
        allocator,
        resolvent=resolvent,
        monitor=monitor,
    )


def test_kennedy_carpenter43_7_owns_converted_call_surface() -> None:
    assert hasattr(SchemeKennedyCarpenter43_7, "__call__")
    assert hasattr(SchemeKennedyCarpenter43_7, "call_inline")
    assert hasattr(SchemeKennedyCarpenter43_7, "call_specialized")
    assert hasattr(SchemeKennedyCarpenter43_7, "call_monitored")

    scheme = make_scheme()

    assert scheme.redirect_call.__func__ is scheme.call_step.__func__
    assert scheme.call_step.__func__ is scheme.call_inline.__func__


def test_kennedy_carpenter43_7_accepts_zero_split_step() -> None:
    scheme = make_scheme()
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(2.0)
    configuration = Configuration(scheme_tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.1)
    assert state.value == pytest.approx(2.0)
    assert scheme.redirect_call.__func__ is scheme.call_step.__func__

    report = scheme.step_control.report()
    assert report.accepted_dt == pytest.approx(0.1)
    assert report.proposed_dt == pytest.approx(0.1)
    assert report.error_ratio == pytest.approx(0.0)
    assert report.rejection_count == 0


def test_kennedy_carpenter43_7_clips_to_remaining_interval() -> None:
    scheme = make_scheme()
    interval = Interval(present=0.25, step=0.1, stop=0.3)
    state = ScalarState(2.0)
    configuration = Configuration(scheme_tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.05)
    assert state.value == pytest.approx(2.0)

    report = scheme.step_control.report()
    assert report.accepted_dt == pytest.approx(0.05)
    assert report.proposed_dt == pytest.approx(0.05)


def test_kennedy_carpenter43_7_monitoring_uses_scheme_owned_boundary() -> None:
    monitor = Monitor()
    scheme = make_scheme(monitor=monitor.scheme)
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(2.0)
    configuration = Configuration(scheme_tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))

    assert scheme.monitor is monitor.scheme
    assert scheme.call_step.__func__ is scheme.call_monitored.__func__
    assert scheme.redirect_call == scheme.call_step

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.1)
    assert len(monitor.scheme.adaptive_steps) == 1

    record = monitor.scheme.adaptive_steps[0]
    assert record.scheme == scheme.short_name
    assert record.t_start == pytest.approx(0.0)
    assert record.t_end == pytest.approx(0.1)
    assert record.proposed_dt == pytest.approx(0.1)
    assert record.accepted_dt == pytest.approx(0.1)
    assert record.error_ratio == pytest.approx(0.0)
    assert record.rejection_count == 0
