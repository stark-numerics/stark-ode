from __future__ import annotations

import pytest

from stark import Interval
from stark.core.contracts import IntervalLike
from stark.diagnostics.monitor import Monitor
from stark.methods.schemes.explicit.fixed.euler import SchemeEuler
from tests.support import (
    DummyScalarAllocator,
    DummyScalarState,
    DummyTableauLinearFixed,
    dummy_exponential_growth_rhs,
)


def test_euler_owns_its_public_call_method() -> None:
    assert "__call__" in SchemeEuler.__dict__


def test_euler_default_call_path_is_scheme_owned_inline_call() -> None:
    scheme = SchemeEuler(dummy_exponential_growth_rhs, DummyScalarAllocator())

    assert scheme.redirect_call == scheme.call_step


def test_euler_public_call_uses_redirect_call() -> None:
    scheme = SchemeEuler(dummy_exponential_growth_rhs, DummyScalarAllocator())
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = DummyScalarState(1.0)

    def replacement_call(
        replacement_interval: IntervalLike,
        replacement_state: DummyScalarState,
    ) -> float:
        del replacement_interval
        replacement_state.value = 42.0
        return 0.03125

    scheme.redirect_call = replacement_call

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.03125)
    assert state.value == pytest.approx(42.0)


def test_euler_inline_call_performs_one_forward_euler_step() -> None:
    scheme = SchemeEuler(dummy_exponential_growth_rhs, DummyScalarAllocator())
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = DummyScalarState(1.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.125)
    assert state.value == pytest.approx(1.125)


def test_euler_inline_call_clips_to_remaining_interval() -> None:
    scheme = SchemeEuler(dummy_exponential_growth_rhs, DummyScalarAllocator())
    interval = Interval(present=0.2, step=0.125, stop=0.25)
    state = DummyScalarState(1.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.05)
    assert state.value == pytest.approx(1.05)


def test_euler_linear_fixed_path_is_selected_inside_scheme() -> None:
    scheme = SchemeEuler(
        dummy_exponential_growth_rhs,
        DummyScalarAllocator(),
        linear_fixed=DummyTableauLinearFixed(),
    )

    assert scheme.redirect_call == scheme.call_step


def test_euler_monitoring_records_fixed_step_without_changing_pure_path() -> None:
    monitor = Monitor()
    scheme = SchemeEuler(dummy_exponential_growth_rhs, DummyScalarAllocator(), monitor=monitor.scheme)
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = DummyScalarState(1.0)

    assert scheme.monitor is monitor.scheme
    assert scheme.redirect_call == scheme.call_step

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.125)
    assert state.value == pytest.approx(1.125)
    assert monitor.scheme.adaptive_steps == []
    assert len(monitor.scheme.fixed_steps) == 1

    step = monitor.scheme.fixed_steps[0]
    assert step.scheme == "Euler"
    assert step.t_start == pytest.approx(0.0)
    assert step.t_end == pytest.approx(0.125)
    assert step.accepted_dt == pytest.approx(0.125)

def test_euler_monitoring_records_linear_fixed_fixed_step() -> None:
    monitor = Monitor()
    scheme = SchemeEuler(
        dummy_exponential_growth_rhs,
        DummyScalarAllocator(),
        linear_fixed=DummyTableauLinearFixed(),
        monitor=monitor.scheme,
    )
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = DummyScalarState(1.0)

    assert scheme.monitor is monitor.scheme
    assert scheme.redirect_call == scheme.call_step

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.125)
    assert state.value == pytest.approx(1.125)
    assert len(monitor.scheme.fixed_steps) == 1
    assert monitor.scheme.fixed_steps[0].scheme == "Euler"


def test_euler_inline_and_linear_fixed_paths_match_for_one_step() -> None:
    interval_inline = Interval(present=0.0, step=0.125, stop=1.0)
    interval_linear_fixed = Interval(present=0.0, step=0.125, stop=1.0)
    state_inline = DummyScalarState(1.0)
    state_linear_fixed = DummyScalarState(1.0)

    inline = SchemeEuler(dummy_exponential_growth_rhs, DummyScalarAllocator())
    linear_fixed = SchemeEuler(
        dummy_exponential_growth_rhs,
        DummyScalarAllocator(),
        linear_fixed=DummyTableauLinearFixed(),
    )

    accepted_dt_inline = inline(interval_inline, state_inline)
    accepted_dt_linear_fixed = linear_fixed(
        interval_linear_fixed,
        state_linear_fixed,
    )

    assert accepted_dt_inline == pytest.approx(accepted_dt_linear_fixed)
    assert state_inline.value == pytest.approx(state_linear_fixed.value)
    assert state_inline.value == pytest.approx(1.125)

