from __future__ import annotations

import pytest

from stark import Interval, Tolerance
from stark.core import Configuration, Integrator, IntegratorStepper
from stark.core.contracts import IntervalLike
from stark.diagnostics.monitor import Monitor
from stark.methods.schemes.explicit.adaptive.cash_karp import SchemeCashKarp
from stark.methods.schemes.explicit.adaptive.fehlberg45 import SchemeFehlberg45
from tests.support import (
    DummyScalarAllocator,
    DummyScalarState,
    DummyTableauLinearFixed,
    dummy_exponential_growth_rhs,
    dummy_zero_rhs,
)


def tight_configuration() -> Configuration:
    return Configuration(scheme_tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))


@pytest.mark.parametrize("scheme_cls", [SchemeCashKarp, SchemeFehlberg45])
def test_cash_karp_fehlberg_scheme_owns_its_public_call_method(scheme_cls) -> None:
    assert "__call__" in scheme_cls.__dict__


@pytest.mark.parametrize("scheme_cls", [SchemeCashKarp, SchemeFehlberg45])
def test_cash_karp_fehlberg_public_call_uses_redirect_call(scheme_cls) -> None:
    scheme = scheme_cls(dummy_zero_rhs, DummyScalarAllocator())
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = DummyScalarState(2.0)

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


@pytest.mark.parametrize("scheme_cls", [SchemeCashKarp, SchemeFehlberg45])
def test_cash_karp_fehlberg_call_returns_accepted_dt_and_updates_next_step(
    scheme_cls,
) -> None:
    scheme = scheme_cls(dummy_zero_rhs, DummyScalarAllocator())
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = DummyScalarState(2.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.1)
    assert interval.step == pytest.approx(0.2)
    assert state.value == pytest.approx(2.0)


@pytest.mark.parametrize("scheme_cls", [SchemeCashKarp, SchemeFehlberg45])
def test_cash_karp_fehlberg_call_clips_to_remaining_interval(scheme_cls) -> None:
    scheme = scheme_cls(dummy_zero_rhs, DummyScalarAllocator())
    interval = Interval(present=0.1, step=1.0, stop=0.3)
    state = DummyScalarState(2.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.2)
    assert interval.step == pytest.approx(0.0)
    assert state.value == pytest.approx(2.0)


@pytest.mark.parametrize("scheme_cls", [SchemeCashKarp, SchemeFehlberg45])
def test_cash_karp_fehlberg_inline_and_linear_fixed_paths_match_for_one_step(
    scheme_cls,
) -> None:
    interval_inline = Interval(present=0.0, step=0.1, stop=0.3)
    interval_linear_fixed = Interval(present=0.0, step=0.1, stop=0.3)
    state_inline = DummyScalarState(1.0)
    state_linear_fixed = DummyScalarState(1.0)

    inline = scheme_cls(dummy_exponential_growth_rhs, DummyScalarAllocator())
    linear_fixed = scheme_cls(
        dummy_exponential_growth_rhs,
        DummyScalarAllocator(),
        linear_fixed=DummyTableauLinearFixed(),
    )

    accepted_dt_inline = inline(interval_inline, state_inline)
    accepted_dt_linear_fixed = linear_fixed(
        interval_linear_fixed,
        state_linear_fixed,
    )

    assert accepted_dt_linear_fixed == pytest.approx(accepted_dt_inline)
    assert state_linear_fixed.value == pytest.approx(state_inline.value)
    assert interval_linear_fixed.step == pytest.approx(interval_inline.step)


@pytest.mark.parametrize(
    ("scheme_cls", "scheme_name"),
    [(SchemeCashKarp, "RKCK"), (SchemeFehlberg45, "RKF45")],
)
def test_cash_karp_fehlberg_monitoring_records_existing_adaptive_fields(
    scheme_cls,
    scheme_name: str,
) -> None:
    monitor = Monitor()
    scheme = scheme_cls(dummy_zero_rhs, DummyScalarAllocator(), monitor=monitor.scheme)
    stepper = IntegratorStepper(scheme)
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = DummyScalarState(2.0)

    list(Integrator().mutating_trajectory(stepper, interval, state))

    assert len(monitor.scheme.adaptive_steps) == 2
    first = monitor.scheme.adaptive_steps[0]
    second = monitor.scheme.adaptive_steps[1]

    assert first.scheme == scheme_name
    assert first.t_start == pytest.approx(0.0)
    assert first.t_end == pytest.approx(0.1)
    assert first.proposed_dt == pytest.approx(0.1)
    assert first.accepted_dt == pytest.approx(0.1)
    assert first.next_dt == pytest.approx(0.2)
    assert first.error_ratio == pytest.approx(0.0)
    assert first.rejection_count == 0

    assert second.scheme == scheme_name
    assert second.t_start == pytest.approx(0.1)
    assert second.t_end == pytest.approx(0.3)
    assert second.proposed_dt == pytest.approx(0.2)
    assert second.accepted_dt == pytest.approx(0.2)
    assert second.next_dt == pytest.approx(0.0)
    assert second.error_ratio == pytest.approx(0.0)
    assert second.rejection_count == 0
