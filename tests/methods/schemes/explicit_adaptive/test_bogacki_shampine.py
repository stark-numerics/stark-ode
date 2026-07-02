from __future__ import annotations

import pytest

from stark import Interval, Tolerance
from stark.core import Configuration, Integrator, IntegratorStepper
from stark.core.contracts import IntervalLike
from stark.diagnostics.monitor import Monitor
from stark.methods.schemes.explicit.adaptive.bogacki_shampine import SchemeBogackiShampine
from tests.support import (
    DummyScalarAllocator,
    DummyScalarState,
    DummyTableauSpecialist,
    dummy_exponential_growth_rhs,
    dummy_zero_rhs,
)


def tight_configuration() -> Configuration:
    return Configuration(scheme_tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))


def test_bogacki_shampine_owns_its_public_call_method() -> None:
    assert "__call__" in SchemeBogackiShampine.__dict__


def test_bogacki_shampine_public_call_uses_redirect_call() -> None:
    scheme = SchemeBogackiShampine(dummy_zero_rhs, DummyScalarAllocator())
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


def test_bogacki_shampine_call_returns_accepted_dt_and_updates_next_step() -> None:
    scheme = SchemeBogackiShampine(dummy_zero_rhs, DummyScalarAllocator())
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = DummyScalarState(2.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.1)
    assert interval.step == pytest.approx(0.2)
    assert state.value == pytest.approx(2.0)


def test_bogacki_shampine_call_clips_to_remaining_interval() -> None:
    scheme = SchemeBogackiShampine(dummy_zero_rhs, DummyScalarAllocator())
    interval = Interval(present=0.1, step=1.0, stop=0.3)
    state = DummyScalarState(2.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.2)
    assert interval.step == pytest.approx(0.0)
    assert state.value == pytest.approx(2.0)


def test_bogacki_shampine_inline_and_specialist_paths_match_for_one_step() -> None:
    interval_inline = Interval(present=0.0, step=0.1, stop=0.3)
    interval_specialist = Interval(present=0.0, step=0.1, stop=0.3)
    state_inline = DummyScalarState(1.0)
    state_specialist = DummyScalarState(1.0)

    inline = SchemeBogackiShampine(dummy_exponential_growth_rhs, DummyScalarAllocator())
    specialist = SchemeBogackiShampine(
        dummy_exponential_growth_rhs,
        DummyScalarAllocator(),
        specialist=DummyTableauSpecialist(),
    )

    accepted_dt_inline = inline(interval_inline, state_inline)
    accepted_dt_specialist = specialist(
        interval_specialist,
        state_specialist,
    )

    assert accepted_dt_specialist == pytest.approx(accepted_dt_inline)
    assert state_specialist.value == pytest.approx(state_inline.value)
    assert interval_specialist.step == pytest.approx(interval_inline.step)


def test_bogacki_shampine_integration_matches_characterized_step_count() -> None:
    scheme = SchemeBogackiShampine(dummy_zero_rhs, DummyScalarAllocator())
    stepper = IntegratorStepper(scheme)
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = DummyScalarState(2.0)

    outputs = list(Integrator().mutating_trajectory(stepper, interval, state))

    assert len(outputs) == 2
    assert interval.present == pytest.approx(0.3)
    assert interval.step == pytest.approx(0.0)
    assert state.value == pytest.approx(2.0)


def test_bogacki_shampine_monitoring_records_existing_adaptive_fields() -> None:
    monitor = Monitor()
    scheme = SchemeBogackiShampine(dummy_zero_rhs, DummyScalarAllocator(), monitor=monitor.scheme)
    stepper = IntegratorStepper(scheme)
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = DummyScalarState(2.0)

    list(Integrator().mutating_trajectory(stepper, interval, state))

    assert len(monitor.scheme.adaptive_steps) == 2
    first = monitor.scheme.adaptive_steps[0]
    second = monitor.scheme.adaptive_steps[1]

    assert first.scheme == "BS23"
    assert first.t_start == pytest.approx(0.0)
    assert first.t_end == pytest.approx(0.1)
    assert first.proposed_dt == pytest.approx(0.1)
    assert first.accepted_dt == pytest.approx(0.1)
    assert first.next_dt == pytest.approx(0.2)
    assert first.error_ratio == pytest.approx(0.0)
    assert first.rejection_count == 0

    assert second.scheme == "BS23"
    assert second.t_start == pytest.approx(0.1)
    assert second.t_end == pytest.approx(0.3)
    assert second.proposed_dt == pytest.approx(0.2)
    assert second.accepted_dt == pytest.approx(0.2)
    assert second.next_dt == pytest.approx(0.0)
    assert second.error_ratio == pytest.approx(0.0)
    assert second.rejection_count == 0

