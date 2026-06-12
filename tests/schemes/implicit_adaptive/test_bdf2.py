from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark import Integrator, Interval, IntegratorStepper, Tolerance
from stark.engines.accelerators import AcceleratorNone
from stark.monitor import Monitor
from stark.methods.resolvents import ResolventPicard
from stark import Configuration
from stark.methods.schemes.implicit.adaptive.bdf2 import SchemeBDF2


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


def constant_rhs(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval, state
    out.value = 1.0


def make_scheme(
    *,
    specialist: object | None = None,
    monitor=None,
) -> SchemeBDF2:
    allocator = ScalarAllocator()
    resolvent = ResolventPicard(
        allocator,
        configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12), resolvent_maximum_steps=8),
        accelerator=AcceleratorNone(),
        tableau=None,
    )
    return SchemeBDF2(
        constant_rhs,
        allocator,
        resolvent=resolvent,
        specialist=specialist,
        monitor=monitor,
    )


def tight_configuration() -> Configuration:
    return Configuration(scheme_tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))


def test_bdf2_owns_its_public_call_method() -> None:
    assert "__call__" in SchemeBDF2.__dict__


def test_bdf2_default_call_path_is_scheme_owned_generic_call() -> None:
    scheme = make_scheme()

    assert scheme.call_step.__self__ is scheme
    assert scheme.call_step.__func__ is SchemeBDF2.call_inline

    assert scheme.redirect_call.__self__ is scheme
    assert scheme.redirect_call.__func__ is scheme.call_step.__func__


def test_bdf2_accepts_specialist_but_remains_generic_only() -> None:
    scheme = make_scheme(specialist=object())

    assert scheme.call_step.__self__ is scheme
    assert scheme.call_step.__func__ is SchemeBDF2.call_inline

    assert scheme.redirect_call.__self__ is scheme
    assert scheme.redirect_call.__func__ is scheme.call_step.__func__


def test_bdf2_specialist_hook_keeps_inline_call_path() -> None:
    scheme = make_scheme(specialist=object())

    assert scheme.call_step.__func__ is SchemeBDF2.call_inline


def test_bdf2_public_call_uses_redirect_call() -> None:
    scheme = make_scheme()
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(0.0)

    def replacement_call(
        replacement_interval: Interval,
        replacement_state: ScalarState,
    ) -> float:
        del replacement_interval
        replacement_state.value = 42.0
        return 0.03125

    scheme.redirect_call = replacement_call

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.03125)
    assert state.value == pytest.approx(42.0)


def test_bdf2_startup_call_returns_accepted_dt_and_establishes_history() -> None:
    scheme = make_scheme()
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(0.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.1)
    assert state.value == pytest.approx(0.1)
    assert interval.step == pytest.approx(0.2)
    assert scheme.has_history is True
    assert scheme.previous_step == pytest.approx(0.1)
    assert scheme.previous_delta.value == pytest.approx(0.1)


def test_bdf2_second_call_uses_history_and_advances_constant_rhs() -> None:
    scheme = make_scheme()
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(0.0)
    configuration = tight_configuration()

    first_dt = scheme(interval, state)
    interval.present += first_dt

    second_dt = scheme(interval, state)

    assert second_dt == pytest.approx(0.2)
    assert state.value == pytest.approx(0.3)
    assert interval.step == pytest.approx(0.0)
    assert scheme.has_history is True
    assert scheme.previous_step == pytest.approx(0.2)
    assert scheme.previous_delta.value == pytest.approx(0.2)


def test_bdf2_falls_back_to_startup_when_step_ratio_is_too_large() -> None:
    scheme = make_scheme()
    interval = Interval(present=0.0, step=0.1, stop=1.0)
    state = ScalarState(0.0)
    configuration = tight_configuration()

    accepted_dt = scheme(interval, state)
    interval.present += accepted_dt

    assert scheme.has_history is True
    assert scheme.previous_step == pytest.approx(0.1)

    interval.step = 0.25

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.25)
    assert state.value == pytest.approx(0.35)
    assert scheme.previous_step == pytest.approx(0.25)
    assert scheme.previous_delta.value == pytest.approx(0.25)


def test_bdf2_call_clips_to_remaining_interval() -> None:
    scheme = make_scheme()
    interval = Interval(present=0.1, step=1.0, stop=0.3)
    state = ScalarState(0.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.2)
    assert state.value == pytest.approx(0.2)
    assert interval.step == pytest.approx(0.0)


def test_bdf2_returns_zero_when_interval_is_complete() -> None:
    scheme = make_scheme()
    interval = Interval(present=1.0, step=0.1, stop=1.0)
    state = ScalarState(0.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.0)
    assert state.value == pytest.approx(0.0)


def test_bdf2_snapshot_and_safety_are_exposed_through_scheme() -> None:
    scheme = make_scheme()
    state = ScalarState(3.0)

    snapshot = scheme.snapshot_state(state)

    assert snapshot is not state
    assert snapshot.value == pytest.approx(3.0)

    state.value = 9.0

    assert snapshot.value == pytest.approx(3.0)

def test_bdf2_monitoring_records_existing_adaptive_fields() -> None:
    monitor = Monitor()
    scheme = make_scheme(monitor=monitor.scheme)
    stepper = IntegratorStepper(scheme)
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(0.0)

    list(Integrator().mutating_trajectory(stepper, interval, state))

    assert len(monitor.scheme.adaptive_steps) == 2

    first = monitor.scheme.adaptive_steps[0]
    second = monitor.scheme.adaptive_steps[1]

    assert first.scheme == "BDF2"
    assert first.t_start == pytest.approx(0.0)
    assert first.t_end == pytest.approx(0.1)
    assert first.proposed_dt == pytest.approx(0.1)
    assert first.accepted_dt == pytest.approx(0.1)
    assert first.next_dt == pytest.approx(0.2)
    assert first.error_ratio >= 0.0
    assert first.error_ratio < 1.0e-6
    assert first.rejection_count == 0

    assert second.scheme == "BDF2"
    assert second.t_start == pytest.approx(0.1)
    assert second.t_end == pytest.approx(0.3)
    assert second.proposed_dt == pytest.approx(0.2)
    assert second.accepted_dt == pytest.approx(0.2)
    assert second.next_dt == pytest.approx(0.0)
    assert second.error_ratio >= 0.0
    assert second.error_ratio < 1.0e-6
    assert second.rejection_count == 0
