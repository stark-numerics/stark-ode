from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark import Executor, Integrator, Interval, Marcher, Tolerance
from stark.accelerators import Accelerator
from stark.monitor import Monitor
from stark.resolvents import ResolventPicard
from stark.resolvents.policy import ResolventPolicy
from stark.schemes.implicit_adaptive.bdf2 import SchemeBDF2


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


def constant_rhs(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval, state
    out.value = 1.0


def make_scheme() -> SchemeBDF2:
    workbench = ScalarWorkbench()
    resolvent = ResolventPicard(
        constant_rhs,
        workbench,
        tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=ResolventPolicy(max_iterations=8),
        accelerator=Accelerator.none(),
        tableau=None,
    )
    return SchemeBDF2(
        constant_rhs,
        workbench,
        resolvent=resolvent,
    )


def tight_executor() -> Executor:
    return Executor(tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))


def test_bdf2_owns_its_public_call_method() -> None:
    assert "__call__" in SchemeBDF2.__dict__


def test_bdf2_default_call_path_is_scheme_owned_generic_call() -> None:
    scheme = make_scheme()

    assert scheme.call_pure.__self__ is scheme
    assert scheme.call_pure.__func__ is SchemeBDF2.call_generic

    # Adaptive schemes bind executor runtime lazily on first public call.
    assert scheme.redirect_call.__self__ is scheme
    assert scheme.redirect_call.__func__ is scheme.call_bind.__func__


def test_bdf2_public_call_uses_redirect_call() -> None:
    scheme = make_scheme()
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(0.0)

    def replacement_call(
        replacement_interval: Interval,
        replacement_state: ScalarState,
        replacement_executor: Executor,
    ) -> float:
        del replacement_interval, replacement_executor
        replacement_state.value = 42.0
        return 0.03125

    scheme.redirect_call = replacement_call

    accepted_dt = scheme(interval, state, tight_executor())

    assert accepted_dt == pytest.approx(0.03125)
    assert state.value == pytest.approx(42.0)


def test_bdf2_startup_call_returns_accepted_dt_and_establishes_history() -> None:
    scheme = make_scheme()
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(0.0)

    accepted_dt = scheme(interval, state, tight_executor())

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
    executor = tight_executor()

    first_dt = scheme(interval, state, executor)
    interval.present += first_dt

    second_dt = scheme(interval, state, executor)

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
    executor = tight_executor()

    accepted_dt = scheme(interval, state, executor)
    interval.present += accepted_dt

    assert scheme.has_history is True
    assert scheme.previous_step == pytest.approx(0.1)

    interval.step = 0.25

    accepted_dt = scheme(interval, state, executor)

    assert accepted_dt == pytest.approx(0.25)
    assert state.value == pytest.approx(0.35)
    assert scheme.previous_step == pytest.approx(0.25)
    assert scheme.previous_delta.value == pytest.approx(0.25)


def test_bdf2_call_clips_to_remaining_interval() -> None:
    scheme = make_scheme()
    interval = Interval(present=0.1, step=1.0, stop=0.3)
    state = ScalarState(0.0)

    accepted_dt = scheme(interval, state, tight_executor())

    assert accepted_dt == pytest.approx(0.2)
    assert state.value == pytest.approx(0.2)
    assert interval.step == pytest.approx(0.0)


def test_bdf2_returns_zero_when_interval_is_complete() -> None:
    scheme = make_scheme()
    interval = Interval(present=1.0, step=0.1, stop=1.0)
    state = ScalarState(0.0)

    accepted_dt = scheme(interval, state, tight_executor())

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

    scheme.set_apply_delta_safety(False)
    scheme.set_apply_delta_safety(True)


def test_bdf2_monitoring_records_existing_adaptive_fields() -> None:
    scheme = make_scheme()
    marcher = Marcher(scheme, tight_executor())
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(0.0)
    monitor = Monitor()

    list(Integrator().live_monitored(marcher, interval, state, monitor))

    assert len(monitor.steps) == 2

    first = monitor.steps[0]
    second = monitor.steps[1]

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