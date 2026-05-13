from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from stark import Executor, Integrator, Interval, Marcher, Tolerance
from stark.monitor import Monitor
from stark.schemes.explicit_adaptive.bogacki_shampine import SchemeBogackiShampine


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


class StubAlgebraist:
    def bind_explicit_scheme(self, tableau):
        del tableau

        def stage2(
            stage: ScalarState,
            state: ScalarState,
            dt: float,
            k1: ScalarTranslation,
        ) -> None:
            stage.value = state.value + 0.5 * dt * k1.value

        def stage3(
            stage: ScalarState,
            state: ScalarState,
            dt: float,
            k2: ScalarTranslation,
        ) -> None:
            stage.value = state.value + 0.75 * dt * k2.value

        def stage4(
            stage: ScalarState,
            state: ScalarState,
            dt: float,
            k1: ScalarTranslation,
            k2: ScalarTranslation,
            k3: ScalarTranslation,
        ) -> None:
            stage.value = state.value + dt * (
                (2.0 / 9.0) * k1.value
                + (1.0 / 3.0) * k2.value
                + (4.0 / 9.0) * k3.value
            )

        def solution(
            out: ScalarTranslation,
            dt: float,
            k1: ScalarTranslation,
            k2: ScalarTranslation,
            k3: ScalarTranslation,
        ) -> ScalarTranslation:
            out.value = dt * (
                (2.0 / 9.0) * k1.value
                + (1.0 / 3.0) * k2.value
                + (4.0 / 9.0) * k3.value
            )
            return out

        def error(
            out: ScalarTranslation,
            dt: float,
            k1: ScalarTranslation,
            k2: ScalarTranslation,
            k3: ScalarTranslation,
            k4: ScalarTranslation,
        ) -> ScalarTranslation:
            out.value = dt * (
                ((2.0 / 9.0) - (7.0 / 24.0)) * k1.value
                + ((1.0 / 3.0) - 0.25) * k2.value
                + ((4.0 / 9.0) - (1.0 / 3.0)) * k3.value
                + (0.0 - 0.125) * k4.value
            )
            return out

        return SimpleNamespace(
            stage_state_calls=(None, stage2, stage3, stage4),
            solution_delta_call=solution,
            error_delta_call=error,
        )


def zero_rhs(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval, state
    out.value = 0.0


def exponential_growth(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval
    out.value = state.value


def tight_executor() -> Executor:
    return Executor(tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))


def test_bogacki_shampine_owns_its_public_call_method() -> None:
    assert "__call__" in SchemeBogackiShampine.__dict__


def test_bogacki_shampine_default_advance_path_is_scheme_owned_generic_advance() -> None:
    scheme = SchemeBogackiShampine(zero_rhs, ScalarWorkbench())

    assert scheme.call_pure.__self__ is scheme
    assert scheme.call_pure.__func__ is SchemeBogackiShampine.call_generic
    assert scheme.redirect_call.__self__ is scheme
    assert scheme.redirect_call.__self__ is scheme
    assert scheme.redirect_call.__func__ is scheme.call_bind.__func__

def test_bogacki_shampine_public_call_uses_redirect_call() -> None:
    scheme = SchemeBogackiShampine(zero_rhs, ScalarWorkbench())
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(2.0)

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


def test_bogacki_shampine_call_returns_accepted_dt_and_updates_next_step() -> None:
    scheme = SchemeBogackiShampine(zero_rhs, ScalarWorkbench())
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(2.0)

    accepted_dt = scheme(interval, state, tight_executor())

    assert accepted_dt == pytest.approx(0.1)
    assert interval.step == pytest.approx(0.2)
    assert state.value == pytest.approx(2.0)


def test_bogacki_shampine_call_clips_to_remaining_interval() -> None:
    scheme = SchemeBogackiShampine(zero_rhs, ScalarWorkbench())
    interval = Interval(present=0.1, step=1.0, stop=0.3)
    state = ScalarState(2.0)

    accepted_dt = scheme(interval, state, tight_executor())

    assert accepted_dt == pytest.approx(0.2)
    assert interval.step == pytest.approx(0.0)
    assert state.value == pytest.approx(2.0)


def test_bogacki_shampine_algebraist_path_is_selected_inside_scheme() -> None:
    scheme = SchemeBogackiShampine(
        exponential_growth,
        ScalarWorkbench(),
        algebraist=StubAlgebraist(),
    )

    assert scheme.call_pure.__self__ is scheme
    assert scheme.call_pure.__func__ is SchemeBogackiShampine.call_algebraist 


def test_bogacki_shampine_generic_and_algebraist_paths_match_for_one_step() -> None:
    interval_generic = Interval(present=0.0, step=0.1, stop=0.3)
    interval_algebraist = Interval(present=0.0, step=0.1, stop=0.3)
    state_generic = ScalarState(1.0)
    state_algebraist = ScalarState(1.0)

    generic = SchemeBogackiShampine(exponential_growth, ScalarWorkbench())
    algebraist = SchemeBogackiShampine(
        exponential_growth,
        ScalarWorkbench(),
        algebraist=StubAlgebraist(),
    )

    accepted_dt_generic = generic(interval_generic, state_generic, tight_executor())
    accepted_dt_algebraist = algebraist(
        interval_algebraist,
        state_algebraist,
        tight_executor(),
    )

    assert accepted_dt_generic == pytest.approx(accepted_dt_algebraist)
    assert state_generic.value == pytest.approx(state_algebraist.value)
    assert interval_generic.step == pytest.approx(interval_algebraist.step)


def test_bogacki_shampine_integration_matches_characterized_step_count() -> None:
    scheme = SchemeBogackiShampine(zero_rhs, ScalarWorkbench())
    marcher = Marcher(scheme, tight_executor())
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(2.0)

    outputs = list(Integrator().live(marcher, interval, state))

    assert len(outputs) == 2
    assert interval.present == pytest.approx(0.3)
    assert interval.step == pytest.approx(0.0)
    assert state.value == pytest.approx(2.0)


def test_bogacki_shampine_monitoring_records_existing_adaptive_fields() -> None:
    scheme = SchemeBogackiShampine(zero_rhs, ScalarWorkbench())
    marcher = Marcher(scheme, tight_executor())
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(2.0)
    monitor = Monitor()

    list(Integrator().live_monitored(marcher, interval, state, monitor))

    assert len(monitor.steps) == 2

    first = monitor.steps[0]
    second = monitor.steps[1]

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