from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from stark import Executor, Interval
from stark.schemes.explicit_fixed.euler import SchemeEuler


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

        def solution_state(
            result: ScalarState,
            origin: ScalarState,
            dt: float,
            k1: ScalarTranslation,
        ) -> None:
            result.value = origin.value + dt * k1.value

        return SimpleNamespace(
            stage_state_calls=(None,),
            solution_state_call=solution_state,
        )


def exponential_growth(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval
    out.value = state.value


def test_euler_owns_its_public_call_method() -> None:
    assert "__call__" in SchemeEuler.__dict__


def test_euler_default_call_path_is_scheme_owned_generic_call() -> None:
    scheme = SchemeEuler(exponential_growth, ScalarWorkbench())

    assert scheme.call_pure.__self__ is scheme
    assert scheme.call_pure.__func__ is SchemeEuler.call_generic
    assert scheme.redirect_call == scheme.call_pure


def test_euler_public_call_uses_redirect_call() -> None:
    scheme = SchemeEuler(exponential_growth, ScalarWorkbench())
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = ScalarState(1.0)

    def replacement_call(
        replacement_interval: Interval,
        replacement_state: ScalarState,
        replacement_executor: Executor,
    ) -> float:
        del replacement_interval, replacement_executor
        replacement_state.value = 42.0
        return 0.03125

    scheme.redirect_call = replacement_call

    accepted_dt = scheme(interval, state, Executor())

    assert accepted_dt == pytest.approx(0.03125)
    assert state.value == pytest.approx(42.0)


def test_euler_generic_call_performs_one_forward_euler_step() -> None:
    scheme = SchemeEuler(exponential_growth, ScalarWorkbench())
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = ScalarState(1.0)

    accepted_dt = scheme(interval, state, Executor())

    assert accepted_dt == pytest.approx(0.125)
    assert state.value == pytest.approx(1.125)


def test_euler_generic_call_clips_to_remaining_interval() -> None:
    scheme = SchemeEuler(exponential_growth, ScalarWorkbench())
    interval = Interval(present=0.2, step=0.125, stop=0.25)
    state = ScalarState(1.0)

    accepted_dt = scheme(interval, state, Executor())

    assert accepted_dt == pytest.approx(0.05)
    assert state.value == pytest.approx(1.05)


def test_euler_algebraist_path_is_selected_inside_scheme() -> None:
    scheme = SchemeEuler(
        exponential_growth,
        ScalarWorkbench(),
        algebraist=StubAlgebraist(),
    )

    assert scheme.call_pure.__self__ is scheme
    assert scheme.call_pure.__func__ is SchemeEuler.call_algebraist
    assert scheme.redirect_call == scheme.call_pure


def test_euler_generic_and_algebraist_paths_match_for_one_step() -> None:
    interval_generic = Interval(present=0.0, step=0.125, stop=1.0)
    interval_algebraist = Interval(present=0.0, step=0.125, stop=1.0)
    state_generic = ScalarState(1.0)
    state_algebraist = ScalarState(1.0)

    generic = SchemeEuler(exponential_growth, ScalarWorkbench())
    algebraist = SchemeEuler(
        exponential_growth,
        ScalarWorkbench(),
        algebraist=StubAlgebraist(),
    )

    accepted_dt_generic = generic(interval_generic, state_generic, Executor())
    accepted_dt_algebraist = algebraist(
        interval_algebraist,
        state_algebraist,
        Executor(),
    )

    assert accepted_dt_generic == pytest.approx(accepted_dt_algebraist)
    assert state_generic.value == pytest.approx(state_algebraist.value)
    assert state_generic.value == pytest.approx(1.125)