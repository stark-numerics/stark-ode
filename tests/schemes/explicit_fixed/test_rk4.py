from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from stark import Executor, Interval
from stark.schemes.explicit_fixed.rk4 import SchemeRK4


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

        def stage2(stage: ScalarState, state: ScalarState, dt: float, k1: ScalarTranslation) -> None:
            stage.value = state.value + 0.5 * dt * k1.value

        def stage3(stage: ScalarState, state: ScalarState, dt: float, k2: ScalarTranslation) -> None:
            stage.value = state.value + 0.5 * dt * k2.value

        def stage4(stage: ScalarState, state: ScalarState, dt: float, k3: ScalarTranslation) -> None:
            stage.value = state.value + dt * k3.value

        def solution_state(
            result: ScalarState,
            origin: ScalarState,
            dt: float,
            k1: ScalarTranslation,
            k2: ScalarTranslation,
            k3: ScalarTranslation,
            k4: ScalarTranslation,
        ) -> None:
            origin_value = origin.value
            result.value = origin_value + dt * (
                (1.0 / 6.0) * k1.value
                + (1.0 / 3.0) * k2.value
                + (1.0 / 3.0) * k3.value
                + (1.0 / 6.0) * k4.value
            )

        return SimpleNamespace(
            stages=(None, stage2, stage3, stage4),
            solution_state=solution_state,
        )


def exponential_growth(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval
    out.value = state.value


def test_rk4_owns_its_public_call_method() -> None:
    assert "__call__" in SchemeRK4.__dict__


def test_rk4_default_call_path_is_scheme_owned_call_generic() -> None:
    scheme = SchemeRK4(exponential_growth, ScalarWorkbench())

    assert scheme.call_pure.__self__ is scheme
    assert scheme.call_pure.__func__ is SchemeRK4.call_generic
    assert scheme.redirect_call == scheme.call_pure


def test_rk4_public_call_uses_redirect_call() -> None:
    scheme = SchemeRK4(exponential_growth, ScalarWorkbench())
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


def test_rk4_call_generic_still_performs_one_rk4_step() -> None:
    scheme = SchemeRK4(exponential_growth, ScalarWorkbench())
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = ScalarState(1.0)

    accepted_dt = scheme(interval, state, Executor())

    assert accepted_dt == pytest.approx(0.125)
    assert state.value == pytest.approx(1.133148193359375)


def test_rk4_call_generic_clips_to_remaining_interval() -> None:
    scheme = SchemeRK4(exponential_growth, ScalarWorkbench())
    interval = Interval(present=0.2, step=0.125, stop=0.25)
    state = ScalarState(1.0)

    accepted_dt = scheme(interval, state, Executor())

    assert accepted_dt == pytest.approx(0.05)
    assert state.value == pytest.approx(1.05127109375)


def test_rk4_algebraist_path_is_selected_inside_scheme() -> None:
    scheme = SchemeRK4(
        exponential_growth,
        ScalarWorkbench(),
        algebraist=StubAlgebraist(),
    )

    assert scheme.call_pure.__self__ is scheme
    assert scheme.call_pure.__func__ is SchemeRK4.call_algebraist
    assert scheme.redirect_call == scheme.call_pure


def test_rk4_generic_and_algebraist_paths_match_for_one_step() -> None:
    interval_generic = Interval(present=0.0, step=0.125, stop=1.0)
    interval_algebraist = Interval(present=0.0, step=0.125, stop=1.0)
    state_generic = ScalarState(1.0)
    state_algebraist = ScalarState(1.0)

    generic = SchemeRK4(exponential_growth, ScalarWorkbench())
    algebraist = SchemeRK4(
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
    assert state_generic.value == pytest.approx(1.133148193359375)

def test_rk4_satisfies_public_scheme_contract_without_base_class_assertions() -> None:
    scheme = SchemeRK4(exponential_growth, ScalarWorkbench())

    assert callable(scheme)
    assert callable(scheme.snapshot_state)
    assert callable(scheme.set_apply_delta_safety)

    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = ScalarState(1.0)

    accepted_dt = scheme(interval, state, Executor())
    snapshot = scheme.snapshot_state(state)

    scheme.set_apply_delta_safety(False)
    scheme.set_apply_delta_safety(True)

    assert accepted_dt == pytest.approx(0.125)
    assert snapshot is not state
    assert snapshot.value == pytest.approx(state.value)


def test_rk4_exposes_copyable_fixed_explicit_scheme_shape() -> None:
    required_names = {
        "__call__",
        "call_generic",
        "initialise_buffers",
        "snapshot_state",
        "set_apply_delta_safety",
    }

    available_names = set(dir(SchemeRK4))

    assert required_names <= available_names

def test_rk4_satisfies_public_scheme_contract_without_base_class_assertions() -> None:
    scheme = SchemeRK4(exponential_growth, ScalarWorkbench())

    assert callable(scheme)
    assert callable(scheme.snapshot_state)
    assert callable(scheme.set_apply_delta_safety)

    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = ScalarState(1.0)

    accepted_dt = scheme(interval, state, Executor())
    snapshot = scheme.snapshot_state(state)

    scheme.set_apply_delta_safety(False)
    scheme.set_apply_delta_safety(True)

    assert accepted_dt == pytest.approx(0.125)
    assert snapshot is not state
    assert snapshot.value == pytest.approx(state.value)