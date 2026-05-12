from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark import Executor, Interval, Tolerance
from stark.accelerators import Accelerator
from stark.resolvents import ResolventPicard
from stark.resolvents.policy import ResolventPolicy
from stark.schemes.imex_fixed.euler import SchemeIMEXEuler


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


class ConstantDerivative:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(
        self,
        interval: Interval,
        state: ScalarState,
        out: ScalarTranslation,
    ) -> None:
        del interval, state
        out.value = self.value


class LinearDerivative:
    def __init__(self, rate: float) -> None:
        self.rate = rate

    def __call__(
        self,
        interval: Interval,
        state: ScalarState,
        out: ScalarTranslation,
    ) -> None:
        del interval
        out.value = self.rate * state.value


@dataclass(slots=True)
class SplitDerivative:
    explicit: object
    implicit: object


def make_resolvent(
    implicit_derivative,
    workbench: ScalarWorkbench,
) -> ResolventPicard:
    return ResolventPicard(
        implicit_derivative,
        workbench,
        tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=ResolventPolicy(max_iterations=32),
        accelerator=Accelerator.none(),
        tableau=SchemeIMEXEuler.tableau,
    )


def make_constant_scheme(
    explicit_value: float = 1.0,
    implicit_value: float = 2.0,
) -> SchemeIMEXEuler:
    workbench = ScalarWorkbench()
    implicit = ConstantDerivative(implicit_value)
    derivative = SplitDerivative(
        explicit=ConstantDerivative(explicit_value),
        implicit=implicit,
    )
    return SchemeIMEXEuler(
        derivative,
        workbench,
        resolvent=make_resolvent(implicit, workbench),
    )


def test_imex_euler_owns_its_public_call_method() -> None:
    assert "__call__" in SchemeIMEXEuler.__dict__


def test_imex_euler_default_call_path_is_scheme_owned_generic_call() -> None:
    scheme = make_constant_scheme()

    assert scheme.call_pure.__self__ is scheme
    assert scheme.call_pure.__func__ is SchemeIMEXEuler.call_generic
    assert scheme.redirect_call == scheme.call_pure


def test_imex_euler_public_call_uses_redirect_call() -> None:
    scheme = make_constant_scheme()
    interval = Interval(present=0.0, step=0.125, stop=1.0)
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

    accepted_dt = scheme(interval, state, Executor())

    assert accepted_dt == pytest.approx(0.03125)
    assert state.value == pytest.approx(42.0)


def test_imex_euler_call_performs_one_split_constant_rhs_step() -> None:
    scheme = make_constant_scheme(explicit_value=1.0, implicit_value=2.0)
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = ScalarState(0.0)

    accepted_dt = scheme(interval, state, Executor())

    assert accepted_dt == pytest.approx(0.125)
    assert state.value == pytest.approx(0.375)
    assert interval.step == pytest.approx(0.125)


def test_imex_euler_call_clips_to_remaining_interval() -> None:
    scheme = make_constant_scheme(explicit_value=1.0, implicit_value=2.0)
    interval = Interval(present=0.2, step=0.125, stop=0.25)
    state = ScalarState(0.0)

    accepted_dt = scheme(interval, state, Executor())

    assert accepted_dt == pytest.approx(0.05)
    assert state.value == pytest.approx(0.15)
    assert interval.step == pytest.approx(0.0)


def test_imex_euler_returns_zero_when_interval_is_complete() -> None:
    scheme = make_constant_scheme()
    interval = Interval(present=1.0, step=0.125, stop=1.0)
    state = ScalarState(0.0)

    accepted_dt = scheme(interval, state, Executor())

    assert accepted_dt == pytest.approx(0.0)
    assert state.value == pytest.approx(0.0)


def test_imex_euler_solves_linear_implicit_split() -> None:
    workbench = ScalarWorkbench()
    implicit = LinearDerivative(rate=-1.0)
    derivative = SplitDerivative(
        explicit=ConstantDerivative(0.0),
        implicit=implicit,
    )
    scheme = SchemeIMEXEuler(
        derivative,
        workbench,
        resolvent=make_resolvent(implicit, workbench),
    )
    interval = Interval(present=0.0, step=0.1, stop=1.0)
    state = ScalarState(1.0)

    accepted_dt = scheme(interval, state, Executor())

    assert accepted_dt == pytest.approx(0.1)
    assert state.value == pytest.approx(1.0 / 1.1)

def test_imex_euler_snapshot_and_safety_are_exposed_through_scheme() -> None:
    scheme = make_constant_scheme()
    state = ScalarState(3.0)

    snapshot = scheme.snapshot_state(state)

    assert snapshot is not state
    assert snapshot.value == pytest.approx(3.0)

    state.value = 9.0

    assert snapshot.value == pytest.approx(3.0)

    scheme.set_apply_delta_safety(False)
    scheme.set_apply_delta_safety(True)