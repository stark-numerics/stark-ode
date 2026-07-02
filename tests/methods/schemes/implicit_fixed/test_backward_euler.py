from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark import Interval, Tolerance
from stark.engines.shared.accelerators import AcceleratorNone
from stark.methods.resolvents import ResolventPicard
from stark import Configuration
from stark.methods.schemes.implicit.fixed.backward_euler import SchemeBackwardEuler


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


def make_scheme() -> SchemeBackwardEuler:
    allocator = ScalarAllocator()
    resolvent = ResolventPicard(
        allocator,
        configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12), resolvent_maximum_steps=8),
        accelerator=AcceleratorNone(),
        tableau=SchemeBackwardEuler.tableau,
    )
    return SchemeBackwardEuler(
        constant_rhs,
        allocator,
        resolvent=resolvent,
    )


def test_backward_euler_owns_its_public_call_method() -> None:
    assert "__call__" in SchemeBackwardEuler.__dict__


def test_backward_euler_default_call_path_is_scheme_owned_generic_call() -> None:
    scheme = make_scheme()

    assert scheme.redirect_call == scheme.call_step


def test_backward_euler_public_call_uses_redirect_call() -> None:
    scheme = make_scheme()
    interval = Interval(present=0.0, step=0.125, stop=1.0)
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


def test_backward_euler_generic_call_performs_one_implicit_step() -> None:
    scheme = make_scheme()
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = ScalarState(0.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.125)
    assert state.value == pytest.approx(0.125)
    assert interval.step == pytest.approx(0.125)


def test_backward_euler_generic_call_clips_to_remaining_interval() -> None:
    scheme = make_scheme()
    interval = Interval(present=0.2, step=0.125, stop=0.25)
    state = ScalarState(0.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.05)
    assert state.value == pytest.approx(0.05)
    assert interval.step == pytest.approx(0.0)


def test_backward_euler_returns_zero_when_interval_is_already_complete() -> None:
    scheme = make_scheme()
    interval = Interval(present=1.0, step=0.125, stop=1.0)
    state = ScalarState(0.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.0)
    assert state.value == pytest.approx(0.0)


def test_backward_euler_snapshot_is_exposed_through_scheme() -> None:
    scheme = make_scheme()
    state = ScalarState(3.0)

    snapshot = scheme.snapshot_state(state)

    assert snapshot is not state
    assert snapshot.value == pytest.approx(3.0)

    state.value = 9.0

    assert snapshot.value == pytest.approx(3.0)
