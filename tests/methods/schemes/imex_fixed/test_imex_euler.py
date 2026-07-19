from __future__ import annotations

import pytest

from stark import Configuration, Dynamics, Interval, Tolerance
from stark.core.contracts import IntervalLike
from stark.engines.accelerators import AcceleratorNone
from stark.methods.resolvents import ResolventPicard
from stark.methods.schemes.imex.fixed.euler import SchemeIMEXEuler
from stark.methods.schemes.linear_fixed_generation.linear_fixed import SchemeLinearFixedLike
from tests.support import (
    DummyScalarAllocator,
    DummyScalarState,
    DummyScalarTranslation,
    DummyTableauLinearFixed,
    dummy_constant_dynamics,
)


class DummyLinearDynamics:
    """Scalar dynamics fixture for the implicit linear solve check."""

    def __init__(self, rate: float) -> None:
        self.rate = rate

    def __call__(
        self,
        interval: IntervalLike,
        state: DummyScalarState,
        out: DummyScalarTranslation,
    ) -> None:
        del interval
        out.value = self.rate * state.value


def make_resolvent(
    allocator: DummyScalarAllocator,
) -> ResolventPicard:
    return ResolventPicard(
        allocator,
        configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12), resolvent_maximum_steps=32),
        accelerator=AcceleratorNone(),
        tableau=SchemeIMEXEuler.tableau,
    )


def make_constant_scheme(
    explicit_value: float = 1.0,
    implicit_value: float = 2.0,
    *,
    linear_fixed: SchemeLinearFixedLike[DummyScalarState, DummyScalarTranslation] | None = None,
) -> SchemeIMEXEuler:
    allocator = DummyScalarAllocator()
    dynamics = Dynamics.split(
        explicit=dummy_constant_dynamics(explicit_value),
        implicit=dummy_constant_dynamics(implicit_value),
    )
    return SchemeIMEXEuler(
        dynamics,
        allocator,
        resolvent=make_resolvent(allocator),
        linear_fixed=linear_fixed,
    )


def test_imex_euler_owns_its_public_call_method() -> None:
    assert "__call__" in SchemeIMEXEuler.__dict__


def test_imex_euler_default_call_path_is_scheme_owned_generic_call() -> None:
    scheme = make_constant_scheme()

    assert scheme.redirect_call == scheme.call_step


def test_imex_euler_linear_fixed_path_is_scheme_owned_generated_call() -> None:
    scheme = make_constant_scheme(linear_fixed=DummyTableauLinearFixed())

    assert scheme.redirect_call == scheme.call_step


def test_imex_euler_linear_fixed_path_matches_generic_path() -> None:
    generic = make_constant_scheme()
    specialized = make_constant_scheme(linear_fixed=DummyTableauLinearFixed())
    generic_interval = Interval(present=0.0, step=0.125, stop=1.0)
    specialized_interval = Interval(present=0.0, step=0.125, stop=1.0)
    generic_state = DummyScalarState(0.0)
    specialized_state = DummyScalarState(0.0)

    generic_dt = generic(generic_interval, generic_state)
    specialized_dt = specialized(specialized_interval, specialized_state)

    assert specialized_dt == pytest.approx(generic_dt)
    assert specialized_state.value == pytest.approx(generic_state.value)


def test_imex_euler_public_call_uses_redirect_call() -> None:
    scheme = make_constant_scheme()
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = DummyScalarState(0.0)

    def replacement_call(
        interval: IntervalLike,
        state: DummyScalarState,
    ) -> float:
        del interval
        state.value = 42.0
        return 0.03125

    scheme.redirect_call = replacement_call

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.03125)
    assert state.value == pytest.approx(42.0)


def test_imex_euler_call_performs_one_split_constant_rhs_step() -> None:
    scheme = make_constant_scheme(explicit_value=1.0, implicit_value=2.0)
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = DummyScalarState(0.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.125)
    assert state.value == pytest.approx(0.375)
    assert interval.step == pytest.approx(0.125)


def test_imex_euler_call_clips_to_remaining_interval() -> None:
    scheme = make_constant_scheme(explicit_value=1.0, implicit_value=2.0)
    interval = Interval(present=0.2, step=0.125, stop=0.25)
    state = DummyScalarState(0.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.05)
    assert state.value == pytest.approx(0.15)
    assert interval.step == pytest.approx(0.0)


def test_imex_euler_returns_zero_when_interval_is_complete() -> None:
    scheme = make_constant_scheme()
    interval = Interval(present=1.0, step=0.125, stop=1.0)
    state = DummyScalarState(0.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.0)
    assert state.value == pytest.approx(0.0)


def test_imex_euler_solves_linear_implicit_split() -> None:
    allocator = DummyScalarAllocator()
    implicit = DummyLinearDynamics(rate=-1.0)
    dynamics = Dynamics.split(
        explicit=dummy_constant_dynamics(0.0),
        implicit=implicit,
    )
    scheme = SchemeIMEXEuler(
        dynamics,
        allocator,
        resolvent=make_resolvent(allocator),
    )
    interval = Interval(present=0.0, step=0.1, stop=1.0)
    state = DummyScalarState(1.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.1)
    assert state.value == pytest.approx(1.0 / 1.1)


def test_imex_euler_snapshot_is_exposed_through_scheme() -> None:
    scheme = make_constant_scheme()
    state = DummyScalarState(3.0)

    snapshot = scheme.snapshot_state(state)

    assert snapshot is not state
    assert snapshot.value == pytest.approx(3.0)

    state.value = 9.0

    assert snapshot.value == pytest.approx(3.0)
