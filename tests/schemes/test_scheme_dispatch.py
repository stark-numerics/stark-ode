from __future__ import annotations

from dataclasses import dataclass
import warnings

import pytest

from stark import Executor, Interval, Tolerance
from stark.resolvents import ResolventPicard
from stark.resolvents.policy import ResolventPolicy
from stark.accelerators import Accelerator
from stark.schemes.explicit_adaptive.bogacki_shampine import SchemeBogackiShampine
from stark.schemes.explicit_fixed.euler import SchemeEuler
from stark.schemes.explicit_fixed.heun import SchemeHeun
from stark.schemes.explicit_fixed.rk4 import SchemeRK4
from stark.schemes.implicit_fixed.backward_euler import SchemeBackwardEuler


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


def exponential_growth(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval
    out.value = state.value


def constant_rhs(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval, state
    out.value = 1.0


def zero_rhs(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval, state
    out.value = 0.0


def make_backward_euler() -> SchemeBackwardEuler:
    workbench = ScalarWorkbench()
    resolvent = ResolventPicard(
        constant_rhs,
        workbench,
        tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=ResolventPolicy(max_iterations=8),
        accelerator=Accelerator.none(),
        tableau=SchemeBackwardEuler.tableau,
    )
    return SchemeBackwardEuler(
        constant_rhs,
        workbench,
        resolvent=resolvent,
    )


def test_fixed_scheme_call_returns_accepted_dt() -> None:
    scheme = SchemeRK4(exponential_growth, ScalarWorkbench())
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = ScalarState(1.0)

    accepted_dt = scheme(interval, state, Executor())

    assert accepted_dt == pytest.approx(0.125)
    assert state.value == pytest.approx(1.133148193359375)


def test_fixed_scheme_call_clips_to_remaining_interval() -> None:
    scheme = SchemeRK4(exponential_growth, ScalarWorkbench())
    interval = Interval(present=0.2, step=0.125, stop=0.25)
    state = ScalarState(1.0)

    accepted_dt = scheme(interval, state, Executor())

    assert accepted_dt == pytest.approx(0.05)
    assert state.value == pytest.approx(1.05127109375)


def test_adaptive_scheme_call_returns_accepted_dt() -> None:
    scheme = SchemeBogackiShampine(zero_rhs, ScalarWorkbench())
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(2.0)
    executor = Executor(tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))

    accepted_dt = scheme(interval, state, executor)

    assert accepted_dt == pytest.approx(0.1)
    assert state.value == pytest.approx(2.0)


def test_adaptive_scheme_call_clips_next_accepted_dt_to_remaining_interval() -> None:
    scheme = SchemeBogackiShampine(zero_rhs, ScalarWorkbench())
    interval = Interval(present=0.1, step=1.0, stop=0.3)
    state = ScalarState(2.0)
    executor = Executor(tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))

    accepted_dt = scheme(interval, state, executor)

    assert accepted_dt == pytest.approx(0.2)
    assert state.value == pytest.approx(2.0)


@pytest.mark.parametrize(
    "scheme",
    [
        SchemeRK4(exponential_growth, ScalarWorkbench()),
        SchemeBogackiShampine(zero_rhs, ScalarWorkbench()),
        make_backward_euler(),
    ],
)
def test_snapshot_state_works_through_scheme_object(scheme) -> None:
    state = ScalarState(3.0)

    snapshot = scheme.snapshot_state(state)

    assert snapshot is not state
    assert snapshot.value == pytest.approx(3.0)

    state.value = 9.0

    assert snapshot.value == pytest.approx(3.0)


@pytest.mark.parametrize(
    "scheme",
    [
        SchemeRK4(exponential_growth, ScalarWorkbench()),
        SchemeBogackiShampine(zero_rhs, ScalarWorkbench()),
        make_backward_euler(),
    ],
)
def test_set_apply_delta_safety_works_through_scheme_object(scheme) -> None:
    # This is a public contract guard. It deliberately avoids asserting private
    # workspace/stepper internals; the important point for the refactor is that
    # schemes continue to expose the safety switch directly.
    scheme.set_apply_delta_safety(False)
    scheme.set_apply_delta_safety(True)

    snapshot = scheme.snapshot_state(ScalarState(1.0))

    assert snapshot.value == pytest.approx(1.0)


def test_self_contained_scheme_exemplars_own_public_call_method() -> None:
    assert "__call__" in SchemeEuler.__dict__
    assert "__call__" in SchemeRK4.__dict__
    assert "__call__" in SchemeBogackiShampine.__dict__
    assert "__call__" in SchemeBackwardEuler.__dict__


def test_unconverted_fixed_explicit_scheme_still_works_without_deprecation_warning() -> None:
    scheme = SchemeHeun(exponential_growth, ScalarWorkbench())
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = ScalarState(1.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        accepted_dt = scheme(interval, state, Executor())

    assert accepted_dt == pytest.approx(0.125)
    assert state.value > 1.0
    assert caught == []