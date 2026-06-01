from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark import Executor, Interval
from stark.schemes.execution.derivative import SchemeDerivative
from stark.schemes.execution.support import SchemeStepSupport
from stark.schemes.explicit.fixed.rk4 import SchemeRK4
from stark.schemes.explicit._support import SchemeSupportExplicit


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


def exponential_growth(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval
    out.value = state.value


def increment_by_one(
    origin: ScalarState,
    result: ScalarState,
) -> None:
    result.value = origin.value + 1.0


class FixedDelta:
    def __call__(self, origin: ScalarState, result: ScalarState) -> None:
        result.value = origin.value + 1.0

    def norm(self) -> float:
        return 1.0

    def __add__(self, other: FixedDelta) -> FixedDelta:
        del other
        return FixedDelta()

    def __rmul__(self, scalar: float) -> FixedDelta:
        del scalar
        return FixedDelta()


def test_explicit_support_constructs_prepared_derivative_and_step_support() -> None:
    support = SchemeSupportExplicit.from_inputs(
        exponential_growth,
        ScalarAllocator(),
    )

    assert isinstance(support.derivative, SchemeDerivative)
    assert support.derivative is exponential_growth
    assert isinstance(support.step_support, SchemeStepSupport)
    assert isinstance(support.first_translation, ScalarTranslation)
    assert support.k1 is support.first_translation


def test_explicit_support_prepared_derivative_calls_original_worker() -> None:
    support = SchemeSupportExplicit.from_inputs(
        exponential_growth,
        ScalarAllocator(),
    )
    interval = Interval(present=0.0, step=0.1, stop=1.0)
    state = ScalarState(3.0)
    out = ScalarTranslation()

    support.derivative(interval, state, out)

    assert out.value == pytest.approx(3.0)


def test_explicit_support_snapshot_state_copies_through_step_support() -> None:
    support = SchemeSupportExplicit.from_inputs(
        exponential_growth,
        ScalarAllocator(),
    )
    state = ScalarState(4.0)

    snapshot = support.snapshot_state(state)

    assert snapshot is not state
    assert snapshot.value == pytest.approx(4.0)

    state.value = 9.0

    assert snapshot.value == pytest.approx(4.0)


def test_explicit_support_step_support_apply_delta_updates_state_in_place() -> None:
    support = SchemeSupportExplicit.from_inputs(
        exponential_growth,
        ScalarAllocator(),
    )
    state = ScalarState(2.0)
    delta = FixedDelta()

    support.step_support.apply_delta(delta, state)

    assert state.value == pytest.approx(3.0)

    support.step_support.apply_delta(delta, state)

    assert state.value == pytest.approx(4.0)


def test_explicit_scheme_base_delegates_to_explicit_support() -> None:
    scheme = SchemeRK4(exponential_growth, ScalarAllocator())

    assert isinstance(scheme.explicit, SchemeSupportExplicit)
    assert scheme.derivative is scheme.explicit.derivative
    assert scheme.workspace is scheme.explicit.step_support
    assert scheme.k1 is scheme.explicit.k1


def test_existing_explicit_scheme_still_runs_after_support_extraction() -> None:
    scheme = SchemeRK4(exponential_growth, ScalarAllocator())
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = ScalarState(1.0)

    accepted_dt = scheme(interval, state, Executor())

    assert accepted_dt == pytest.approx(0.125)
    assert state.value == pytest.approx(1.133148193359375)
