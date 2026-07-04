from __future__ import annotations

import pytest

from stark import Interval
from stark.core.contracts import IntervalLike
from stark.methods.schemes.execution.derivative import SchemeDerivative
from stark.methods.schemes.execution.step_support import SchemeStepSupport
from stark.methods.schemes.explicit.fixed.rk4 import SchemeRK4
from stark.methods.schemes.explicit.runtime import SchemeRuntimeExplicit
from tests.support import DummyScalarAllocator, DummyScalarState, DummyScalarTranslation


ScalarState = DummyScalarState
ScalarTranslation = DummyScalarTranslation
ScalarAllocator = DummyScalarAllocator


def exponential_growth(
    interval: IntervalLike,
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


def test_explicit_runtime_constructs_prepared_derivative_and_step_support() -> None:
    runtime = SchemeRuntimeExplicit(
        exponential_growth,
        ScalarAllocator(),
    )

    assert isinstance(runtime.derivative, SchemeDerivative)
    assert runtime.derivative is exponential_growth
    assert isinstance(runtime.workspace, SchemeStepSupport)
    assert isinstance(runtime.first_translation, ScalarTranslation)
    assert runtime.k1 is runtime.first_translation


def test_explicit_runtime_prepared_derivative_calls_original_worker() -> None:
    runtime = SchemeRuntimeExplicit(
        exponential_growth,
        ScalarAllocator(),
    )
    interval = Interval(present=0.0, step=0.1, stop=1.0)
    state = ScalarState(3.0)
    out = ScalarTranslation()

    runtime.derivative(interval, state, out)

    assert out.value == pytest.approx(3.0)


def test_explicit_runtime_snapshot_state_copies_through_step_support() -> None:
    runtime = SchemeRuntimeExplicit(
        exponential_growth,
        ScalarAllocator(),
    )
    state = ScalarState(4.0)

    snapshot = runtime.snapshot_state(state)

    assert snapshot is not state
    assert snapshot.value == pytest.approx(4.0)

    state.value = 9.0

    assert snapshot.value == pytest.approx(4.0)


def test_explicit_runtime_workspace_apply_delta_updates_state_in_place() -> None:
    runtime = SchemeRuntimeExplicit(
        exponential_growth,
        ScalarAllocator(),
    )
    state = ScalarState(2.0)
    delta = ScalarTranslation(1.0)

    runtime.workspace.apply_delta(delta, state)

    assert state.value == pytest.approx(3.0)

    runtime.workspace.apply_delta(delta, state)

    assert state.value == pytest.approx(4.0)


def test_explicit_scheme_owns_explicit_runtime() -> None:
    scheme = SchemeRK4(exponential_growth, ScalarAllocator())

    assert isinstance(scheme.runtime, SchemeRuntimeExplicit)
    assert scheme.derivative is scheme.runtime.derivative
    assert scheme.workspace is scheme.runtime.workspace
    assert scheme.k1 is scheme.runtime.k1


def test_existing_explicit_scheme_still_runs_after_support_extraction() -> None:
    scheme = SchemeRK4(exponential_growth, ScalarAllocator())
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = ScalarState(1.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.125)
    assert state.value == pytest.approx(1.133148193359375)
