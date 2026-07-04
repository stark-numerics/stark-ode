from __future__ import annotations

import numpy as np
import pytest

from stark import Configuration, Interval
from stark.core import Integrator, IntegratorStepper
from stark.methods.schemes.explicit.adaptive.cash_karp import SchemeCashKarp
from stark.methods.schemes.explicit.fixed.rk4 import SchemeRK4
from tests.support import (
    DummyArrayAllocator,
    DummyArrayDynamics,
    DummyArraySpecialist,
    DummyArrayState,
    dummy_array_state,
)


def build_state() -> DummyArrayState:
    return dummy_array_state()


def test_fixed_explicit_specialist_path_matches_inline_semantics() -> None:
    inline_state = build_state()
    specialist_state = build_state()

    inline = SchemeRK4(DummyArrayDynamics(), DummyArrayAllocator(3))
    specialized = SchemeRK4(
        DummyArrayDynamics(),
        DummyArrayAllocator(3),
        specialist=DummyArraySpecialist(),
    )

    inline_dt = inline(
        Interval(present=0.0, step=0.05, stop=0.05),
        inline_state,
    )
    specialist_dt = specialized(
        Interval(present=0.0, step=0.05, stop=0.05),
        specialist_state,
    )

    assert specialist_dt == pytest.approx(inline_dt)
    np.testing.assert_allclose(
        specialist_state.y,
        inline_state.y,
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_adaptive_explicit_specialist_path_matches_inline_semantics() -> None:
    inline_scheme = SchemeCashKarp(DummyArrayDynamics(), DummyArrayAllocator(3))
    specialized_scheme = SchemeCashKarp(
        DummyArrayDynamics(),
        DummyArrayAllocator(3),
        specialist=DummyArraySpecialist(),
    )

    inline_state, inline_steps, inline_next_step = run_adaptive_solve(inline_scheme)
    specialist_state, specialist_steps, specialist_next_step = run_adaptive_solve(
        specialized_scheme
    )

    assert specialist_steps == inline_steps
    assert specialist_next_step == pytest.approx(
        inline_next_step,
        rel=1.0e-14,
        abs=1.0e-14,
    )
    np.testing.assert_allclose(
        specialist_state.y,
        inline_state.y,
        rtol=1.0e-14,
        atol=1.0e-14,
    )

    inline_report = inline_scheme.step_control.report()
    specialist_report = specialized_scheme.step_control.report()

    assert specialist_report.accepted_dt == pytest.approx(
        inline_report.accepted_dt,
        rel=1.0e-14,
        abs=1.0e-14,
    )
    assert specialist_report.proposed_dt == pytest.approx(
        inline_report.proposed_dt,
        rel=1.0e-14,
        abs=1.0e-14,
    )
    assert specialist_report.next_dt == pytest.approx(
        inline_report.next_dt,
        rel=1.0e-14,
        abs=1.0e-14,
    )
    assert specialist_report.error_ratio == pytest.approx(
        inline_report.error_ratio,
        rel=1.0e-14,
        abs=1.0e-14,
    )
    assert specialist_report.rejection_count == inline_report.rejection_count


def run_adaptive_solve(
    scheme: SchemeCashKarp,
) -> tuple[DummyArrayState, int, float]:
    state = build_state()
    interval = Interval(present=0.0, step=0.05, stop=0.25)
    stepper = IntegratorStepper(scheme)
    integrator = Integrator(configuration=Configuration())

    steps = 0
    for _snapshot_interval, _snapshot_state in integrator.mutating_trajectory(stepper, interval, state):
        del _snapshot_interval, _snapshot_state
        steps += 1

    return state, steps, interval.step


def test_non_algebraist_scheme_does_not_carry_generated_source_state() -> None:
    scheme = SchemeRK4(DummyArrayDynamics(), DummyArrayAllocator(3))

    assert not hasattr(scheme, "sources")
    assert not hasattr(scheme, "kernel_sources")
    assert not hasattr(scheme, "wrapper_sources")
    assert not hasattr(scheme, "generated_source")
