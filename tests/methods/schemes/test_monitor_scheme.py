from __future__ import annotations

from pathlib import Path

import pytest

from stark import Interval, Monitor
from stark.core import Integrator, IntegratorStepper
from stark.core.contracts import IntervalLike
from stark.methods.schemes.explicit.adaptive.cash_karp import SchemeCashKarp
from stark.methods.schemes.explicit.fixed.euler import SchemeEuler
from stark.problem import Dynamics, DynamicsStyle
from tests.support import (
    DummyScalarAllocator,
    DummyScalarState,
    DummyScalarTranslation,
    DummyTableauSpecialist,
    dummy_constant_dynamics,
    dummy_zero_rhs,
)


ROOT = Path(__file__).resolve().parents[3]


def failing_dynamics() -> Dynamics:
    """Dynamics fixture that raises after the scheme monitor path is selected."""

    def write(
        interval: IntervalLike,
        state: DummyScalarState,
        out: DummyScalarTranslation,
    ) -> None:
        del interval, state, out
        raise RuntimeError("intentional example failure")

    return Dynamics(DynamicsStyle.accepts_interval_writes(write))


def test_assigning_scheme_monitor_selects_monitored_path() -> None:
    monitor = Monitor()
    scheme = SchemeEuler(dummy_constant_dynamics(), DummyScalarAllocator())

    assert scheme.redirect_call == scheme.call_step
    
    scheme = SchemeEuler(
        dummy_constant_dynamics(),
        DummyScalarAllocator(),
        monitor=monitor.scheme,
    )




def test_unmonitored_integration_creates_no_scheme_monitor_records() -> None:
    scheme = SchemeCashKarp(dummy_zero_rhs, DummyScalarAllocator())
    stepper = IntegratorStepper(scheme)
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    monitor = Monitor()

    list(Integrator().mutating_trajectory(stepper, interval, DummyScalarState()))

    assert monitor.scheme.fixed_steps == []
    assert monitor.scheme.adaptive_steps == []
    assert scheme.monitor is None


def test_direct_scheme_monitor_remains_available_after_integration_exception() -> None:
    monitor = Monitor()
    scheme = SchemeEuler(
        failing_dynamics(),
        DummyScalarAllocator(),
        monitor=monitor.scheme,
    )
    stepper = IntegratorStepper(scheme)

    with pytest.raises(RuntimeError, match="intentional example failure"):
        list(
            Integrator().mutating_trajectory(
                stepper,
                Interval(present=0.0, step=0.1, stop=0.2),
                DummyScalarState(),
            )
        )

    assert scheme.monitor is monitor.scheme
    assert monitor.scheme.fixed_steps == []
    assert monitor.scheme.adaptive_steps == []


def test_specialist_fixed_path_is_monitored_only_at_scheme_boundary() -> None:
    scheme = SchemeEuler(
        dummy_constant_dynamics(),
        DummyScalarAllocator(),
        specialist=DummyTableauSpecialist(),
    )
    monitor = Monitor()
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = DummyScalarState()


    scheme = SchemeEuler(
        dummy_constant_dynamics(),
        DummyScalarAllocator(),
        specialist=DummyTableauSpecialist(),
        monitor=monitor.scheme,
    )

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.125)
    assert state.value == pytest.approx(0.125)
    assert len(monitor.scheme.fixed_steps) == 1


def test_schemes_depend_on_monitor_protocol_not_concrete_monitor_records() -> None:
    scheme_files = [
        path
        for path in (ROOT / "stark" / "methods" / "schemes").rglob("*.py")
        if "__pycache__" not in path.parts
    ]
    offenders = [
        path.relative_to(ROOT)
        for path in scheme_files
        if "stark.diagnostics.monitor" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
