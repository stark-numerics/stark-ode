from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from stark import Executor, Integrator, Interval, Marcher, Monitor, Tolerance
from stark.schemes.explicit_adaptive.cash_karp import SchemeCashKarp
from stark.schemes.explicit_fixed.euler import SchemeEuler


ROOT = Path(__file__).resolve().parents[2]


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
            origin: ScalarState,
            dt: float,
            k1: ScalarTranslation,
            result: ScalarState,
        ) -> None:
            result.value = origin.value + dt * k1.value

        return SimpleNamespace(
            stage_state_calls=(None,),
            require_stage_state_call=lambda index, scheme_name: (None,)[index],
            solution_state_call=solution_state,
        )


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


def failing_rhs(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval, state, out
    raise RuntimeError("intentional example failure")


def tight_executor() -> Executor:
    return Executor(tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))


def test_assigning_scheme_monitor_selects_monitored_path_and_unassign_restores_pure_path() -> None:
    scheme = SchemeEuler(constant_rhs, ScalarWorkbench())
    monitor = Monitor()

    assert scheme.redirect_call == scheme.call_pure

    scheme.assign_monitor(monitor.scheme)

    assert scheme.redirect_call.__func__ is scheme.call_monitored.__func__

    scheme.unassign_monitor()

    assert scheme.redirect_call == scheme.call_pure


def test_unmonitored_integration_creates_no_scheme_monitor_records() -> None:
    scheme = SchemeCashKarp(zero_rhs, ScalarWorkbench())
    marcher = Marcher(scheme, tight_executor())
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    monitor = Monitor()

    list(Integrator().live(marcher, interval, ScalarState()))

    assert monitor.scheme.fixed_steps == []
    assert monitor.scheme.adaptive_steps == []
    assert scheme.step_control.monitor is None


def test_monitor_is_unassigned_after_monitored_integration_exception() -> None:
    scheme = SchemeEuler(failing_rhs, ScalarWorkbench())
    marcher = Marcher(scheme, Executor())
    monitor = Monitor()

    with pytest.raises(RuntimeError, match="intentional example failure"):
        list(
            Integrator().monitored(
                marcher,
                Interval(present=0.0, step=0.1, stop=0.2),
                ScalarState(),
                monitor,
            )
        )

    assert marcher.monitor is None
    assert scheme.redirect_call == scheme.call_pure
    assert monitor.scheme.fixed_steps == []
    assert monitor.scheme.adaptive_steps == []


def test_algebraist_fixed_path_is_monitored_only_at_scheme_boundary() -> None:
    scheme = SchemeEuler(
        constant_rhs,
        ScalarWorkbench(),
        algebraist=StubAlgebraist(),
    )
    monitor = Monitor()
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = ScalarState()

    assert scheme.call_pure.__func__ is SchemeEuler.call_specialized

    scheme.assign_monitor(monitor.scheme)
    accepted_dt = scheme(interval, state, Executor())

    assert accepted_dt == pytest.approx(0.125)
    assert state.value == pytest.approx(0.125)
    assert scheme.call_pure.__func__ is SchemeEuler.call_specialized
    assert len(monitor.scheme.fixed_steps) == 1


def test_schemes_depend_on_monitor_protocol_not_concrete_monitor_records() -> None:
    scheme_files = [
        path
        for path in (ROOT / "stark" / "schemes").rglob("*.py")
        if "__pycache__" not in path.parts
    ]

    offenders = [
        path.relative_to(ROOT)
        for path in scheme_files
        if "stark.monitor" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
