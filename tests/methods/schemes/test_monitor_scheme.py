from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from stark import Interval, Monitor
from stark.core import Integrator, IntegratorStepper, Tolerance
from stark.methods.schemes.explicit.adaptive.cash_karp import SchemeCashKarp
from stark.methods.schemes.explicit.fixed.euler import SchemeEuler
from stark.methods.schemes.specialization.stencil import SchemeStencil


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


class ScalarAllocator:
    def allocate_state(self) -> ScalarState:
        return ScalarState()

    def copy_state(self, source: ScalarState, out: ScalarState) -> None:
        out.value = source.value

    def allocate_translation(self) -> ScalarTranslation:
        return ScalarTranslation()


class StubSpecialist:
    def provide_delta(self, stencil: SchemeStencil):
        coefficients = stencil.coefficients
        fixed_scale = stencil.scale

        if stencil.apply:

            def apply_kernel(
                step: float,
                origin,
                *terms,
            ):
                *translations, result = terms
                result.value = origin.value + step * fixed_scale * sum(
                    coefficient * translation.value
                    for coefficient, translation in zip(
                        coefficients,
                        translations,
                        strict=True,
                    )
                )
                return result

            return apply_kernel

        def delta_kernel(
            step: float,
            *terms,
        ):
            *translations, out = terms
            out.value = step * fixed_scale * sum(
                coefficient * translation.value
                for coefficient, translation in zip(coefficients, translations, strict=True)
            )
            return out

        return delta_kernel

    provide_apply = provide_delta


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


def tight_configuration() -> Configuration:
    return Configuration(scheme_tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))


def test_assigning_scheme_monitor_selects_monitored_path() -> None:
    monitor = Monitor()
    scheme = SchemeEuler(constant_rhs, ScalarAllocator())

    assert scheme.redirect_call == scheme.call_step
    
    scheme = SchemeEuler(constant_rhs, ScalarAllocator(), monitor=monitor.scheme)

    assert scheme.redirect_call.__func__ is scheme.call_monitored.__func__



def test_unmonitored_integration_creates_no_scheme_monitor_records() -> None:
    scheme = SchemeCashKarp(zero_rhs, ScalarAllocator())
    stepper = IntegratorStepper(scheme)
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    monitor = Monitor()

    list(Integrator().mutating_trajectory(stepper, interval, ScalarState()))

    assert monitor.scheme.fixed_steps == []
    assert monitor.scheme.adaptive_steps == []
    assert scheme.monitor is None


def test_direct_scheme_monitor_remains_available_after_integration_exception() -> None:
    monitor = Monitor()
    scheme = SchemeEuler(failing_rhs, ScalarAllocator(), monitor=monitor.scheme)
    stepper = IntegratorStepper(scheme)

    with pytest.raises(RuntimeError, match="intentional example failure"):
        list(
            Integrator().mutating_trajectory(
                stepper,
                Interval(present=0.0, step=0.1, stop=0.2),
                ScalarState(),
            )
        )

    assert scheme.monitor is monitor.scheme
    assert scheme.redirect_call.__func__ is scheme.call_monitored.__func__
    assert monitor.scheme.fixed_steps == []
    assert monitor.scheme.adaptive_steps == []


def test_specialist_fixed_path_is_monitored_only_at_scheme_boundary() -> None:
    scheme = SchemeEuler(
        constant_rhs,
        ScalarAllocator(),
        specialist=StubSpecialist(),
    )
    monitor = Monitor()
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = ScalarState()

    assert scheme.call_step.__func__ is SchemeEuler.call_specialized

    scheme = SchemeEuler(
        constant_rhs,
        ScalarAllocator(),
        specialist=StubSpecialist(),
        monitor=monitor.scheme,
    )

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.125)
    assert state.value == pytest.approx(0.125)
    assert scheme.call_body.__func__ is SchemeEuler.call_specialized
    assert scheme.call_step.__func__ is scheme.call_monitored.__func__
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
        if "stark.diagnostics.monitor" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
