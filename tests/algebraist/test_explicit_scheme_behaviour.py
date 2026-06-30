from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from stark import Configuration, Interval, Tolerance
from stark.core import Integrator, IntegratorStepper
from stark.methods.schemes.explicit.adaptive.cash_karp import SchemeCashKarp
from stark.methods.schemes.explicit.fixed.rk4 import SchemeRK4


@dataclass(slots=True)
class ArrayState:
    y: np.ndarray

    def copy(self) -> "ArrayState":
        return ArrayState(self.y.copy())


@dataclass(slots=True)
class ArrayTranslation:
    dy: np.ndarray

    def __call__(self, origin: ArrayState, result: ArrayState) -> None:
        result.y[...] = origin.y + self.dy

    def norm(self) -> float:
        return float(np.linalg.norm(self.dy))

    def __add__(self, other: "ArrayTranslation") -> "ArrayTranslation":
        return ArrayTranslation(self.dy + other.dy)

    def __rmul__(self, scalar: float) -> "ArrayTranslation":
        return ArrayTranslation(scalar * self.dy)


class ArrayAllocator:
    def __init__(self, size: int) -> None:
        self.size = size

    def allocate_state(self) -> ArrayState:
        return ArrayState(np.zeros(self.size))

    def copy_state(self, source: ArrayState, out: ArrayState) -> None:
        out.y[...] = source.y

    def allocate_translation(self) -> ArrayTranslation:
        return ArrayTranslation(np.zeros(self.size))


class ArrayDerivative:
    def __call__(
        self,
        interval: Interval,
        state: ArrayState,
        out: ArrayTranslation,
    ) -> None:
        out.dy[...] = np.array(
            [
                state.y[1] + 0.25 * interval.present,
                -state.y[0] + 0.1 * state.y[2],
                -0.5 * state.y[2] + 0.2,
            ]
        )


class ArraySpecialist:
    def provide_delta(self, stencil):
        coefficients = tuple(stencil.coefficients)
        stencil_scale = stencil.scale

        if stencil.apply:
            def apply_kernel(
                step: float,
                origin,
                *terms,
            ):
                *translations, result = terms
                delta = _combine_delta(step, stencil_scale, coefficients, translations)
                delta(origin, result)
                return result

            return apply_kernel

        def delta_kernel(
            step: float,
            *terms,
        ):
            *translations, out = terms
            delta = _combine_delta(step, stencil_scale, coefficients, translations)
            out.dy[...] = delta.dy
            return out

        return delta_kernel

    provide_apply = provide_delta


def _combine_delta(
    step: float,
    stencil_scale: float,
    coefficients: tuple[float, ...],
    translations: tuple[ArrayTranslation, ...],
) -> ArrayTranslation:
    if len(coefficients) != len(translations):
        raise AssertionError(
            f"stencil arity {len(coefficients)} received "
            f"{len(translations)} translation(s)"
        )

    if not translations:
        return ArrayTranslation(np.zeros(3))

    total = 0.0 * translations[0]
    for coefficient, translation in zip(coefficients, translations, strict=True):
        total = total + (step * stencil_scale * coefficient) * translation
    return total


def build_state() -> ArrayState:
    return ArrayState(np.array([1.0, -0.5, 0.25]))


def test_fixed_explicit_specialist_path_matches_inline_semantics() -> None:
    inline_state = build_state()
    specialist_state = build_state()

    inline = SchemeRK4(ArrayDerivative(), ArrayAllocator(3))
    specialized = SchemeRK4(
        ArrayDerivative(),
        ArrayAllocator(3),
        specialist=ArraySpecialist(),
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
    inline_scheme = SchemeCashKarp(ArrayDerivative(), ArrayAllocator(3))
    specialized_scheme = SchemeCashKarp(
        ArrayDerivative(),
        ArrayAllocator(3),
        specialist=ArraySpecialist(),
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
) -> tuple[ArrayState, int, float]:
    state = build_state()
    interval = Interval(present=0.0, step=0.05, stop=0.25)
    configuration = Configuration(scheme_tolerance=Tolerance(atol=1.0e-10, rtol=1.0e-10))
    stepper = IntegratorStepper(scheme)
    integrator = Integrator(configuration=Configuration())

    steps = 0
    for _snapshot_interval, _snapshot_state in integrator.mutating_trajectory(stepper, interval, state):
        del _snapshot_interval, _snapshot_state
        steps += 1

    return state, steps, interval.step


def test_non_algebraist_scheme_does_not_carry_generated_source_state() -> None:
    scheme = SchemeRK4(ArrayDerivative(), ArrayAllocator(3))

    assert not hasattr(scheme, "sources")
    assert not hasattr(scheme, "kernel_sources")
    assert not hasattr(scheme, "wrapper_sources")
    assert not hasattr(scheme, "generated_source")
