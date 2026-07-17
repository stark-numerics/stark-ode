from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, cast

import numpy as np
import pytest

from stark import Interval, Tolerance
from stark.core.contracts import IntervalLike, LinearCombine
from stark.core import Integrator, IntegratorStepper
from stark.methods.schemes.explicit.adaptive.bogacki_shampine import SchemeBogackiShampine
from stark.methods.schemes.explicit.adaptive.cash_karp import SchemeCashKarp
from stark.methods.schemes.explicit.adaptive.dormand_prince import SchemeDormandPrince
from stark.methods.schemes.explicit.adaptive.fehlberg45 import SchemeFehlberg45
from stark.methods.schemes.explicit.adaptive.tsitouras5 import SchemeTsitouras5
from stark.methods.schemes.explicit.fixed.euler import SchemeEuler
from stark.methods.schemes.explicit.fixed.heun import SchemeHeun
from stark.methods.schemes.explicit.fixed.kutta3 import SchemeKutta3
from stark.methods.schemes.explicit.fixed.midpoint import SchemeMidpoint
from stark.methods.schemes.explicit.fixed.ralston import SchemeRalston
from stark.methods.schemes.explicit.fixed.rk38 import SchemeRK38
from stark.methods.schemes.explicit.fixed.rk4 import SchemeRK4
from stark.methods.schemes.explicit.fixed.ssprk33 import SchemeSSPRK33
from stark.core import Configuration

@dataclass(slots=True)
class ArrayState:
    y: np.ndarray


@dataclass(slots=True)
class ArrayTranslation:
    dy: np.ndarray
    linear_combine: ClassVar[LinearCombine]

    def __call__(self, origin: ArrayState, result: ArrayState) -> None:
        result.y[...] = origin.y + self.dy

    def norm(self) -> float:
        return float(np.linalg.norm(self.dy))

    def __add__(self, other: "ArrayTranslation") -> "ArrayTranslation":
        return ArrayTranslation(self.dy + other.dy)

    def __rmul__(self, scalar: float) -> "ArrayTranslation":
        return ArrayTranslation(scalar * self.dy)


def array_combine_many(*terms: object) -> ArrayTranslation:
    out = terms[-1]
    if not isinstance(out, ArrayTranslation):
        raise TypeError("array combine needs an ArrayTranslation output.")

    out.dy[...] = 0.0
    for index in range(0, len(terms) - 1, 2):
        scalar = cast(float, terms[index])
        translation = terms[index + 1]
        if not isinstance(translation, ArrayTranslation):
            raise TypeError("array combine terms must be translations.")
        out.dy[...] += scalar * translation.dy
    return out


ArrayTranslation.linear_combine = tuple(array_combine_many for _ in range(12))


class ArrayAllocator:
    linear_combine = ArrayTranslation.linear_combine

    def __init__(self, size: int) -> None:
        self.size = size

    def allocate_state(self) -> ArrayState:
        return ArrayState(np.zeros(self.size))

    def copy_state(self, source: ArrayState, out: ArrayState) -> None:
        out.y[...] = source.y

    def allocate_translation(self) -> ArrayTranslation:
        return ArrayTranslation(np.zeros(self.size))


class ArrayDynamics:
    def __call__(self, interval: IntervalLike, state: ArrayState, out: ArrayTranslation) -> None:
        out.dy[...] = np.array(
            [
                state.y[1] + 0.25 * interval.present,
                -state.y[0] + 0.1 * state.y[2],
                -0.5 * state.y[2] + 0.2,
            ]
        )


class ArrayLinearFixed:
    def __call__(self, stencil):
        return self.provide_delta(stencil)

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
                delta = _combine_delta(step, stencil_scale, coefficients, tuple(translations))
                delta(origin, result)
                return result

            return apply_kernel

        def delta_kernel(
            step: float,
            *terms,
        ):
            *translations, out = terms
            delta = _combine_delta(step, stencil_scale, coefficients, tuple(translations))
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


FIXED_SCHEMES = [
    SchemeEuler,
    SchemeHeun,
    SchemeMidpoint,
    SchemeRalston,
    SchemeKutta3,
    SchemeSSPRK33,
    SchemeRK38,
    SchemeRK4,
]

ADAPTIVE_SCHEMES = [
    SchemeBogackiShampine,
    SchemeCashKarp,
    SchemeDormandPrince,
    SchemeFehlberg45,
    SchemeTsitouras5,
]


def build_state() -> ArrayState:
    return ArrayState(np.array([1.0, -0.5, 0.25]))


def run_fixed_step(scheme_type, *, linear_fixed=None) -> ArrayState:
    state = build_state()
    interval = Interval(0.0, 0.05, 0.05)
    scheme = scheme_type(ArrayDynamics(), ArrayAllocator(3), linear_fixed=linear_fixed)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.05)
    return state


def run_adaptive_solve(scheme_type, *, linear_fixed=None) -> tuple[ArrayState, int, float]:
    state = build_state()
    interval = Interval(0.0, 0.05, 0.25)
    scheme = scheme_type(ArrayDynamics(), ArrayAllocator(3), linear_fixed=linear_fixed)
    stepper = IntegratorStepper(scheme)
    integrator = Integrator(configuration=Configuration())

    steps = 0
    for _interval, _state in integrator.mutating_trajectory(stepper, interval, state):
        del _interval, _state
        steps += 1

    return state, steps, interval.step
@pytest.mark.parametrize("scheme_type", ADAPTIVE_SCHEMES)
def test_adaptive_explicit_scheme_linear_fixed_path_matches_inline_path(scheme_type):
    inline_state, inline_steps, inline_next_step = run_adaptive_solve(scheme_type)
    linear_fixed_state, linear_fixed_steps, linear_fixed_next_step = run_adaptive_solve(
        scheme_type,
        linear_fixed=ArrayLinearFixed(),
    )

    assert linear_fixed_steps == inline_steps
    assert linear_fixed_next_step == pytest.approx(inline_next_step, rel=1.0e-14, abs=1.0e-14)
    np.testing.assert_allclose(linear_fixed_state.y, inline_state.y, rtol=1.0e-14, atol=1.0e-14)
