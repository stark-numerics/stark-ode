from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from stark import Executor, Integrator, Interval, Marcher, Tolerance
from stark.algebraist import Algebraist, AlgebraistField, AlgebraistLooped
from stark.schemes.explicit_adaptive.bogacki_shampine import SchemeBogackiShampine
from stark.schemes.explicit_adaptive.cash_karp import SchemeCashKarp
from stark.schemes.explicit_adaptive.dormand_prince import SchemeDormandPrince
from stark.schemes.explicit_adaptive.fehlberg45 import SchemeFehlberg45
from stark.schemes.explicit_adaptive.tsitouras5 import SchemeTsitouras5
from stark.schemes.explicit_fixed.euler import SchemeEuler
from stark.schemes.explicit_fixed.heun import SchemeHeun
from stark.schemes.explicit_fixed.kutta3 import SchemeKutta3
from stark.schemes.explicit_fixed.midpoint import SchemeMidpoint
from stark.schemes.explicit_fixed.ralston import SchemeRalston
from stark.schemes.explicit_fixed.rk38 import SchemeRK38
from stark.schemes.explicit_fixed.rk4 import SchemeRK4
from stark.schemes.explicit_fixed.ssprk33 import SchemeSSPRK33


ALGEBRAIST = Algebraist(
    fields=(AlgebraistField("dy", "y", policy=AlgebraistLooped(rank=1)),),
    generate_norm="l2",
)


@dataclass(slots=True)
class ArrayState:
    y: np.ndarray


@dataclass(slots=True)
class ArrayTranslation:
    dy: np.ndarray

    linear_combine = ALGEBRAIST.linear_combine
    __call__ = ALGEBRAIST.apply
    norm = ALGEBRAIST.norm

    def __add__(self, other: "ArrayTranslation") -> "ArrayTranslation":
        return ArrayTranslation(self.dy + other.dy)

    def __rmul__(self, scalar: float) -> "ArrayTranslation":
        return ArrayTranslation(scalar * self.dy)


class ArrayWorkbench:
    def __init__(self, size: int) -> None:
        self.size = size

    def allocate_state(self) -> ArrayState:
        return ArrayState(np.zeros(self.size))

    def copy_state(self, dst: ArrayState, src: ArrayState) -> None:
        dst.y[...] = src.y

    def allocate_translation(self) -> ArrayTranslation:
        return ArrayTranslation(np.zeros(self.size))


class ArrayDerivative:
    def __call__(self, interval: Interval, state: ArrayState, out: ArrayTranslation) -> None:
        out.dy[...] = np.array(
            [
                state.y[1] + 0.25 * interval.present,
                -state.y[0] + 0.1 * state.y[2],
                -0.5 * state.y[2] + 0.2,
            ]
        )


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


def run_fixed_step(scheme_type, *, algebraist: Algebraist | None = None) -> ArrayState:
    state = build_state()
    interval = Interval(0.0, 0.05, 0.05)
    scheme = scheme_type(ArrayDerivative(), ArrayWorkbench(3), algebraist=algebraist)

    accepted_dt = scheme(interval, state, Executor())

    assert accepted_dt == pytest.approx(0.05)
    return state


def run_adaptive_solve(scheme_type, *, algebraist: Algebraist | None = None) -> tuple[ArrayState, int, float]:
    state = build_state()
    interval = Interval(0.0, 0.05, 0.25)
    scheme = scheme_type(ArrayDerivative(), ArrayWorkbench(3), algebraist=algebraist)
    marcher = Marcher(scheme, Executor(tolerance=Tolerance(atol=1.0e-10, rtol=1.0e-10)))
    integrator = Integrator(executor=marcher.executor)

    steps = 0
    for _interval, _state in integrator.live(marcher, interval, state):
        del _interval, _state
        steps += 1

    return state, steps, interval.step


@pytest.mark.parametrize("scheme_type", FIXED_SCHEMES)
def test_fixed_explicit_scheme_algebraist_path_matches_generic_path(scheme_type):
    generic = run_fixed_step(scheme_type)
    algebraist = run_fixed_step(scheme_type, algebraist=ALGEBRAIST)

    np.testing.assert_allclose(algebraist.y, generic.y, rtol=1.0e-14, atol=1.0e-14)


@pytest.mark.parametrize("scheme_type", ADAPTIVE_SCHEMES)
def test_adaptive_explicit_scheme_algebraist_path_matches_generic_path(scheme_type):
    generic_state, generic_steps, generic_next_step = run_adaptive_solve(scheme_type)
    algebraist_state, algebraist_steps, algebraist_next_step = run_adaptive_solve(
        scheme_type,
        algebraist=ALGEBRAIST,
    )

    assert algebraist_steps == generic_steps
    assert algebraist_next_step == pytest.approx(generic_next_step, rel=1.0e-14, abs=1.0e-14)
    np.testing.assert_allclose(algebraist_state.y, generic_state.y, rtol=1.0e-14, atol=1.0e-14)


def test_adaptive_scheme_algebraist_binding_uses_base_advance_body_redirect():
    scheme = SchemeCashKarp(ArrayDerivative(), ArrayWorkbench(3), algebraist=ALGEBRAIST)

    assert scheme.redirect_advance_body.__func__ is scheme.advance_body_algebraist.__func__
