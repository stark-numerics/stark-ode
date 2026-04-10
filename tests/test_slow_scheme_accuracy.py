from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, sqrt

import pytest

from stark import Marcher, Integrator, Interval, Tolerance
from stark.scheme_library import (
    SchemeBogackiShampine,
    SchemeCashKarp,
    SchemeDormandPrince,
    SchemeEuler,
    SchemeFehlberg45,
    SchemeHeun,
    SchemeKutta3,
    SchemeMidpoint,
    SchemeRalston,
    SchemeRK4,
    SchemeRK38,
    SchemeSSPRK33,
    SchemeTsitouras5,
)


STOP = 6.0


@dataclass(frozen=True, slots=True)
class SchemeCase:
    label: str
    scheme_type: type
    step: float
    tolerance: Tolerance | None
    max_error: float


SCHEME_CASES = [
    SchemeCase("Euler", SchemeEuler, 2.0e-4, None, 1.0e-4),
    SchemeCase("Heun", SchemeHeun, 1.0e-3, None, 2.0e-6),
    SchemeCase("Midpoint", SchemeMidpoint, 1.0e-3, None, 2.0e-6),
    SchemeCase("Ralston", SchemeRalston, 1.0e-3, None, 2.0e-6),
    SchemeCase("Kutta3", SchemeKutta3, 2.0e-3, None, 2.0e-7),
    SchemeCase("SSPRK33", SchemeSSPRK33, 2.0e-3, None, 2.0e-7),
    SchemeCase("RK4", SchemeRK4, 5.0e-3, None, 2.0e-8),
    SchemeCase("RK38", SchemeRK38, 5.0e-3, None, 2.0e-8),
    SchemeCase("BS23", SchemeBogackiShampine, 1.0e-2, Tolerance(atol=1.0e-8, rtol=1.0e-8), 5.0e-7),
    SchemeCase("RKCK", SchemeCashKarp, 1.0e-2, Tolerance(atol=1.0e-9, rtol=1.0e-9), 5.0e-8),
    SchemeCase("RKF45", SchemeFehlberg45, 1.0e-2, Tolerance(atol=1.0e-9, rtol=1.0e-9), 5.0e-8),
    SchemeCase("RKDP", SchemeDormandPrince, 1.0e-2, Tolerance(atol=1.0e-9, rtol=1.0e-9), 5.0e-8),
    SchemeCase("TSIT5", SchemeTsitouras5, 1.0e-2, Tolerance(atol=1.0e-9, rtol=1.0e-9), 5.0e-8),
]


@dataclass(slots=True)
class RiccatiState:
    t: float
    x: float


@dataclass(slots=True)
class RiccatiTranslation:
    dt: float = 0.0
    dx: float = 0.0

    def __call__(self, origin: RiccatiState, result: RiccatiState) -> None:
        result.t = origin.t + self.dt
        result.x = origin.x + self.dx

    def norm(self) -> float:
        return sqrt(self.dt * self.dt + self.dx * self.dx)

    def __add__(self, other: RiccatiTranslation) -> RiccatiTranslation:
        return RiccatiTranslation(self.dt + other.dt, self.dx + other.dx)

    def __rmul__(self, scalar: float) -> RiccatiTranslation:
        return RiccatiTranslation(scalar * self.dt, scalar * self.dx)

    def scale(out: RiccatiTranslation, a: float, x: RiccatiTranslation) -> RiccatiTranslation:
        out.dt = a * x.dt
        out.dx = a * x.dx
        return out

    def combine2(
        out: RiccatiTranslation,
        a0: float,
        x0: RiccatiTranslation,
        a1: float,
        x1: RiccatiTranslation,
    ) -> RiccatiTranslation:
        out.dt = a0 * x0.dt + a1 * x1.dt
        out.dx = a0 * x0.dx + a1 * x1.dx
        return out

    def combine3(
        out: RiccatiTranslation,
        a0: float,
        x0: RiccatiTranslation,
        a1: float,
        x1: RiccatiTranslation,
        a2: float,
        x2: RiccatiTranslation,
    ) -> RiccatiTranslation:
        out.dt = a0 * x0.dt + a1 * x1.dt + a2 * x2.dt
        out.dx = a0 * x0.dx + a1 * x1.dx + a2 * x2.dx
        return out

    def combine4(
        out: RiccatiTranslation,
        a0: float,
        x0: RiccatiTranslation,
        a1: float,
        x1: RiccatiTranslation,
        a2: float,
        x2: RiccatiTranslation,
        a3: float,
        x3: RiccatiTranslation,
    ) -> RiccatiTranslation:
        out.dt = a0 * x0.dt + a1 * x1.dt + a2 * x2.dt + a3 * x3.dt
        out.dx = a0 * x0.dx + a1 * x1.dx + a2 * x2.dx + a3 * x3.dx
        return out

    def combine5(
        out: RiccatiTranslation,
        a0: float,
        x0: RiccatiTranslation,
        a1: float,
        x1: RiccatiTranslation,
        a2: float,
        x2: RiccatiTranslation,
        a3: float,
        x3: RiccatiTranslation,
        a4: float,
        x4: RiccatiTranslation,
    ) -> RiccatiTranslation:
        out.dt = a0 * x0.dt + a1 * x1.dt + a2 * x2.dt + a3 * x3.dt + a4 * x4.dt
        out.dx = a0 * x0.dx + a1 * x1.dx + a2 * x2.dx + a3 * x3.dx + a4 * x4.dx
        return out

    def combine6(
        out: RiccatiTranslation,
        a0: float,
        x0: RiccatiTranslation,
        a1: float,
        x1: RiccatiTranslation,
        a2: float,
        x2: RiccatiTranslation,
        a3: float,
        x3: RiccatiTranslation,
        a4: float,
        x4: RiccatiTranslation,
        a5: float,
        x5: RiccatiTranslation,
    ) -> RiccatiTranslation:
        out.dt = a0 * x0.dt + a1 * x1.dt + a2 * x2.dt + a3 * x3.dt + a4 * x4.dt + a5 * x5.dt
        out.dx = a0 * x0.dx + a1 * x1.dx + a2 * x2.dx + a3 * x3.dx + a4 * x4.dx + a5 * x5.dx
        return out

    def combine7(
        out: RiccatiTranslation,
        a0: float,
        x0: RiccatiTranslation,
        a1: float,
        x1: RiccatiTranslation,
        a2: float,
        x2: RiccatiTranslation,
        a3: float,
        x3: RiccatiTranslation,
        a4: float,
        x4: RiccatiTranslation,
        a5: float,
        x5: RiccatiTranslation,
        a6: float,
        x6: RiccatiTranslation,
    ) -> RiccatiTranslation:
        out.dt = (
            a0 * x0.dt
            + a1 * x1.dt
            + a2 * x2.dt
            + a3 * x3.dt
            + a4 * x4.dt
            + a5 * x5.dt
            + a6 * x6.dt
        )
        out.dx = (
            a0 * x0.dx
            + a1 * x1.dx
            + a2 * x2.dx
            + a3 * x3.dx
            + a4 * x4.dx
            + a5 * x5.dx
            + a6 * x6.dx
        )
        return out

    linear_combine = [scale, combine2, combine3, combine4, combine5, combine6, combine7]


class RiccatiWorkbench:
    def allocate_state(self) -> RiccatiState:
        return RiccatiState(0.0, 0.0)

    def copy_state(self, dst: RiccatiState, src: RiccatiState) -> None:
        dst.t = src.t
        dst.x = src.x

    def allocate_translation(self) -> RiccatiTranslation:
        return RiccatiTranslation()


def exact_solution(t: float) -> float:
    return 0.7 + 0.15 * sin(0.8 * t) + 0.05 * cos(1.7 * t)


def exact_derivative(t: float) -> float:
    return 0.12 * cos(0.8 * t) - 0.085 * sin(1.7 * t)


def linear_coefficient(t: float) -> float:
    return 0.25 * cos(1.3 * t)


def quadratic_coefficient(t: float) -> float:
    return -0.18 + 0.07 * sin(0.9 * t)


def constant_coefficient(t: float) -> float:
    x = exact_solution(t)
    return exact_derivative(t) - linear_coefficient(t) * x - quadratic_coefficient(t) * x * x


class RiccatiDerivative:
    def __call__(self, state: RiccatiState, out: RiccatiTranslation) -> None:
        t = state.t
        x = state.x
        out.dt = 1.0
        out.dx = constant_coefficient(t) + linear_coefficient(t) * x + quadratic_coefficient(t) * x * x


@pytest.mark.slow
@pytest.mark.parametrize("case", SCHEME_CASES, ids=lambda case: case.label)
def test_scheme_matches_time_dependent_riccati_solution(case: SchemeCase) -> None:
    state = RiccatiState(0.0, exact_solution(0.0))
    interval = Interval(present=0.0, step=case.step, stop=STOP)
    scheme = case.scheme_type(RiccatiDerivative(), RiccatiWorkbench())
    marcher = Marcher(scheme, tolerance=case.tolerance)
    integrate = Integrator()

    steps = 0
    for _interval, _state in integrate.live(marcher, interval, state):
        steps += 1

    expected = exact_solution(STOP)
    error = abs(state.x - expected)
    print(
        f"{case.label:>6} | steps={steps:5d} | final={state.x:.12f} "
        f"| exact={expected:.12f} | error={error:.6e}"
    )

    assert abs(state.t - STOP) < 1.0e-12
    assert error < case.max_error
