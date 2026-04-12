from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from stark import Integrator, Inversion, InverterGMRES, Marcher, Resolution, ResolverNewton, ResolverPicard, Tolerance
from stark.primitives import Interval
from stark.scheme_library import SchemeSDIRK21
from stark.scheme_library.implicit import SchemeBackwardEuler


@dataclass(slots=True)
class ScalarTranslation:
    value: float = 0.0

    def __call__(self, origin: "ScalarState", result: "ScalarState") -> None:
        result.value = origin.value + self.value

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: "ScalarTranslation") -> "ScalarTranslation":
        return ScalarTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> "ScalarTranslation":
        return ScalarTranslation(scalar * self.value)


@dataclass(slots=True)
class ScalarState:
    value: float = 0.0


class ScalarWorkbench:
    def allocate_state(self) -> ScalarState:
        return ScalarState()

    def copy_state(self, dst: ScalarState, src: ScalarState) -> None:
        dst.value = src.value

    def allocate_translation(self) -> ScalarTranslation:
        return ScalarTranslation()


class ScalarDerivative:
    def __init__(self, rate: float) -> None:
        self.rate = rate

    def __call__(self, state: ScalarState, out: ScalarTranslation) -> None:
        out.value = self.rate * state.value


class ScalarLinearizer:
    def __init__(self, rate: float) -> None:
        self.rate = rate

    def __call__(self, out, state: ScalarState) -> None:
        del state

        def apply(result: ScalarTranslation, translation: ScalarTranslation) -> None:
            result.value = self.rate * translation.value

        out.apply = apply


def scalar_inner_product(left: ScalarTranslation, right: ScalarTranslation) -> float:
    return left.value * right.value


def test_resolution_matches_tolerance_style_contract() -> None:
    resolution = Resolution(atol=1.0e-6, rtol=1.0e-3, max_iterations=12)

    assert resolution.bound(2.0) == 0.002001
    assert resolution.ratio(0.001, 2.0) < 1.0
    assert resolution.accepts(0.001, 2.0)


def test_resolver_picard_solves_scalar_backward_euler_step() -> None:
    workbench = ScalarWorkbench()
    derivative = ScalarDerivative(rate=-1.0)
    resolver = ResolverPicard(workbench, resolution=Resolution(atol=1.0e-12, rtol=1.0e-12, max_iterations=32))
    scheme = SchemeBackwardEuler(derivative, workbench, resolver=resolver)
    marcher = Marcher(scheme, tolerance=Tolerance())
    interval = Interval(present=0.0, step=0.1, stop=0.1)
    state = ScalarState(1.0)

    marcher(interval, state)

    assert abs(state.value - (1.0 / 1.1)) < 1.0e-10
    assert interval.present == 0.1
    assert interval.step == 0.0


def test_backward_euler_matches_closed_form_for_quadratic_decay() -> None:
    class QuadraticDerivative:
        def __call__(self, state: ScalarState, out: ScalarTranslation) -> None:
            out.value = -(state.value ** 2)

    workbench = ScalarWorkbench()
    scheme = SchemeBackwardEuler(
        QuadraticDerivative(),
        workbench,
        resolver=ResolverPicard(workbench, resolution=Resolution(atol=1.0e-12, rtol=1.0e-12, max_iterations=64)),
    )
    marcher = Marcher(scheme, tolerance=Tolerance())
    interval = Interval(present=0.0, step=0.1, stop=0.1)
    state = ScalarState(1.0)

    marcher(interval, state)

    expected = (-1.0 + sqrt(1.4)) / 0.2
    assert abs(state.value - expected) < 1.0e-10


def test_resolver_newton_solves_scalar_backward_euler_step() -> None:
    workbench = ScalarWorkbench()
    derivative = ScalarDerivative(rate=-10.0)
    inverter = InverterGMRES(
        workbench,
        scalar_inner_product,
        inversion=Inversion(atol=1.0e-12, rtol=1.0e-12, max_iterations=8, restart=4),
    )
    resolver = ResolverNewton(
        workbench,
        inverter=inverter,
        resolution=Resolution(atol=1.0e-12, rtol=1.0e-12, max_iterations=8),
    )
    scheme = SchemeBackwardEuler(
        derivative,
        workbench,
        linearizer=ScalarLinearizer(rate=-10.0),
        resolver=resolver,
    )
    marcher = Marcher(scheme, tolerance=Tolerance())
    interval = Interval(present=0.0, step=0.1, stop=0.1)
    state = ScalarState(1.0)

    marcher(interval, state)

    assert abs(state.value - 0.5) < 1.0e-10


def test_resolver_newton_requires_linearized_residual() -> None:
    workbench = ScalarWorkbench()
    derivative = ScalarDerivative(rate=-1.0)
    inverter = InverterGMRES(workbench, scalar_inner_product)
    resolver = ResolverNewton(workbench, inverter=inverter)
    scheme = SchemeBackwardEuler(
        derivative,
        workbench,
        resolver=resolver,
    )
    marcher = Marcher(scheme, tolerance=Tolerance())
    interval = Interval(present=0.0, step=0.1, stop=0.1)
    state = ScalarState(1.0)

    try:
        marcher(interval, state)
    except RuntimeError as exc:
        assert "linearizer" in str(exc).lower()
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("Expected Newton-backed backward Euler to require a linearizer.")


def test_sdirk21_advances_linear_decay_with_adaptive_control() -> None:
    workbench = ScalarWorkbench()
    derivative = ScalarDerivative(rate=-10.0)
    inverter = InverterGMRES(
        workbench,
        scalar_inner_product,
        inversion=Inversion(atol=1.0e-12, rtol=1.0e-12, max_iterations=8, restart=4),
    )
    resolver = ResolverNewton(
        workbench,
        inverter=inverter,
        resolution=Resolution(atol=1.0e-12, rtol=1.0e-12, max_iterations=8),
    )
    scheme = SchemeSDIRK21(
        derivative,
        workbench,
        linearizer=ScalarLinearizer(rate=-10.0),
        resolver=resolver,
    )
    marcher = Marcher(scheme, tolerance=Tolerance(atol=1.0e-10, rtol=1.0e-6))
    integrate = Integrator()
    interval = Interval(present=0.0, step=0.1, stop=0.1)
    state = ScalarState(1.0)

    for _interval, _state in integrate.live(marcher, interval, state):
        pass

    assert abs(state.value - 0.36787944117144233) < 5.0e-4
