from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from stark import (
    Integrator,
    InverterGMRES,
    InverterPolicy,
    InverterTolerance,
    Marcher,
    ResolverNewton,
    ResolverPicard,
    ResolverPolicy,
    ResolverTolerance,
    Tolerance,
)
from stark.primitives import Interval
from stark.scheme_library import SchemeBDF2, SchemeKvaerno3, SchemeKvaerno4, SchemeSDIRK21
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

    @staticmethod
    def scale(out: "ScalarTranslation", a: float, x: "ScalarTranslation") -> "ScalarTranslation":
        out.value = a * x.value
        return out

    @staticmethod
    def combine2(
        out: "ScalarTranslation",
        a0: float,
        x0: "ScalarTranslation",
        a1: float,
        x1: "ScalarTranslation",
    ) -> "ScalarTranslation":
        out.value = a0 * x0.value + a1 * x1.value
        return out

    linear_combine = [scale, combine2]


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


def test_resolver_tolerance_matches_general_tolerance_contract() -> None:
    tolerance = ResolverTolerance(atol=1.0e-6, rtol=1.0e-3)

    assert tolerance.bound(2.0) == 0.002001
    assert tolerance.ratio(0.001, 2.0) < 1.0
    assert tolerance.accepts(0.001, 2.0)


def test_resolver_picard_solves_scalar_backward_euler_step() -> None:
    workbench = ScalarWorkbench()
    derivative = ScalarDerivative(rate=-1.0)
    resolver = ResolverPicard(
        workbench,
        tolerance=ResolverTolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=ResolverPolicy(max_iterations=32),
    )
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
        resolver=ResolverPicard(
            workbench,
            tolerance=ResolverTolerance(atol=1.0e-12, rtol=1.0e-12),
            policy=ResolverPolicy(max_iterations=64),
        ),
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
        tolerance=InverterTolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=InverterPolicy(max_iterations=8, restart=4),
    )
    resolver = ResolverNewton(
        workbench,
        inverter=inverter,
        tolerance=ResolverTolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=ResolverPolicy(max_iterations=8),
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
        tolerance=InverterTolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=InverterPolicy(max_iterations=8, restart=4),
    )
    resolver = ResolverNewton(
        workbench,
        inverter=inverter,
        tolerance=ResolverTolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=ResolverPolicy(max_iterations=8),
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


def test_kvaerno3_advances_linear_decay_with_adaptive_control() -> None:
    workbench = ScalarWorkbench()
    derivative = ScalarDerivative(rate=-10.0)
    inverter = InverterGMRES(
        workbench,
        scalar_inner_product,
        tolerance=InverterTolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=InverterPolicy(max_iterations=8, restart=4),
    )
    resolver = ResolverNewton(
        workbench,
        inverter=inverter,
        tolerance=ResolverTolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=ResolverPolicy(max_iterations=8),
    )
    scheme = SchemeKvaerno3(
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

    assert abs(state.value - 0.36787944117144233) < 1.0e-4


def test_kvaerno4_advances_linear_decay_with_adaptive_control() -> None:
    workbench = ScalarWorkbench()
    derivative = ScalarDerivative(rate=-10.0)
    inverter = InverterGMRES(
        workbench,
        scalar_inner_product,
        tolerance=InverterTolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=InverterPolicy(max_iterations=8, restart=4),
    )
    resolver = ResolverNewton(
        workbench,
        inverter=inverter,
        tolerance=ResolverTolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=ResolverPolicy(max_iterations=8),
    )
    scheme = SchemeKvaerno4(
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

    assert abs(state.value - 0.36787944117144233) < 5.0e-5


def test_bdf2_advances_linear_decay_with_adaptive_control() -> None:
    workbench = ScalarWorkbench()
    derivative = ScalarDerivative(rate=-10.0)
    inverter = InverterGMRES(
        workbench,
        scalar_inner_product,
        tolerance=InverterTolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=InverterPolicy(max_iterations=8, restart=4),
    )
    resolver = ResolverNewton(
        workbench,
        inverter=inverter,
        tolerance=ResolverTolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=ResolverPolicy(max_iterations=8),
    )
    scheme = SchemeBDF2(
        derivative,
        workbench,
        linearizer=ScalarLinearizer(rate=-10.0),
        resolver=resolver,
    )
    marcher = Marcher(scheme, tolerance=Tolerance(atol=1.0e-10, rtol=1.0e-6))
    integrate = Integrator()
    interval = Interval(present=0.0, step=0.05, stop=0.2)
    state = ScalarState(1.0)

    for _interval, _state in integrate.live(marcher, interval, state):
        pass

    assert abs(state.value - 0.1353352832366127) < 2.0e-2
