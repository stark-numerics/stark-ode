from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from stark.core import IntegratorStepper
from stark.engines.accelerators import AcceleratorNone
from stark.engines.algebraist.arity import AlgebraistArity
from stark.engines.algebraist.generator import AlgebraistGeneratorLinearCombine
from stark.core.auditor import Auditor
from stark.core.integrator.integrator import Integrator
from stark.diagnostics.monitor import Monitor
from stark import Tolerance
from stark.core.interval import Interval
from stark.methods.resolvents import ResolventPicard
from stark.methods.schemes.explicit.adaptive.bogacki_shampine import SchemeBogackiShampine
from stark.methods.schemes.explicit.adaptive.cash_karp import SchemeCashKarp
from stark.methods.schemes.explicit.adaptive.dormand_prince import SchemeDormandPrince
from stark.methods.schemes.explicit.adaptive.fehlberg45 import SchemeFehlberg45
from stark.methods.schemes.explicit.adaptive.tsitouras5 import SchemeTsitouras5
from stark.methods.schemes.imex.adaptive.kennedy_carpenter32 import SchemeKennedyCarpenter32
from stark.methods.schemes.imex.adaptive.kennedy_carpenter43_6 import SchemeKennedyCarpenter43_6
from stark.methods.schemes.imex.adaptive.kennedy_carpenter43_7 import SchemeKennedyCarpenter43_7
from stark.methods.schemes.imex.adaptive.kennedy_carpenter54 import SchemeKennedyCarpenter54
from stark.methods.schemes.imex.adaptive.kennedy_carpenter54b import SchemeKennedyCarpenter54b
from stark.methods.schemes.imex.fixed.euler import SchemeIMEXEuler
from stark.methods.schemes.explicit.fixed.euler import SchemeEuler
from stark.methods.schemes.explicit.fixed.heun import SchemeHeun
from stark.methods.schemes.explicit.fixed.kutta3 import SchemeKutta3
from stark.methods.schemes.explicit.fixed.midpoint import SchemeMidpoint
from stark.methods.schemes.explicit.fixed.ralston import SchemeRalston
from stark.methods.schemes.explicit.fixed.rk4 import SchemeRK4
from stark.methods.schemes.explicit.fixed.rk38 import SchemeRK38
from stark.methods.schemes.explicit.fixed.ssprk33 import SchemeSSPRK33
from stark.methods.schemes.execution.step_support import SchemeStepSupport
from stark import Dynamics
from stark.core.contracts import DynamicsSplitLike, IntervalLike
from stark.problem.frame import Field, Frame


@dataclass(slots=True)
class DummyTranslation:
    value: float = 0.0

    def __call__(self, origin: float, result: float) -> None:
        del origin, result

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: "DummyTranslation") -> "DummyTranslation":
        return DummyTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> "DummyTranslation":
        return DummyTranslation(self.value * scalar)


@dataclass(slots=True)
class FastTranslation(DummyTranslation):
    @staticmethod
    def scale(a: float, x: "FastTranslation", out: "FastTranslation") -> "FastTranslation":
        out.value = a * x.value
        return out

    @staticmethod
    def combine2(
        a0: float,
        x0: "FastTranslation",
        a1: float,
        x1: "FastTranslation",
        out: "FastTranslation",
    ) -> "FastTranslation":
        out.value = a0 * x0.value + a1 * x1.value
        return out

    linear_combine = [scale, combine2]


@dataclass(slots=True)
class PairwiseOnlyTranslation:
    value: float = 0.0

    def __call__(self, origin: float, result: float) -> None:
        del origin, result

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: "PairwiseOnlyTranslation") -> "PairwiseOnlyTranslation":
        del other
        raise AssertionError("Synthesized fast combines should not call __add__.")

    def __rmul__(self, scalar: float) -> "PairwiseOnlyTranslation":
        del scalar
        raise AssertionError("Synthesized fast combines should not call __rmul__.")

    @staticmethod
    def scale(a: float, x: "PairwiseOnlyTranslation", out: "PairwiseOnlyTranslation") -> "PairwiseOnlyTranslation":
        out.value = a * x.value
        return out

    @staticmethod
    def combine2(
        a0: float,
        x0: "PairwiseOnlyTranslation",
        a1: float,
        x1: "PairwiseOnlyTranslation",
        out: "PairwiseOnlyTranslation",
    ) -> "PairwiseOnlyTranslation":
        out.value = a0 * x0.value + a1 * x1.value
        return out

    linear_combine = [scale, combine2]


class DummyScheme:
    def __init__(self, dynamics, allocator, translation) -> None:
        Auditor.require_scheme_inputs(dynamics, allocator, translation)
        self.dynamics = dynamics
        self.workspace = SchemeStepSupport(allocator, translation)

    def scale(self, a, x, y):
        return self.workspace.scale(a, x, y)

    def combine2(self, a0, x0, a1, x1, y):
        return self.workspace.combine2(a0, x0, a1, x1, y)

    def snapshot_state(self, state):
        return self.workspace.snapshot_state(state)

    def __call__(self, interval, state) -> float:
        del interval, state
        return 0.0


class DummyAllocator:
    def allocate_state(self) -> object:
        return object()

    def copy_state(self, source: object, out: object) -> None:
        del out, source

    def allocate_translation(self) -> DummyTranslation:
        return DummyTranslation()


class PairwiseOnlyAllocator:
    def allocate_state(self) -> object:
        return object()

    def copy_state(self, source: object, out: object) -> None:
        del out, source

    def allocate_translation(self) -> PairwiseOnlyTranslation:
        return PairwiseOnlyTranslation()


def _dummy_dynamics(interval, state, out) -> None:
    del interval, state, out


def _imex_picard(split: DynamicsSplitLike, allocator, tableau):
    return ResolventPicard(allocator, accelerator=AcceleratorNone(), tableau=tableau)


def test_scheme_falls_back_to_arithmetic_linear_combination() -> None:
    x0 = DummyTranslation(2.0)
    x1 = DummyTranslation(3.0)
    scheme = DummyScheme(_dummy_dynamics, DummyAllocator(), x0)
    out = DummyTranslation()

    scaled = scheme.scale(4.0, x0, out)
    combined = scheme.combine2(2.0, x0, -1.0, x1, out)

    assert scaled.value == 8.0
    assert combined.value == 1.0


def test_scheme_uses_translation_linear_combine_when_available() -> None:
    x0 = FastTranslation(2.0)
    x1 = FastTranslation(3.0)
    scheme = DummyScheme(_dummy_dynamics, DummyAllocator(), x0)
    out_scaled = FastTranslation()
    out_combined = FastTranslation()

    scaled = scheme.scale(4.0, x0, out_scaled)
    combined = scheme.combine2(2.0, x0, -1.0, x1, out_combined)

    assert scaled is out_scaled
    assert scaled.value == 8.0
    assert combined is out_combined
    assert combined.value == 1.0


def test_scheme_synthesizes_missing_fast_combines_from_combine2() -> None:
    translations = [PairwiseOnlyTranslation(float(value)) for value in range(1, 13)]
    scheme = DummyScheme(_dummy_dynamics, PairwiseOnlyAllocator(), translations[0])
    out = PairwiseOnlyTranslation()

    terms = []
    for index, translation in enumerate(translations, start=1):
        terms.extend([float(index), translation])

    combined = scheme.workspace.combine12(*terms, out)

    assert combined is out
    assert combined.value == 650.0


def test_scheme_step_support_consumes_algebraist_linear_combine_contract() -> None:
    class DummyAlgebraistTranslation:
        linear_combine: ClassVar[tuple[Any, ...]] = ()

        def __init__(self, value=None) -> None:
            self.value = np.zeros(2) if value is None else np.array(value, dtype=float)

        def __call__(self, origin, result) -> None:
            result["value"] = origin["value"] + self.value

        def norm(self) -> float:
            return float(np.sqrt(np.sum(self.value**2)))

        def __add__(self, other):
            del other
            raise AssertionError("Generated linear-combine path should not use __add__.")

        def __mul__(self, scalar):
            del scalar
            raise AssertionError("Generated linear-combine path should not use __mul__.")

        def __rmul__(self, scalar):
            del scalar
            raise AssertionError("Generated linear-combine path should not use __rmul__.")

    class DummyAlgebraistAllocator:
        def allocate_state(self) -> dict[str, np.ndarray]:
            return {"value": np.zeros(2)}

        def copy_state(self, source: dict[str, np.ndarray], out: dict[str, np.ndarray]) -> None:
            out["value"][...] = source["value"]

        def allocate_translation(self) -> DummyAlgebraistTranslation:
            return DummyAlgebraistTranslation()

    allocator = DummyAlgebraistAllocator()
    provider = AlgebraistGeneratorLinearCombine(
        translation=DummyAlgebraistTranslation([1.0, 2.0]),
        allocator=allocator,
        frame=Frame(
            fields=(Field("value", translation="value"),),
        ),
    )
    DummyAlgebraistTranslation.linear_combine = (
        provider.provide(AlgebraistArity(1)),
        provider.provide(AlgebraistArity(2)),
        provider.provide(AlgebraistArity(3)),
    )

    workspace = SchemeStepSupport(allocator, DummyAlgebraistTranslation([1.0, 2.0]))
    out = DummyAlgebraistTranslation()
    left = DummyAlgebraistTranslation([1.0, 2.0])
    right = DummyAlgebraistTranslation([3.0, 4.0])

    combined = workspace.combine2(2.0, left, 3.0, right, out)

    assert combined is out
    np.testing.assert_allclose(out.value, np.array([11.0, 16.0]))


def test_scheme_repr_includes_names_and_tableau() -> None:
    split = Dynamics.split(implicit=_dummy_dynamics, explicit=_dummy_dynamics)
    imex_allocator = DummyAllocator()
    euler = SchemeEuler(_dummy_dynamics, DummyAllocator())
    heun = SchemeHeun(_dummy_dynamics, DummyAllocator())
    midpoint = SchemeMidpoint(_dummy_dynamics, DummyAllocator())
    ralston = SchemeRalston(_dummy_dynamics, DummyAllocator())
    kutta3 = SchemeKutta3(_dummy_dynamics, DummyAllocator())
    ssprk33 = SchemeSSPRK33(_dummy_dynamics, DummyAllocator())
    rk4 = SchemeRK4(_dummy_dynamics, DummyAllocator())
    rk38 = SchemeRK38(_dummy_dynamics, DummyAllocator())
    rkck = SchemeCashKarp(_dummy_dynamics, DummyAllocator())
    rkf45 = SchemeFehlberg45(_dummy_dynamics, DummyAllocator())
    bs23 = SchemeBogackiShampine(_dummy_dynamics, DummyAllocator())
    rkdp = SchemeDormandPrince(_dummy_dynamics, DummyAllocator())
    tsit5 = SchemeTsitouras5(_dummy_dynamics, DummyAllocator())
    imex_euler = SchemeIMEXEuler(split, imex_allocator, resolvent=_imex_picard(split, imex_allocator, SchemeIMEXEuler.tableau))
    ark324 = SchemeKennedyCarpenter32(split, imex_allocator, resolvent=_imex_picard(split, imex_allocator, SchemeKennedyCarpenter32.tableau))
    ark436 = SchemeKennedyCarpenter43_6(split, imex_allocator, resolvent=_imex_picard(split, imex_allocator, SchemeKennedyCarpenter43_6.tableau))
    ark437 = SchemeKennedyCarpenter43_7(split, imex_allocator, resolvent=_imex_picard(split, imex_allocator, SchemeKennedyCarpenter43_7.tableau))
    ark548 = SchemeKennedyCarpenter54(split, imex_allocator, resolvent=_imex_picard(split, imex_allocator, SchemeKennedyCarpenter54.tableau))
    ark548b = SchemeKennedyCarpenter54b(split, imex_allocator, resolvent=_imex_picard(split, imex_allocator, SchemeKennedyCarpenter54b.tableau))

    euler_repr = repr(euler)
    heun_repr = repr(heun)
    midpoint_repr = repr(midpoint)
    ralston_repr = repr(ralston)
    kutta3_repr = repr(kutta3)
    ssprk33_repr = repr(ssprk33)
    rk4_repr = repr(rk4)
    rk38_repr = repr(rk38)
    rkck_repr = repr(rkck)
    rkf45_repr = repr(rkf45)
    bs23_repr = repr(bs23)
    rkdp_repr = repr(rkdp)
    tsit5_repr = repr(tsit5)
    imex_euler_repr = repr(imex_euler)
    ark324_repr = repr(ark324)
    ark436_repr = repr(ark436)
    ark437_repr = repr(ark437)
    ark548_repr = repr(ark548)
    ark548b_repr = repr(ark548b)

    assert "Euler" in euler_repr
    assert "Forward Euler" in euler_repr
    assert "Butcher tableau" in euler_repr
    assert "Heun" in heun_repr
    assert "Butcher tableau" in heun_repr
    assert "Midpoint" in midpoint_repr
    assert "Explicit Midpoint" in midpoint_repr
    assert "Butcher tableau" in midpoint_repr
    assert "Ralston" in ralston_repr
    assert "Butcher tableau" in ralston_repr
    assert "Kutta3" in kutta3_repr
    assert "Kutta Third-Order" in kutta3_repr
    assert "Butcher tableau" in kutta3_repr
    assert "SSPRK33" in ssprk33_repr
    assert "SSP RK33" in ssprk33_repr
    assert "Butcher tableau" in ssprk33_repr
    assert "RK4" in rk4_repr
    assert "Classical Runge-Kutta" in rk4_repr
    assert "Butcher tableau" in rk4_repr
    assert "RK38" in rk38_repr
    assert "3/8 Rule Runge-Kutta" in rk38_repr
    assert "Butcher tableau" in rk38_repr
    assert "BS23" in bs23_repr
    assert "Bogacki-Shampine" in bs23_repr
    assert "Butcher tableau" in bs23_repr
    assert "RKCK" in rkck_repr
    assert "Cash Karp" in rkck_repr
    assert "Butcher tableau" in rkck_repr
    assert "RKF45" in rkf45_repr
    assert "Fehlberg 4(5)" in rkf45_repr
    assert "Butcher tableau" in rkf45_repr
    assert "RKDP" in rkdp_repr
    assert "Dormand-Prince" in rkdp_repr
    assert "Butcher tableau" in rkdp_repr
    assert "TSIT5" in tsit5_repr
    assert "Tsitouras 5" in tsit5_repr
    assert "Butcher tableau" in tsit5_repr
    assert "IMEXEuler" in imex_euler_repr
    assert "IMEX Euler" in imex_euler_repr
    assert "IMEX Butcher tableau" in imex_euler_repr
    assert "KC32" in ark324_repr
    assert "Kennedy-Carpenter 3(2)" in ark324_repr
    assert "IMEX Butcher tableau" in ark324_repr
    assert "KC43-6" in ark436_repr
    assert "Kennedy-Carpenter 4(3) 6-stage" in ark436_repr
    assert "IMEX Butcher tableau" in ark436_repr
    assert "KC43-7" in ark437_repr
    assert "Kennedy-Carpenter 4(3) 7-stage" in ark437_repr
    assert "IMEX Butcher tableau" in ark437_repr
    assert "KC54" in ark548_repr
    assert "Kennedy-Carpenter 5(4)" in ark548_repr
    assert "IMEX Butcher tableau" in ark548_repr
    assert "KC54b" in ark548b_repr
    assert "Kennedy-Carpenter 5(4) b" in ark548b_repr
    assert "IMEX Butcher tableau" in ark548b_repr


def test_adaptive_scheme_updates_next_interval_step() -> None:
    scheme = SchemeCashKarp(_dummy_dynamics, DummyAllocator())
    interval = Interval(present=0.0, step=0.1, stop=1.0)

    accepted_dt = scheme(interval, object())

    assert accepted_dt == 0.1
    assert interval.step >= 0.1


@dataclass(slots=True)
class TimeState:
    value: float = 0.0


@dataclass(slots=True)
class TimeTranslation:
    value: float = 0.0

    def __call__(self, origin: TimeState, result: TimeState) -> None:
        result.value = origin.value + self.value

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: "TimeTranslation") -> "TimeTranslation":
        return TimeTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> "TimeTranslation":
        return TimeTranslation(scalar * self.value)


class TimeAllocator:
    def allocate_state(self) -> TimeState:
        return TimeState()

    def copy_state(self, source: TimeState, out: TimeState) -> None:
        out.value = source.value

    def allocate_translation(self) -> TimeTranslation:
        return TimeTranslation()


def test_midpoint_uses_stage_time_for_non_autonomous_dynamics() -> None:
    class TimeDynamics:
        def __call__(self, interval: IntervalLike, state: TimeState, out: TimeTranslation) -> None:
            del state
            out.value = interval.present

    scheme = SchemeMidpoint(TimeDynamics(), TimeAllocator())
    interval = Interval(present=0.0, step=1.0, stop=1.0)
    state = TimeState(0.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == 1.0
    assert abs(state.value - 0.5) < 1.0e-12


def test_stepper_keeps_explicitly_supplied_scheme_dynamics() -> None:
    class ConstantDynamics:
        def __call__(self, interval: IntervalLike, state: TimeState, out: TimeTranslation) -> None:
            del interval, state
            out.value = 1.0

    scheme = SchemeEuler(ConstantDynamics(), TimeAllocator())
    stepper = IntegratorStepper(scheme)
    interval = Interval(present=0.0, step=0.5, stop=0.5)
    state = TimeState(0.0)

    stepper(interval, state)

    assert abs(state.value - 0.5) < 1.0e-12


def test_imex_euler_handles_purely_explicit_split() -> None:
    def implicit(interval: IntervalLike, state: TimeState, out: TimeTranslation) -> None:
        del interval, state
        out.value = 0.0

    def explicit(interval: IntervalLike, state: TimeState, out: TimeTranslation) -> None:
        del interval, state
        out.value = 1.0

    split = Dynamics.split(implicit=implicit, explicit=explicit)
    allocator = TimeAllocator()
    scheme = SchemeIMEXEuler(split, allocator, resolvent=_imex_picard(split, allocator, SchemeIMEXEuler.tableau))
    interval = Interval(present=0.0, step=0.5, stop=0.5)
    state = TimeState(0.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == 0.5
    assert abs(state.value - 0.5) < 1.0e-12


def test_imex_ark324_accepts_constant_split_rhs() -> None:
    def implicit(interval: IntervalLike, state: TimeState, out: TimeTranslation) -> None:
        del interval, state
        out.value = 0.0

    def explicit(interval: IntervalLike, state: TimeState, out: TimeTranslation) -> None:
        del interval, state
        out.value = 1.0

    split = Dynamics.split(implicit=implicit, explicit=explicit)
    allocator = TimeAllocator()
    scheme = SchemeKennedyCarpenter32(
        split,
        allocator,
        resolvent=_imex_picard(split, allocator, SchemeKennedyCarpenter32.tableau),
    )
    interval = Interval(present=0.0, step=0.25, stop=1.0)
    state = TimeState(0.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == 0.25
    assert abs(state.value - 0.25) < 1.0e-12
    assert interval.step >= 0.25


def test_scheme_monitor_collects_adaptive_step_payloads_during_integration() -> None:
    monitor = Monitor()
    scheme = SchemeCashKarp(_dummy_dynamics, DummyAllocator(), monitor=monitor.scheme)
    stepper = IntegratorStepper(scheme)
    interval = Interval(present=0.0, step=0.1, stop=0.3)

    list(Integrator().mutating_trajectory(stepper, interval, object()))

    assert len(monitor.scheme.adaptive_steps) == 2
    assert [round(step.t_start, 12) for step in monitor.scheme.adaptive_steps] == [0.0, 0.1]
    assert [round(step.t_end, 12) for step in monitor.scheme.adaptive_steps] == [0.1, 0.3]
    assert all(step.scheme == "RKCK" for step in monitor.scheme.adaptive_steps)
    assert monitor.scheme.adaptive_steps[0].accepted_dt == 0.1
    assert monitor.scheme.adaptive_steps[0].rejection_count == 0
