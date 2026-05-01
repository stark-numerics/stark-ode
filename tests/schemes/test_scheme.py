from dataclasses import dataclass

import numpy as np

from stark import Executor, Marcher
from stark.accelerators import Accelerator
from stark.algebraist import Algebraist, AlgebraistField
from stark.auditor import Auditor
from stark.contracts import AccelerationRequest, AccelerationRole
from stark.integrate import Integrator
from stark.monitor import Monitor
from stark.execution.tolerance import Tolerance
from stark.interval import Interval
from stark.resolvents import ResolventPicard
from stark.schemes.explicit_adaptive.bogacki_shampine import SchemeBogackiShampine
from stark.schemes.explicit_adaptive.cash_karp import SchemeCashKarp
from stark.schemes.explicit_adaptive.dormand_prince import SchemeDormandPrince
from stark.schemes.explicit_adaptive.fehlberg45 import SchemeFehlberg45
from stark.schemes.explicit_adaptive.tsitouras5 import SchemeTsitouras5
from stark.schemes.imex_adaptive.ark324l2sa import SchemeKennedyCarpenter32
from stark.schemes.imex_adaptive.ark436l2sa import SchemeKennedyCarpenter43_6
from stark.schemes.imex_adaptive.ark437l2sa import SchemeKennedyCarpenter43_7
from stark.schemes.imex_adaptive.ark548l2sa import SchemeKennedyCarpenter54
from stark.schemes.imex_adaptive.ark548l2sab import SchemeKennedyCarpenter54b
from stark.schemes.imex_fixed.euler import SchemeIMEXEuler
from stark.schemes.explicit_fixed.euler import SchemeEuler
from stark.schemes.explicit_fixed.heun import SchemeHeun
from stark.schemes.explicit_fixed.kutta3 import SchemeKutta3
from stark.schemes.explicit_fixed.midpoint import SchemeMidpoint
from stark.schemes.explicit_fixed.ralston import SchemeRalston
from stark.schemes.explicit_fixed.rk4 import SchemeRK4
from stark.schemes.explicit_fixed.rk38 import SchemeRK38
from stark.schemes.explicit_fixed.ssprk33 import SchemeSSPRK33
from stark.machinery.stage_solve.workspace import SchemeWorkspace
from stark.machinery.stage_solve.workers import ImExStepper
from stark import ImExDerivative


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
    def scale(out: "FastTranslation", a: float, x: "FastTranslation") -> "FastTranslation":
        out.value = a * x.value
        return out

    def combine2(
        out: "FastTranslation",
        a0: float,
        x0: "FastTranslation",
        a1: float,
        x1: "FastTranslation",
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

    def scale(out: "PairwiseOnlyTranslation", a: float, x: "PairwiseOnlyTranslation") -> "PairwiseOnlyTranslation":
        out.value = a * x.value
        return out

    def combine2(
        out: "PairwiseOnlyTranslation",
        a0: float,
        x0: "PairwiseOnlyTranslation",
        a1: float,
        x1: "PairwiseOnlyTranslation",
    ) -> "PairwiseOnlyTranslation":
        out.value = a0 * x0.value + a1 * x1.value
        return out

    linear_combine = [scale, combine2]


class DummyScheme:
    def __init__(self, derivative, workbench, translation) -> None:
        Auditor.require_scheme_inputs(derivative, workbench, translation)
        self.derivative = derivative
        self.workspace = SchemeWorkspace(workbench, translation)

    def scale(self, y, a, x):
        return self.workspace.scale(y, a, x)

    def combine2(self, y, a0, x0, a1, x1):
        return self.workspace.combine2(y, a0, x0, a1, x1)

    def set_apply_delta_safety(self, enabled: bool) -> None:
        self.workspace.set_apply_delta_safety(enabled)

    def snapshot_state(self, state):
        return self.workspace.snapshot_state(state)

    def __call__(self, interval, state, executor: Executor) -> float:
        del interval, state, executor
        return 0.0


class DummyWorkbench:
    def allocate_state(self) -> object:
        return object()

    def copy_state(self, dst: object, src: object) -> None:
        del dst, src

    def allocate_translation(self) -> DummyTranslation:
        return DummyTranslation()


class PairwiseOnlyWorkbench(DummyWorkbench):
    def allocate_translation(self) -> PairwiseOnlyTranslation:
        return PairwiseOnlyTranslation()


@dataclass(slots=True)
class AliasSensitiveTranslation:
    dx: float = 0.0
    dy: float = 0.0

    def __call__(self, origin: dict[str, float], result: dict[str, float]) -> None:
        result["x"] = origin["x"] + self.dx
        result["y"] = origin["x"] - origin["y"] + self.dy

    def norm(self) -> float:
        return abs(self.dx) + abs(self.dy)

    def __add__(self, other: "AliasSensitiveTranslation") -> "AliasSensitiveTranslation":
        return AliasSensitiveTranslation(self.dx + other.dx, self.dy + other.dy)

    def __rmul__(self, scalar: float) -> "AliasSensitiveTranslation":
        return AliasSensitiveTranslation(scalar * self.dx, scalar * self.dy)


class AliasWorkbench:
    def allocate_state(self) -> dict[str, float]:
        return {"x": 0.0, "y": 0.0}

    def copy_state(self, dst: dict[str, float], src: dict[str, float]) -> None:
        dst["x"] = src["x"]
        dst["y"] = src["y"]

    def allocate_translation(self) -> AliasSensitiveTranslation:
        return AliasSensitiveTranslation()


def _dummy_derivative(interval, state, out) -> None:
    del interval, state, out


def _imex_picard(split: ImExDerivative, workbench, tableau):
    return ResolventPicard(split.implicit, workbench, accelerator=Accelerator.none(), tableau=tableau)


def test_scheme_falls_back_to_arithmetic_linear_combination() -> None:
    x0 = DummyTranslation(2.0)
    x1 = DummyTranslation(3.0)
    scheme = DummyScheme(_dummy_derivative, DummyWorkbench(), x0)
    out = DummyTranslation()

    scaled = scheme.scale(out, 4.0, x0)
    combined = scheme.combine2(out, 2.0, x0, -1.0, x1)

    assert scaled.value == 8.0
    assert combined.value == 1.0


def test_scheme_uses_translation_linear_combine_when_available() -> None:
    x0 = FastTranslation(2.0)
    x1 = FastTranslation(3.0)
    scheme = DummyScheme(_dummy_derivative, DummyWorkbench(), x0)
    out_scaled = FastTranslation()
    out_combined = FastTranslation()

    scaled = scheme.scale(out_scaled, 4.0, x0)
    combined = scheme.combine2(out_combined, 2.0, x0, -1.0, x1)

    assert scaled is out_scaled
    assert scaled.value == 8.0
    assert combined is out_combined
    assert combined.value == 1.0


def test_scheme_synthesizes_missing_fast_combines_from_combine2() -> None:
    translations = [PairwiseOnlyTranslation(float(value)) for value in range(1, 13)]
    scheme = DummyScheme(_dummy_derivative, PairwiseOnlyWorkbench(), translations[0])
    out = PairwiseOnlyTranslation()

    terms = []
    for index, translation in enumerate(translations, start=1):
        terms.extend([float(index), translation])

    combined = scheme.workspace.combine12(out, *terms)

    assert combined is out
    assert combined.value == 650.0


def test_scheme_workspace_consumes_algebraist_linear_combine_contract() -> None:
    algebraist = Algebraist(
        fields=(AlgebraistField("value", "value"),),
        fused_up_to=3,
    )

    class AlgebraistTranslation:
        linear_combine = algebraist.linear_combine

        def __init__(self, value=None) -> None:
            self.value = np.zeros(2) if value is None else np.array(value, dtype=float)

        def __call__(self, origin, result) -> None:
            result["value"] = origin["value"] + self.value

        def norm(self) -> float:
            return float(np.sqrt(np.sum(self.value**2)))

    class AlgebraistWorkbench:
        def allocate_state(self) -> dict[str, np.ndarray]:
            return {"value": np.zeros(2)}

        def copy_state(self, dst: dict[str, np.ndarray], src: dict[str, np.ndarray]) -> None:
            dst["value"][...] = src["value"]

        def allocate_translation(self) -> AlgebraistTranslation:
            return AlgebraistTranslation()

    workspace = SchemeWorkspace(AlgebraistWorkbench(), AlgebraistTranslation([1.0, 2.0]))
    out = AlgebraistTranslation()
    left = AlgebraistTranslation([1.0, 2.0])
    right = AlgebraistTranslation([3.0, 4.0])

    combined = workspace.combine2(out, 2.0, left, 3.0, right)

    assert combined is out
    np.testing.assert_allclose(out.value, np.array([11.0, 16.0]))


class RecordingWorkspace:
    def __init__(self) -> None:
        self.combine2_called = False
        self.combine12_called = False

    def scale(self, out, coefficient, translation):
        out.value = coefficient * translation.value
        return out

    def combine2(self, out, *terms):
        del out, terms
        self.combine2_called = True
        raise AssertionError("IMEX accumulation should dispatch to combine12.")

    def combine12(self, out, *terms):
        self.combine12_called = True
        out.value = sum(
            coefficient * translation.value
            for coefficient, translation in zip(terms[0::2], terms[1::2])
        )
        return out


def test_imex_accumulation_dispatches_to_direct_combine12() -> None:
    stepper = object.__new__(ImExStepper)
    workspace = RecordingWorkspace()
    stepper.workspace = workspace
    out = PairwiseOnlyTranslation()
    coefficients = [float(value) for value in range(1, 13)]
    translations = [PairwiseOnlyTranslation(float(value)) for value in range(1, 13)]

    combined = stepper._accumulate_terms(out, 12, coefficients, translations)

    assert combined is out
    assert combined.value == 650.0
    assert workspace.combine12_called
    assert not workspace.combine2_called


def test_scheme_repr_includes_names_and_tableau() -> None:
    split = ImExDerivative(implicit=_dummy_derivative, explicit=_dummy_derivative)
    imex_workbench = DummyWorkbench()
    euler = SchemeEuler(_dummy_derivative, DummyWorkbench())
    heun = SchemeHeun(_dummy_derivative, DummyWorkbench())
    midpoint = SchemeMidpoint(_dummy_derivative, DummyWorkbench())
    ralston = SchemeRalston(_dummy_derivative, DummyWorkbench())
    kutta3 = SchemeKutta3(_dummy_derivative, DummyWorkbench())
    ssprk33 = SchemeSSPRK33(_dummy_derivative, DummyWorkbench())
    rk4 = SchemeRK4(_dummy_derivative, DummyWorkbench())
    rk38 = SchemeRK38(_dummy_derivative, DummyWorkbench())
    rkck = SchemeCashKarp(_dummy_derivative, DummyWorkbench())
    rkf45 = SchemeFehlberg45(_dummy_derivative, DummyWorkbench())
    bs23 = SchemeBogackiShampine(_dummy_derivative, DummyWorkbench())
    rkdp = SchemeDormandPrince(_dummy_derivative, DummyWorkbench())
    tsit5 = SchemeTsitouras5(_dummy_derivative, DummyWorkbench())
    imex_euler = SchemeIMEXEuler(split, imex_workbench, resolvent=_imex_picard(split, imex_workbench, SchemeIMEXEuler.tableau))
    ark324 = SchemeKennedyCarpenter32(split, imex_workbench, resolvent=_imex_picard(split, imex_workbench, SchemeKennedyCarpenter32.tableau))
    ark436 = SchemeKennedyCarpenter43_6(split, imex_workbench, resolvent=_imex_picard(split, imex_workbench, SchemeKennedyCarpenter43_6.tableau))
    ark437 = SchemeKennedyCarpenter43_7(split, imex_workbench, resolvent=_imex_picard(split, imex_workbench, SchemeKennedyCarpenter43_7.tableau))
    ark548 = SchemeKennedyCarpenter54(split, imex_workbench, resolvent=_imex_picard(split, imex_workbench, SchemeKennedyCarpenter54.tableau))
    ark548b = SchemeKennedyCarpenter54b(split, imex_workbench, resolvent=_imex_picard(split, imex_workbench, SchemeKennedyCarpenter54b.tableau))

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
    scheme = SchemeCashKarp(_dummy_derivative, DummyWorkbench())
    interval = Interval(present=0.0, step=0.1, stop=1.0)

    accepted_dt = scheme(interval, object(), Executor(tolerance=Tolerance(atol=1.0e-6)))

    assert accepted_dt == 0.1
    assert interval.step >= 0.1


def test_scheme_applies_translation_without_aliasing_state() -> None:
    def derivative(interval, state: dict[str, float], out: AliasSensitiveTranslation) -> None:
        del interval
        del state
        out.dx = 1.0
        out.dy = 0.0

    scheme = SchemeEuler(derivative, AliasWorkbench())
    interval = Interval(present=0.0, step=1.0, stop=1.0)
    state = {"x": 1.0, "y": 2.0}

    accepted_dt = scheme(interval, state, Tolerance())

    assert accepted_dt == 1.0
    assert state == {"x": 2.0, "y": -1.0}


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


class TimeWorkbench:
    def allocate_state(self) -> TimeState:
        return TimeState()

    def copy_state(self, dst: TimeState, src: TimeState) -> None:
        dst.value = src.value

    def allocate_translation(self) -> TimeTranslation:
        return TimeTranslation()


def test_midpoint_uses_stage_time_for_non_autonomous_derivative() -> None:
    class TimeDerivative:
        def __call__(self, interval: Interval, state: TimeState, out: TimeTranslation) -> None:
            del state
            out.value = interval.present

    scheme = SchemeMidpoint(TimeDerivative(), TimeWorkbench())
    interval = Interval(present=0.0, step=1.0, stop=1.0)
    state = TimeState(0.0)

    accepted_dt = scheme(interval, state, Tolerance())

    assert accepted_dt == 1.0
    assert abs(state.value - 0.5) < 1.0e-12


def test_marcher_binds_executor_accelerator_into_built_in_scheme_derivative() -> None:
    class AcceleratedDerivative:
        def __call__(self, interval: Interval, state: TimeState, out: TimeTranslation) -> None:
            del interval, state
            out.value = 2.0

    class DerivativeWithAcceleration:
        def __call__(self, interval: Interval, state: TimeState, out: TimeTranslation) -> None:
            del interval, state
            out.value = 1.0

        def accelerated(self, accelerator: Accelerator, request: AccelerationRequest):
            if request.role is AccelerationRole.DERIVATIVE and accelerator.name == "none":
                return AcceleratedDerivative()
            return self

    scheme = SchemeEuler(DerivativeWithAcceleration(), TimeWorkbench())
    marcher = Marcher(scheme, Executor(tolerance=Tolerance(), accelerator=Accelerator.none()))
    interval = Interval(present=0.0, step=0.5, stop=0.5)
    state = TimeState(0.0)

    marcher(interval, state)

    assert abs(state.value - 1.0) < 1.0e-12


def test_imex_euler_handles_purely_explicit_split() -> None:
    def implicit(interval: Interval, state: TimeState, out: TimeTranslation) -> None:
        del interval, state
        out.value = 0.0

    def explicit(interval: Interval, state: TimeState, out: TimeTranslation) -> None:
        del interval, state
        out.value = 1.0

    split = ImExDerivative(implicit=implicit, explicit=explicit)
    workbench = TimeWorkbench()
    scheme = SchemeIMEXEuler(split, workbench, resolvent=_imex_picard(split, workbench, SchemeIMEXEuler.tableau))
    interval = Interval(present=0.0, step=0.5, stop=0.5)
    state = TimeState(0.0)

    accepted_dt = scheme(interval, state, Tolerance())

    assert accepted_dt == 0.5
    assert abs(state.value - 0.5) < 1.0e-12


def test_imex_ark324_accepts_constant_split_rhs() -> None:
    def implicit(interval: Interval, state: TimeState, out: TimeTranslation) -> None:
        del interval, state
        out.value = 0.0

    def explicit(interval: Interval, state: TimeState, out: TimeTranslation) -> None:
        del interval, state
        out.value = 1.0

    split = ImExDerivative(implicit=implicit, explicit=explicit)
    workbench = TimeWorkbench()
    scheme = SchemeKennedyCarpenter32(
        split,
        workbench,
        resolvent=_imex_picard(split, workbench, SchemeKennedyCarpenter32.tableau),
    )
    interval = Interval(present=0.0, step=0.25, stop=1.0)
    state = TimeState(0.0)

    accepted_dt = scheme(interval, state, Executor(tolerance=Tolerance(atol=1.0e-6, rtol=1.0e-6)))

    assert accepted_dt == 0.25
    assert abs(state.value - 0.25) < 1.0e-12
    assert interval.step >= 0.25


def test_integrator_monitored_collects_adaptive_step_payloads() -> None:
    scheme = SchemeCashKarp(_dummy_derivative, DummyWorkbench())
    marcher = Marcher(scheme, Executor(tolerance=Tolerance(atol=1.0e-6, rtol=1.0e-6)))
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    monitor = Monitor()

    list(Integrator().monitored(marcher, interval, object(), monitor))

    assert len(monitor.steps) == 2
    assert [round(step.t_start, 12) for step in monitor.steps] == [0.0, 0.1]
    assert [round(step.t_end, 12) for step in monitor.steps] == [0.1, 0.3]
    assert all(step.scheme == "RKCK" for step in monitor.steps)
    assert monitor.steps[0].accepted_dt == 0.1
    assert monitor.steps[0].rejection_count == 0
    assert marcher.monitor is None













