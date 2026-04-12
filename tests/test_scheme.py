from dataclasses import dataclass

from stark.audit import Auditor
from stark.control import Tolerance
from stark.primitives import Interval
from stark.scheme_library.adaptive.bogacki_shampine import SchemeBogackiShampine
from stark.scheme_library.adaptive.cash_karp import SchemeCashKarp
from stark.scheme_library.adaptive.dormand_prince import SchemeDormandPrince
from stark.scheme_library.adaptive.fehlberg45 import SchemeFehlberg45
from stark.scheme_library.adaptive.tsitouras5 import SchemeTsitouras5
from stark.scheme_library.fixed_step.euler import SchemeEuler
from stark.scheme_library.fixed_step.heun import SchemeHeun
from stark.scheme_library.fixed_step.kutta3 import SchemeKutta3
from stark.scheme_library.fixed_step.midpoint import SchemeMidpoint
from stark.scheme_library.fixed_step.ralston import SchemeRalston
from stark.scheme_library.fixed_step.rk4 import SchemeRK4
from stark.scheme_library.fixed_step.rk38 import SchemeRK38
from stark.scheme_library.fixed_step.ssprk33 import SchemeSSPRK33
from stark.scheme_support.workspace import SchemeWorkspace


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

    def __call__(self, interval, state, tolerance: Tolerance) -> float:
        del interval, state, tolerance
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


def test_scheme_falls_back_to_arithmetic_linear_combination() -> None:
    x0 = DummyTranslation(2.0)
    x1 = DummyTranslation(3.0)
    scheme = DummyScheme(lambda state, out: None, DummyWorkbench(), x0)
    out = DummyTranslation()

    scaled = scheme.scale(out, 4.0, x0)
    combined = scheme.combine2(out, 2.0, x0, -1.0, x1)

    assert scaled.value == 8.0
    assert combined.value == 1.0


def test_scheme_uses_translation_linear_combine_when_available() -> None:
    x0 = FastTranslation(2.0)
    x1 = FastTranslation(3.0)
    scheme = DummyScheme(lambda state, out: None, DummyWorkbench(), x0)
    out_scaled = FastTranslation()
    out_combined = FastTranslation()

    scaled = scheme.scale(out_scaled, 4.0, x0)
    combined = scheme.combine2(out_combined, 2.0, x0, -1.0, x1)

    assert scaled is out_scaled
    assert scaled.value == 8.0
    assert combined is out_combined
    assert combined.value == 1.0


def test_scheme_synthesizes_missing_fast_combines_from_combine2() -> None:
    translations = [PairwiseOnlyTranslation(float(value)) for value in range(1, 8)]
    scheme = DummyScheme(lambda state, out: None, PairwiseOnlyWorkbench(), translations[0])
    out = PairwiseOnlyTranslation()

    combined = scheme.workspace.combine7(
        out,
        1.0,
        translations[0],
        2.0,
        translations[1],
        3.0,
        translations[2],
        4.0,
        translations[3],
        5.0,
        translations[4],
        6.0,
        translations[5],
        7.0,
        translations[6],
    )

    assert combined is out
    assert combined.value == 140.0


def test_scheme_repr_includes_names_and_tableau() -> None:
    euler = SchemeEuler(lambda state, out: None, DummyWorkbench())
    heun = SchemeHeun(lambda state, out: None, DummyWorkbench())
    midpoint = SchemeMidpoint(lambda state, out: None, DummyWorkbench())
    ralston = SchemeRalston(lambda state, out: None, DummyWorkbench())
    kutta3 = SchemeKutta3(lambda state, out: None, DummyWorkbench())
    ssprk33 = SchemeSSPRK33(lambda state, out: None, DummyWorkbench())
    rk4 = SchemeRK4(lambda state, out: None, DummyWorkbench())
    rk38 = SchemeRK38(lambda state, out: None, DummyWorkbench())
    rkck = SchemeCashKarp(lambda state, out: None, DummyWorkbench())
    rkf45 = SchemeFehlberg45(lambda state, out: None, DummyWorkbench())
    bs23 = SchemeBogackiShampine(lambda state, out: None, DummyWorkbench())
    rkdp = SchemeDormandPrince(lambda state, out: None, DummyWorkbench())
    tsit5 = SchemeTsitouras5(lambda state, out: None, DummyWorkbench())

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


def test_adaptive_scheme_updates_next_interval_step() -> None:
    scheme = SchemeCashKarp(lambda state, out: None, DummyWorkbench())
    interval = Interval(present=0.0, step=0.1, stop=1.0)

    accepted_dt = scheme(interval, object(), Tolerance(atol=1.0e-6))

    assert accepted_dt == 0.1
    assert interval.step >= 0.1


def test_scheme_applies_translation_without_aliasing_state() -> None:
    def derivative(state: dict[str, float], out: AliasSensitiveTranslation) -> None:
        del state
        out.dx = 1.0
        out.dy = 0.0

    scheme = SchemeEuler(derivative, AliasWorkbench())
    interval = Interval(present=0.0, step=1.0, stop=1.0)
    state = {"x": 1.0, "y": 2.0}

    accepted_dt = scheme(interval, state, Tolerance())

    assert accepted_dt == 1.0
    assert state == {"x": 2.0, "y": -1.0}
