from dataclasses import dataclass

from stark.audit import AuditError, Auditor
from stark.tolerance import Tolerance
from stark.primitives import Interval


@dataclass(slots=True)
class DummyTranslation:
    value: float = 0.0

    def __call__(self, origin: dict[str, float], result: dict[str, float]) -> None:
        result["x"] = origin["x"] + self.value

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: "DummyTranslation") -> "DummyTranslation":
        return DummyTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> "DummyTranslation":
        return DummyTranslation(scalar * self.value)


class DummyWorkbench:
    def allocate_state(self) -> dict[str, float]:
        return {"x": 0.0}

    def copy_state(self, dst: dict[str, float], src: dict[str, float]) -> None:
        dst["x"] = src["x"]

    def allocate_translation(self) -> DummyTranslation:
        return DummyTranslation()


class BadWorkbench:
    def allocate_state(self) -> dict[str, float]:
        return {"x": 0.0}

    def allocate_translation(self) -> DummyTranslation:
        return DummyTranslation()


def derivative(state: dict[str, float], out: DummyTranslation) -> None:
    out.value = state["x"]


class DummyScheme:
    def __call__(self, interval: Interval, state: object, tolerance: Tolerance) -> float:
        del interval, state, tolerance
        return 0.0

    def snapshot_state(self, state: object) -> object:
        return state

    def set_apply_delta_safety(self, enabled: bool) -> None:
        del enabled


def test_auditor_reports_ready_configuration() -> None:
    auditor = Auditor(
        state={"x": 1.0},
        derivative=derivative,
        translation=DummyTranslation(),
        workbench=DummyWorkbench(),
        interval=Interval(0.0, 0.1, 1.0),
        scheme=DummyScheme(),
        tolerance=Tolerance(),
    )

    assert auditor.ok
    report = str(auditor)
    assert "STARK audit checklist" in report
    assert "Object" in report
    assert "Required behavior" in report
    assert "Present" in report
    assert "yes" in report
    assert report.index("Interval") < report.index("Derivative")
    assert report.index("Derivative") < report.index("Translation")
    assert report.index("Translation") < report.index("Workbench")
    assert report.index("Workbench") < report.index("Scheme")
    assert "Overall: ready." in report


def test_auditor_reports_missing_requirements() -> None:
    auditor = Auditor(
        state={"x": 1.0},
        derivative=derivative,
        translation=DummyTranslation(),
        workbench=BadWorkbench(),
        interval=Interval(0.0, 0.1, 1.0),
    )

    assert not auditor.ok
    report = str(auditor)
    assert "no" in report
    assert "copy_state" in report


def test_require_scheme_inputs_raises_helpful_error() -> None:
    try:
        Auditor.require_scheme_inputs(derivative, BadWorkbench(), DummyTranslation())
    except AuditError as exc:
        message = str(exc)
        assert "Workbench provides copy_state" in message
        assert "Overall: incomplete." in message
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("Expected the audit to reject a bad workbench.")


def test_require_linear_residual_rejects_missing_linearize() -> None:
    class ResidualOnly:
        def __call__(self, out, block) -> None:
            del out, block

    try:
        Auditor.require_linear_residual(ResidualOnly())
    except AuditError as exc:
        message = str(exc)
        assert "Residual provides linearize(out, block)" in message
        assert "Overall: incomplete." in message
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("Expected the audit to reject a residual without linearize().")

