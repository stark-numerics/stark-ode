from dataclasses import dataclass

from stark.engines.shared.accelerators import AcceleratorNone
from stark import Derivative, DerivativeStyle
from stark.core.auditor import AuditError, Auditor
from stark import Tolerance
from stark.core.interval import Interval
from stark.problem.derivative import DerivativeSplit


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

    @staticmethod
    def scale(scalar: float, translation: "DummyTranslation", out: "DummyTranslation") -> "DummyTranslation":
        out.value = scalar * translation.value
        return out

    @staticmethod
    def combine2(
        a0: float,
        x0: "DummyTranslation",
        a1: float,
        x1: "DummyTranslation",
        out: "DummyTranslation",
    ) -> "DummyTranslation":
        out.value = a0 * x0.value + a1 * x1.value
        return out

    linear_combine = [scale, combine2]


class DummyAllocator:
    def allocate_state(self) -> dict[str, float]:
        return {"x": 0.0}

    def copy_state(self, source: dict[str, float], out: dict[str, float]) -> None:
        out["x"] = source["x"]

    def allocate_translation(self) -> DummyTranslation:
        return DummyTranslation()


class BadAllocator:
    def allocate_state(self) -> dict[str, float]:
        return {"x": 0.0}

    def allocate_translation(self) -> DummyTranslation:
        return DummyTranslation()


def derivative(interval: Interval, state: dict[str, float], out: DummyTranslation) -> None:
    del interval
    out.value = state["x"]


class DummyScheme:
    def __call__(self, interval: Interval, state: object) -> float:
        del interval, state
        return 0.0

    def snapshot_state(self, state: object) -> object:
        return state


class BadAccelerator:
    name = "bad"


class UserAccelerator:
    name = "user"
    strict = False

    @staticmethod
    def compile(function=None, /, **options):
        del options
        if function is None:
            return lambda target: target
        return function

    @staticmethod
    def compile_examples(function, *signatures):
        del signatures
        return function


def test_auditor_reports_ready_configuration() -> None:
    auditor = Auditor(
        state={"x": 1.0},
        derivative=derivative,
        translation=DummyTranslation(),
        allocator=DummyAllocator(),
        interval=Interval(0.0, 0.1, 1.0),
        scheme=DummyScheme(),
        tolerance=Tolerance(),
        accelerator=AcceleratorNone(),
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
    assert report.index("Translation") < report.index("Allocator")
    assert report.index("Allocator") < report.index("Accelerator")
    assert report.index("Accelerator") < report.index("Scheme")
    assert "Overall: ready." in report


def test_auditor_reports_missing_requirements() -> None:
    auditor = Auditor(
        state={"x": 1.0},
        derivative=derivative,
        translation=DummyTranslation(),
        allocator=BadAllocator(),
        interval=Interval(0.0, 0.1, 1.0),
    )

    assert not auditor.ok
    report = str(auditor)
    assert "no" in report
    assert "copy_state" in report


def test_require_scheme_inputs_raises_helpful_error() -> None:
    try:
        Auditor.require_scheme_inputs(derivative, BadAllocator(), DummyTranslation())
    except AuditError as exc:
        message = str(exc)
        assert "Allocator provides copy_state" in message
        assert "Overall: incomplete." in message
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("Expected the audit to reject a bad allocator.")


def test_require_linear_residual_rejects_missing_linearize() -> None:
    class ResidualOnly:
        def __call__(self, block, out) -> None:
            del block, out

    try:
        Auditor.require_linear_residual(ResidualOnly())
    except AuditError as exc:
        message = str(exc)
        assert "Residual provides linearize(block, out)" in message
        assert "Overall: incomplete." in message
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("Expected the audit to reject a residual without linearize().")


def test_auditor_reports_ready_imex_derivative() -> None:
    imex = Derivative.split(implicit=derivative, explicit=derivative)

    auditor = Auditor(
        state={"x": 1.0},
        imex_derivative=imex,
        translation=DummyTranslation(),
        allocator=DummyAllocator(),
        interval=Interval(0.0, 0.1, 1.0),
    )

    assert auditor.ok
    report = str(auditor)
    assert "DerivativeSplit provides implicit(interval, state, translation)" in report
    assert "DerivativeSplit provides explicit(interval, state, translation)" in report


def test_derivative_style_declares_imex_split() -> None:
    styled = DerivativeStyle.split(implicit=derivative, explicit=derivative)
    direct = Derivative.split(implicit=derivative, explicit=derivative)

    assert isinstance(styled, DerivativeSplit)
    assert styled.implicit is derivative
    assert styled.explicit is derivative
    assert isinstance(direct, DerivativeSplit)


def test_require_imex_scheme_inputs_rejects_missing_explicit_part() -> None:
    class BadImEx:
        implicit = staticmethod(derivative)
        explicit = None

    try:
        Auditor.require_imex_scheme_inputs(BadImEx(), DummyAllocator(), DummyTranslation())
    except AuditError as exc:
        message = str(exc)
        assert "DerivativeSplit provides explicit" in message
        assert "Overall: incomplete." in message
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("Expected the audit to reject an IMEX split without an explicit part.")


def test_auditor_reports_missing_accelerator_requirements() -> None:
    auditor = Auditor(accelerator=BadAccelerator(), exercise=False)

    assert not auditor.ok
    report = str(auditor)
    assert "Accelerator provides compile(function=None, **options)" in report
    assert "Accelerator provides compile_examples(function, *examples)" in report


def test_auditor_reports_ready_built_in_accelerator() -> None:
    auditor = Auditor(accelerator=AcceleratorNone(), exercise=False)

    assert auditor.ok
    assert "Accelerator provides compile(function=None, **options)" in str(auditor)


def test_auditor_reports_ready_user_defined_accelerator() -> None:
    auditor = Auditor(accelerator=UserAccelerator(), exercise=False)

    assert auditor.ok
    assert "Overall: ready." in str(auditor)


def test_accelerator_compile_accepts_decorator_form() -> None:
    accelerator = AcceleratorNone()

    @accelerator.compile
    def worker(value: float) -> float:
        return 2.0 * value

    assert worker(3.0) == 6.0


def test_accelerator_compile_accepts_configured_decorator_form() -> None:
    accelerator = AcceleratorNone()

    @accelerator.compile(label="audit")
    def worker(value: float) -> float:
        return 3.0 * value

    assert worker(4.0) == 12.0


