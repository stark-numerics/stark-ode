from dataclasses import dataclass

from stark.accelerators import Accelerator
from stark.contracts import AccelerationRequest, AccelerationRole
from stark import Executor, ImExDerivative
from stark.auditor import AuditError, Auditor
from stark.execution.tolerance import Tolerance
from stark.interval import Interval


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
    def scale(out: "DummyTranslation", scalar: float, translation: "DummyTranslation") -> "DummyTranslation":
        out.value = scalar * translation.value
        return out

    @staticmethod
    def combine2(
        out: "DummyTranslation",
        a0: float,
        x0: "DummyTranslation",
        a1: float,
        x1: "DummyTranslation",
    ) -> "DummyTranslation":
        out.value = a0 * x0.value + a1 * x1.value
        return out

    linear_combine = [scale, combine2]


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


def derivative(interval: Interval, state: dict[str, float], out: DummyTranslation) -> None:
    del interval
    out.value = state["x"]


class DummyScheme:
    def __call__(self, interval: Interval, state: object, executor: Executor) -> float:
        del interval, state, executor
        return 0.0

    def snapshot_state(self, state: object) -> object:
        return state

    def set_apply_delta_safety(self, enabled: bool) -> None:
        del enabled


class BadAccelerator:
    name = "bad"
    available = True


class UserAccelerator:
    name = "user"
    available = True
    strict = False

    @staticmethod
    def decorate(function=None, /, **kwargs):
        del kwargs
        if function is None:
            return lambda target: target
        return function

    @staticmethod
    def compile_examples(function, *signatures):
        del signatures
        return function

    @staticmethod
    def resolve(target, request):
        del request
        return target

    def resolve_derivative(self, derivative):
        return self.resolve(derivative, AccelerationRequest(AccelerationRole.DERIVATIVE))

    def resolve_linearizer(self, linearizer):
        return self.resolve(linearizer, AccelerationRequest(AccelerationRole.LINEARIZER))

    def resolve_support(self, worker, *, label=None, **values):
        return self.resolve(worker, AccelerationRequest(AccelerationRole.SUPPORT, label=label, values=values))


class AcceleratedDerivative:
    def __call__(self, interval: Interval, state: dict[str, float], out: DummyTranslation) -> None:
        del interval
        out.value = 2.0 * state["x"]


class DerivativeWithAcceleration:
    def __call__(self, interval: Interval, state: dict[str, float], out: DummyTranslation) -> None:
        del interval
        out.value = state["x"]

    def accelerated(self, accelerator: Accelerator, request: AccelerationRequest):
        if request.role is AccelerationRole.DERIVATIVE and accelerator.name == "none":
            return AcceleratedDerivative()
        return self


class AcceleratedLinearizer:
    def __call__(self, interval: Interval, out: object, state: dict[str, float]) -> None:
        del interval, state

        def apply(result: DummyTranslation, translation: DummyTranslation) -> None:
            result.value = 3.0 * translation.value

        setattr(out, "apply", apply)


class LinearizerWithAcceleration:
    def __call__(self, interval: Interval, out: object, state: dict[str, float]) -> None:
        del interval, state

        def apply(result: DummyTranslation, translation: DummyTranslation) -> None:
            result.value = translation.value

        setattr(out, "apply", apply)

    def accelerated(self, accelerator: Accelerator, request: AccelerationRequest):
        if request.role is AccelerationRole.LINEARIZER and accelerator.name == "none":
            return AcceleratedLinearizer()
        return self


def test_auditor_reports_ready_configuration() -> None:
    auditor = Auditor(
        state={"x": 1.0},
        derivative=derivative,
        translation=DummyTranslation(),
        workbench=DummyWorkbench(),
        interval=Interval(0.0, 0.1, 1.0),
        scheme=DummyScheme(),
        tolerance=Tolerance(),
        accelerator=Accelerator.none(),
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
    assert report.index("Workbench") < report.index("Accelerator")
    assert report.index("Accelerator") < report.index("Scheme")
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


def test_auditor_reports_ready_imex_derivative() -> None:
    imex = ImExDerivative(implicit=derivative, explicit=derivative)

    auditor = Auditor(
        state={"x": 1.0},
        imex_derivative=imex,
        translation=DummyTranslation(),
        workbench=DummyWorkbench(),
        interval=Interval(0.0, 0.1, 1.0),
    )

    assert auditor.ok
    report = str(auditor)
    assert "ImExDerivative provides implicit(interval, state, translation)" in report
    assert "ImExDerivative provides explicit(interval, state, translation)" in report


def test_require_imex_scheme_inputs_rejects_missing_explicit_part() -> None:
    class BadImEx:
        implicit = staticmethod(derivative)
        explicit = None

    try:
        Auditor.require_imex_scheme_inputs(BadImEx(), DummyWorkbench(), DummyTranslation())
    except AuditError as exc:
        message = str(exc)
        assert "ImExDerivative provides explicit" in message
        assert "Overall: incomplete." in message
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("Expected the audit to reject an IMEX split without an explicit part.")


def test_auditor_reports_missing_accelerator_requirements() -> None:
    auditor = Auditor(accelerator=BadAccelerator(), exercise=False)

    assert not auditor.ok
    report = str(auditor)
    assert "Accelerator provides decorate(function=None, **kwargs)" in report
    assert "Accelerator provides compile_examples(function, *signatures)" in report
    assert "Accelerator provides resolve(target, request)" in report


def test_auditor_reports_ready_built_in_accelerator() -> None:
    auditor = Auditor(accelerator=Accelerator.none(), exercise=False)

    assert auditor.ok
    assert "Accelerator provides resolve(target, request)" in str(auditor)


def test_auditor_reports_ready_user_defined_accelerator() -> None:
    auditor = Auditor(accelerator=UserAccelerator(), exercise=False)

    assert auditor.ok
    assert "Overall: ready." in str(auditor)


def test_accelerator_can_resolve_a_typed_derivative_variant() -> None:
    accelerator = Accelerator.none()
    derivative = accelerator.resolve_derivative(DerivativeWithAcceleration())
    out = DummyTranslation()

    derivative(Interval(0.0, 0.1, 1.0), {"x": 3.0}, out)

    assert out.value == 6.0


def test_accelerated_linearizer_resolution_still_satisfies_linearizer_contract() -> None:
    accelerator = Accelerator.none()
    linearizer = accelerator.resolve_linearizer(LinearizerWithAcceleration())
    result = DummyTranslation()

    Auditor.require_linearizer_inputs(linearizer, DummyWorkbench(), DummyTranslation())
    linearizer(Interval(0.0, 0.1, 1.0), operator := type("OperatorProbe", (), {})(), {"x": 2.0})
    operator.apply(result, DummyTranslation(4.0))

    assert result.value == 12.0


def test_executor_rejects_non_conformant_accelerator_early_and_clearly() -> None:
    try:
        Executor(accelerator=BadAccelerator())
    except AuditError as exc:
        message = str(exc)
        assert "Accelerator provides decorate(function=None, **kwargs)" in message
        assert "Overall: incomplete." in message
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("Expected Executor(...) to reject a non-conformant accelerator.")












