"""Contract auditing for user-supplied STARK objects."""

from __future__ import annotations

from typing import Any

from stark.core.contracts.accelerator import AcceleratorAudit
from stark.core.contracts.derivative_imex import DerivativeIMEXAudit
from stark.core.contracts.derivative import DerivativeAudit
from stark.core.contracts.interval import IntervalAudit
from stark.core.contracts.linearizer import LinearizerAudit
from stark.core.contracts.stepper import IntegratorStepperAudit
from stark.core.contracts.residual import ResidualAudit
from stark.core.contracts.scheme import SchemeAudit
from stark.core.contracts.translation import TranslationAudit
from stark.core.contracts.allocator import AllocatorAudit
from stark.core.interval import Interval
from stark.core.tolerance import ToleranceAudit


class AuditError(TypeError):
    """Raised when a STARK contract audit fails."""


class Auditor:
    """
    Check whether user-supplied objects satisfy the STARK contracts.

    Instantiate `Auditor(...)` to produce a checklist-style report before
    starting an integration. Internal STARK constructors use the `require_*`
    class methods to fail fast with helpful errors.
    """

    __slots__ = ("checks",)

    translation_audit = TranslationAudit()
    derivative_audit = DerivativeAudit()
    derivative_imex_audit = DerivativeIMEXAudit()
    allocator_audit = AllocatorAudit()
    linearizer_audit = LinearizerAudit()
    interval_audit = IntervalAudit()
    scheme_audit = SchemeAudit()
    stepper_audit = IntegratorStepperAudit()
    residual_audit = ResidualAudit()
    acceleration_audit = AcceleratorAudit()
    tolerance_audit = ToleranceAudit()

    def __init__(
        self,
        *,
        state: Any | None = None,
        derivative: Any | None = None,
        imex_derivative: Any | None = None,
        translation: Any | None = None,
        allocator: Any | None = None,
        interval: Any | None = None,
        stepper: Any | None = None,
        scheme: Any | None = None,
        tolerance: Any | None = None,
        accelerator: Any | None = None,
        residual: Any | None = None,
        linear_residual: bool = False,
        snapshots: bool = True,
        exercise: bool = True,
    ) -> None:
        self.checks: list[tuple[bool, str, str | None]] = []

        sample_state = None
        second_state = None
        sample_translation = None

        if derivative is not None:
            self.derivative_audit(self, derivative)

        if imex_derivative is not None:
            self.derivative_imex_audit(self, imex_derivative)

        if allocator is not None:
            sample_state, second_state, sample_translation = self.allocator_audit(self, allocator, exercise=exercise)

        if translation is not None:
            self.translation_audit(
                self,
                translation,
                exercise=exercise,
                state=state,
                result_state=sample_state,
                sample_translation=sample_translation if sample_translation is not None else translation,
            )

        if tolerance is not None:
            self.tolerance_audit(self, tolerance)

        if accelerator is not None:
            self.acceleration_audit(self, accelerator, exercise=exercise)

        if residual is not None:
            self.residual_audit(self, residual, linear=linear_residual)

        if scheme is not None:
            self.scheme_audit(self, scheme)

        if stepper is not None:
            self.stepper_audit(self, stepper, snapshots=snapshots)

        if interval is not None:
            self.interval_audit(self, interval, exercise=exercise)

        if exercise and allocator is not None and derivative is not None and state is not None and interval is not None:
            candidate_translation = sample_translation if sample_translation is not None else translation
            if candidate_translation is not None:
                self.derivative_audit.exercise(self, derivative, interval, state, candidate_translation)

        if exercise and allocator is not None and imex_derivative is not None and state is not None and interval is not None:
            candidate_translation = sample_translation if sample_translation is not None else translation
            if candidate_translation is not None:
                self.derivative_imex_audit.exercise(self, imex_derivative, interval, state, candidate_translation)

        if exercise and allocator is not None and state is not None and sample_state is not None:
            self.allocator_audit.exercise_copy_state(self, allocator, state, sample_state)

        if exercise and sample_state is not None and second_state is not None and allocator is not None:
            self.allocator_audit.exercise_allocator_copy(self, allocator, second_state, sample_state)

    @property
    def ok(self) -> bool:
        return all(passed for passed, _summary, _detail in self.checks)

    def __repr__(self) -> str:
        passed = sum(1 for ok, _summary, _detail in self.checks if ok)
        total = len(self.checks)
        status = "ready" if self.ok else "incomplete"
        return f"Auditor(status={status!r}, passed={passed}, total={total})"

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise AuditError(str(self))

    def check(self, passed: bool, summary: str, detail: str | None = None) -> None:
        self.checks.append((passed, summary, detail))

    def record_exception(self, summary: str, exc: Exception, detail: str | None = None) -> None:
        message = detail if detail is not None else f"{type(exc).__name__}: {exc}"
        self.check(False, summary, message)

    @classmethod
    def require_scheme_inputs(cls, derivative: Any, allocator: Any, translation: Any) -> None:
        cls(derivative=derivative, allocator=allocator, translation=translation, exercise=False).raise_if_invalid()

    @classmethod
    def require_imex_scheme_inputs(cls, imex_derivative: Any, allocator: Any, translation: Any) -> None:
        cls(imex_derivative=imex_derivative, allocator=allocator, translation=translation, exercise=False).raise_if_invalid()

    @classmethod
    def require_stepper_inputs(cls, scheme: Any) -> None:
        auditor = cls(
            scheme=scheme,
            snapshots=True,
            exercise=False,
        )
        auditor.raise_if_invalid()

    @classmethod
    def require_integration_inputs(cls, stepper: Any, interval: Any, state: Any, *, snapshots: bool) -> None:
        del state
        cls(stepper=stepper, interval=interval, snapshots=snapshots, exercise=False).raise_if_invalid()

    @classmethod
    def require_linear_residual(cls, residual: Any) -> None:
        cls(residual=residual, linear_residual=True, exercise=False).raise_if_invalid()

    @classmethod
    def require_linearizer_inputs(cls, linearizer: Any, allocator: Any, translation: Any) -> None:
        auditor = cls(allocator=allocator, translation=translation, exercise=False)
        auditor.linearizer_audit(auditor, linearizer)
        linear_combine = getattr(translation, "linear_combine", None)
        auditor.check(
            isinstance(linear_combine, (list, tuple))
            and len(linear_combine) >= 2
            and callable(linear_combine[0])
            and callable(linear_combine[1]),
            "Translation provides in-place scale and combine2 kernels for strict operator algebra.",
            "Add translation.linear_combine = [scale, combine2, ...] with output-last in-place kernels before using a linearizer.",
        )
        if not auditor.ok:
            auditor.raise_if_invalid()

        try:
            sample_state = allocator.allocate_state()
        except Exception as exc:
            auditor.record_exception("Allocator.allocate_state() succeeds for linearizer audit.", exc)
        else:
            auditor.check(True, "Allocator.allocate_state() succeeds for linearizer audit.")
            auditor.linearizer_audit.exercise(auditor, linearizer, Interval(0.0, 1.0, 1.0), sample_state, translation)
        auditor.raise_if_invalid()

    def __str__(self) -> str:
        headers = ("Object", "Required behavior", "Present")
        checks = sorted(
            enumerate(self.checks),
            key=lambda indexed: (
                self._object_order(self._object_name(indexed[1][1])),
                indexed[0],
            ),
        )
        rows = [self._display_row(summary, passed) for _index, (passed, summary, _detail) in checks]
        widths = [max(len(row[index]) for row in (headers, *rows)) for index in range(len(headers))]

        def render(row: tuple[str, str, str]) -> str:
            return " | ".join(value.ljust(width) for value, width in zip(row, widths, strict=True))

        lines = [
            "STARK audit checklist",
            render(headers),
            "-+-".join("-" * width for width in widths),
        ]
        for _index, (passed, summary, detail) in checks:
            lines.append(render(self._display_row(summary, passed)))
            if detail and not passed:
                lines.append(f"  detail: {detail}")

        status = "ready" if self.ok else "incomplete"
        lines.append(f"Overall: {status}.")
        return "\n".join(lines)

    @classmethod
    def _display_row(cls, summary: str, passed: bool) -> tuple[str, str, str]:
        return cls._object_name(summary), summary.rstrip("."), "yes" if passed else "no"

    @staticmethod
    def _object_name(summary: str) -> str:
        if summary.startswith("Derivative"):
            return "Derivative"
        if summary.startswith("DerivativeIMEX"):
            return "Derivative"
        if summary.startswith("Translation"):
            return "Translation"
        if summary.startswith("Allocator"):
            return "Allocator"
        if summary.startswith("Tolerance"):
            return "Tolerance"
        if summary.startswith("Accelerator"):
            return "Accelerator"
        if summary.startswith("Residual"):
            return "Residual"
        if summary.startswith("Interval"):
            return "Interval"
        if summary.startswith("IntegratorStepper"):
            return "IntegratorStepper"
        if summary.startswith("Scheme"):
            return "Scheme"
        return "Interface"

    @staticmethod
    def _object_order(object_name: str) -> int:
        order = {
            "Interval": 0,
            "Derivative": 1,
            "Translation": 2,
            "Allocator": 3,
            "Accelerator": 4,
            "Scheme": 5,
            "Tolerance": 6,
            "Residual": 7,
            "IntegratorStepper": 8,
            "Interface": 9,
        }
        return order.get(object_name, order["Interface"])


__all__ = ["AuditError", "Auditor"]


