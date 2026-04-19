from __future__ import annotations

from typing import Any

from stark.contracts.acceleration import AcceleratorAudit
from stark.contracts.integration import IntegrationAudit
from stark.contracts.intervals import IntervalAudit
from stark.contracts.problems import ProblemAudit
from stark.contracts.solvers import SolverAudit
from stark.contracts.translations import TranslationAudit
from stark.execution.safety import SafetyAudit
from stark.execution.tolerance import ToleranceAudit
from stark.interval import Interval


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
    problem_audit = ProblemAudit()
    interval_audit = IntervalAudit()
    integration_audit = IntegrationAudit()
    solver_audit = SolverAudit()
    acceleration_audit = AcceleratorAudit()
    tolerance_audit = ToleranceAudit()
    safety_audit = SafetyAudit()

    def __init__(
        self,
        *,
        state: Any | None = None,
        derivative: Any | None = None,
        imex_derivative: Any | None = None,
        translation: Any | None = None,
        workbench: Any | None = None,
        interval: Any | None = None,
        marcher: Any | None = None,
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
            self.problem_audit.derivative(self, derivative)

        if imex_derivative is not None:
            self.problem_audit.imex_derivative(self, imex_derivative)

        if workbench is not None:
            sample_state, second_state, sample_translation = self.problem_audit.workbench(self, workbench, exercise=exercise)

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
            self.solver_audit.residual(self, residual, linear=linear_residual)

        if scheme is not None:
            self.integration_audit.scheme(self, scheme)

        if marcher is not None:
            self.integration_audit.marcher(self, marcher, snapshots=snapshots)

        if interval is not None:
            self.interval_audit(self, interval, exercise=exercise)

        if exercise and workbench is not None and derivative is not None and state is not None and interval is not None:
            candidate_translation = sample_translation if sample_translation is not None else translation
            if candidate_translation is not None:
                self.problem_audit.exercise_derivative(self, derivative, interval, state, candidate_translation)

        if exercise and workbench is not None and imex_derivative is not None and state is not None and interval is not None:
            candidate_translation = sample_translation if sample_translation is not None else translation
            if candidate_translation is not None:
                self.problem_audit.exercise_imex_derivative(self, imex_derivative, interval, state, candidate_translation)

        if exercise and workbench is not None and state is not None and sample_state is not None:
            self.problem_audit.exercise_copy_state(self, workbench, sample_state, state)

        if exercise and sample_state is not None and second_state is not None and workbench is not None:
            self.problem_audit.exercise_workbench_copy(self, workbench, sample_state, second_state)

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
    def require_scheme_inputs(cls, derivative: Any, workbench: Any, translation: Any) -> None:
        cls(derivative=derivative, workbench=workbench, translation=translation, exercise=False).raise_if_invalid()

    @classmethod
    def require_imex_scheme_inputs(cls, imex_derivative: Any, workbench: Any, translation: Any) -> None:
        cls(imex_derivative=imex_derivative, workbench=workbench, translation=translation, exercise=False).raise_if_invalid()

    @classmethod
    def require_marcher_inputs(cls, scheme: Any, tolerance: Any, safety: Any, accelerator: Any) -> None:
        auditor = cls(
            scheme=scheme,
            tolerance=tolerance,
            accelerator=accelerator,
            snapshots=True,
            exercise=False,
        )
        auditor.safety_audit(auditor, safety)
        auditor.raise_if_invalid()

    @classmethod
    def require_integration_inputs(cls, marcher: Any, interval: Any, state: Any, *, snapshots: bool) -> None:
        del state
        cls(marcher=marcher, interval=interval, snapshots=snapshots, exercise=False).raise_if_invalid()

    @classmethod
    def require_linear_residual(cls, residual: Any) -> None:
        cls(residual=residual, linear_residual=True, exercise=False).raise_if_invalid()

    @classmethod
    def require_linearizer_inputs(cls, linearizer: Any, workbench: Any, translation: Any) -> None:
        auditor = cls(workbench=workbench, translation=translation, exercise=False)
        auditor.problem_audit.linearizer(auditor, linearizer)
        linear_combine = getattr(translation, "linear_combine", None)
        auditor.check(
            isinstance(linear_combine, (list, tuple))
            and len(linear_combine) >= 2
            and callable(linear_combine[0])
            and callable(linear_combine[1]),
            "Translation provides in-place scale and combine2 kernels for strict operator algebra.",
            "Add translation.linear_combine = [scale, combine2, ...] with in-place kernels before using a linearizer.",
        )
        if not auditor.ok:
            auditor.raise_if_invalid()

        try:
            sample_state = workbench.allocate_state()
        except Exception as exc:
            auditor.record_exception("Workbench.allocate_state() succeeds for linearizer audit.", exc)
        else:
            auditor.check(True, "Workbench.allocate_state() succeeds for linearizer audit.")
            auditor.problem_audit.exercise_linearizer(auditor, linearizer, Interval(0.0, 1.0, 1.0), sample_state, translation)
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
        if summary.startswith("ImExDerivative"):
            return "Derivative"
        if summary.startswith("Translation"):
            return "Translation"
        if summary.startswith("Workbench"):
            return "Workbench"
        if summary.startswith("Tolerance"):
            return "Tolerance"
        if summary.startswith("Accelerator"):
            return "Accelerator"
        if summary.startswith("Residual"):
            return "Residual"
        if summary.startswith("Interval"):
            return "Interval"
        if summary.startswith("Marcher"):
            return "Marcher"
        if summary.startswith("Scheme"):
            return "Scheme"
        if summary.startswith("safety"):
            return "Marcher"
        return "Interface"

    @staticmethod
    def _object_order(object_name: str) -> int:
        order = {
            "Interval": 0,
            "Derivative": 1,
            "Translation": 2,
            "Workbench": 3,
            "Accelerator": 4,
            "Scheme": 5,
            "Tolerance": 6,
            "Residual": 7,
            "Marcher": 8,
            "Interface": 9,
        }
        return order.get(object_name, order["Interface"])


__all__ = ["AuditError", "Auditor"]


