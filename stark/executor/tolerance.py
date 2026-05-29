from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stark.contracts.audit_support import AuditRecorder


@dataclass(frozen=True, slots=True)
class ExecutorTolerance:
    """
    Executor-owned tolerance object for normalized error control.

    Any object providing `bound(scale)`, `ratio(error, scale)`, and
    `accepts(error, scale)` can be used in its place. Schemes see this through
    their local `SchemeTolerance` protocol.
    """

    atol: float = 1.0e-6
    rtol: float = 1.0e-6

    def __repr__(self) -> str:
        return f"{type(self).__name__}(atol={self.atol!r}, rtol={self.rtol!r})"

    def __str__(self) -> str:
        return f"atol={self.atol:g}, rtol={self.rtol:g}"

    def bound(self, scale: float) -> float:
        return self.atol + self.rtol * scale

    def ratio(self, error: float, scale: float) -> float:
        return error / self.bound(scale)

    def accepts(self, error: float, scale: float) -> bool:
        return self.ratio(error, scale) <= 1.0


class ExecutorToleranceAudit:
    def __call__(self, recorder: AuditRecorder, tolerance: Any) -> None:
        recorder.check(callable(getattr(tolerance, "bound", None)), "ExecutorTolerance provides bound(scale).", "Pass stark.ExecutorTolerance(...) or add bound(scale).")
        recorder.check(callable(getattr(tolerance, "ratio", None)), "ExecutorTolerance provides ratio(error, scale).", "Add ratio(error, scale) for adaptive schemes.")
        recorder.check(
            callable(getattr(tolerance, "accepts", None)),
            "ExecutorTolerance provides accepts(error, scale).",
            "Add accepts(error, scale) if you want compatibility with STARK's ExecutorTolerance interface.",
        )


__all__ = ["ExecutorTolerance", "ExecutorToleranceAudit"]









