from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stark.contracts.audit_support import AuditRecorder


@dataclass(slots=True)
class Tolerance:
    """
    General STARK tolerance object for normalized error control.

    Any object providing `bound(scale)`, `ratio(error, scale)`, and
    `accepts(error, scale)` can be used in its place, but this class is the
    common duck-typed default for scheme, resolver, and inverter tolerances.
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


@dataclass(slots=True)
class SchemeTolerance(Tolerance):
    """Scheme-facing tolerance object for adaptive step acceptance."""


class ToleranceAudit:
    def __call__(self, recorder: AuditRecorder, tolerance: Any) -> None:
        recorder.check(callable(getattr(tolerance, "bound", None)), "Tolerance provides bound(scale).", "Pass stark.Tolerance(...) or add bound(scale).")
        recorder.check(callable(getattr(tolerance, "ratio", None)), "Tolerance provides ratio(error, scale).", "Add ratio(error, scale) for adaptive schemes.")
        recorder.check(
            callable(getattr(tolerance, "accepts", None)),
            "Tolerance provides accepts(error, scale).",
            "Add accepts(error, scale) if you want compatibility with STARK's tolerance interface.",
        )


__all__ = ["SchemeTolerance", "Tolerance", "ToleranceAudit"]









