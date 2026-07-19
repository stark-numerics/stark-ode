from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stark.core.contracts.shared.contract_audit import AuditRecorder


@dataclass(frozen=True, slots=True)
class Tolerance:
    """Shared normalized error-control tolerance."""

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


class ToleranceAudit:
    def __call__(self, recorder: AuditRecorder, tolerance: Any) -> None:
        recorder.check(
            callable(getattr(tolerance, "bound", None)),
            "Tolerance provides bound(scale).",
            "Pass stark.Tolerance(...) or add bound(scale).",
        )
        recorder.check(
            callable(getattr(tolerance, "ratio", None)),
            "Tolerance provides ratio(error, scale).",
            "Add ratio(error, scale) for adaptive schemes.",
        )
        recorder.check(
            callable(getattr(tolerance, "accepts", None)),
            "Tolerance provides accepts(error, scale).",
            "Add accepts(error, scale) for STARK tolerance compatibility.",
        )


__all__ = ["Tolerance", "ToleranceAudit"]
