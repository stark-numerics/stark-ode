from __future__ import annotations

"""Small protocols shared by STARK's audit workers.

The audit layer is deliberately lightweight: each domain-specific audit object
receives an `AuditRecorder`, records the checks it can perform, and optionally
exercises the supplied object on small samples.
"""

from typing import Any, Protocol


class AuditRecorder(Protocol):
    """Collect pass/fail checks and caught exceptions during an audit."""

    def check(self, passed: bool, summary: str, detail: str | None = None) -> None:
        ...

    def record_exception(self, summary: str, exc: Exception, detail: str | None = None) -> None:
        ...


class AuditWorker(Protocol):
    """Audit one target object and record the outcome."""

    def __call__(self, recorder: AuditRecorder, target: Any, *, exercise: bool = True) -> Any:
        ...


__all__ = ["AuditRecorder", "AuditWorker"]


