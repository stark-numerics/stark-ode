from __future__ import annotations

from typing import Any, Protocol


class AuditRecorder(Protocol):
    def check(self, passed: bool, summary: str, detail: str | None = None) -> None:
        ...

    def record_exception(self, summary: str, exc: Exception, detail: str | None = None) -> None:
        ...


class AuditWorker(Protocol):
    def __call__(self, recorder: AuditRecorder, target: Any, *, exercise: bool = True) -> Any:
        ...


__all__ = ["AuditRecorder", "AuditWorker"]


