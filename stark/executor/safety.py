from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stark.contracts.audit_support import AuditRecorder


@dataclass(frozen=True, slots=True)
class ExecutorSafety:
    """Executor-owned safety policy for STARK workers."""

    progress: bool = True
    block_sizes: bool = True
    apply_delta: bool = True

    def __repr__(self) -> str:
        return (
            "ExecutorSafety("
            f"progress={self.progress!r}, "
            f"block_sizes={self.block_sizes!r}, "
            f"apply_delta={self.apply_delta!r})"
        )

    def __str__(self) -> str:
        return (
            f"progress={self.progress!r}, "
            f"block_sizes={self.block_sizes!r}, "
            f"apply_delta={self.apply_delta!r}"
        )

    @classmethod
    def fast(cls) -> "ExecutorSafety":
        return cls(progress=False, block_sizes=False, apply_delta=False)


class ExecutorSafetyAudit:
    def __call__(self, recorder: AuditRecorder, safety: Any) -> None:
        recorder.check(
            isinstance(safety, ExecutorSafety),
            "safety is an ExecutorSafety policy.",
            "Pass safety=ExecutorSafety(...) or use the default ExecutorSafety().",
        )


__all__ = ["ExecutorSafety", "ExecutorSafetyAudit"]









