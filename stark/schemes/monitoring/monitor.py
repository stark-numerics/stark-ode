from __future__ import annotations

from typing import Protocol


class MonitorSchemeLike(Protocol):
    """Minimal scheme-side recording surface for monitorable schemes."""

    def record_fixed_step(
        self,
        scheme: str,
        t_start: float,
        accepted_dt: float,
    ) -> None: ...

    def record_adaptive_step(
        self,
        scheme: str,
        t_start: float,
        proposed_dt: float,
        accepted_dt: float,
        next_dt: float,
        error_ratio: float,
        rejection_count: int,
    ) -> None: ...


__all__ = ["MonitorSchemeLike"]
