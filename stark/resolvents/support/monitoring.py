from __future__ import annotations

from typing import Protocol


class MonitorResolventLike(Protocol):
    """Minimal resolvent-side recording surface for monitorable resolvents."""

    def record_solve(
        self,
        resolvent: str,
        alpha: float,
        block_size: int,
        iteration_count: int,
        error: float,
        scale: float,
        converged: bool,
    ) -> None: ...


__all__ = [
    "MonitorResolventLike",
]
