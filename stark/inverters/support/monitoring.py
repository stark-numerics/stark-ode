from __future__ import annotations

from typing import Protocol


class MonitorInverterLike(Protocol):
    """Minimal inverter-side recording surface for monitorable inverters."""

    def record_solve(
        self,
        inverter: str,
        converged: bool,
        iteration_count: int | None,
        initial_residual: float | None,
        final_residual: float | None,
        failure_reason: str | None,
    ) -> None: ...


__all__ = [
    "MonitorInverterLike",
]
