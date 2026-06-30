from __future__ import annotations

from typing import Protocol


class MonitorInverterLike(Protocol):
    """Minimal inverter-side recording surface."""

    def record_solve(
        self,
        inverter: str,
        converged: bool,
        iteration_count: int | None,
        initial_residual: float | None,
        final_residual: float | None,
        failure_reason: str | None,
    ) -> None: ...


class InverterRecordSolve(Protocol):
    """Decorator-installed solve recorder on monitored inverters."""

    def __call__(
        self,
        *,
        converged: bool,
        iteration_count: int | None,
        initial_residual: float | None,
        final_residual: float | None,
        failure_reason: str | None = None,
    ) -> None: ...


def with_inverter_monitoring(cls):
    """Install optional solve recording for an inverter class."""

    def record_solve(
        self,
        converged: bool,
        iteration_count: int | None,
        initial_residual: float | None,
        final_residual: float | None,
        failure_reason: str | None = None,
    ) -> None:
        monitor = getattr(self, "monitor", None)
        if monitor is None:
            return

        descriptor = getattr(type(self), "descriptor", None)
        inverter_name = getattr(descriptor, "short_name", type(self).__name__)
        monitor.record_solve(
            inverter_name,
            converged,
            iteration_count,
            initial_residual,
            final_residual,
            failure_reason,
        )

    cls.record_solve = record_solve
    return cls


__all__ = ["InverterRecordSolve", "MonitorInverterLike", "with_inverter_monitoring"]
