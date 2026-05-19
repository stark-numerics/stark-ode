from __future__ import annotations

from typing import Protocol

from stark.contracts import IntervalLike, State


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


def refresh_fixed_step_call(scheme) -> None:
    """Select the fixed-step pure or monitored call path for a scheme."""

    scheme.redirect_call = (
        scheme.call_monitored if scheme._monitor is not None else scheme.call_pure
    )


def with_fixed_step_monitoring(cls):
    """Install the standard fixed-step monitor boundary on a scheme class."""

    def assign_monitor(self, monitor: MonitorSchemeLike) -> None:
        self._monitor = monitor
        refresh_fixed_step_call(self)

    def unassign_monitor(self) -> None:
        self._monitor = None
        refresh_fixed_step_call(self)

    def call_monitored(
        self,
        interval: IntervalLike,
        state: State,
        executor,
    ) -> float:
        t_start = interval.present
        accepted_dt = self.call_pure(interval, state, executor)
        monitor = self._monitor
        if monitor is not None and accepted_dt > 0.0:
            monitor.record_fixed_step(self.short_name, t_start, accepted_dt)
        return accepted_dt

    cls.assign_monitor = assign_monitor
    cls.unassign_monitor = unassign_monitor
    cls.call_monitored = call_monitored
    return cls


__all__ = [
    "MonitorSchemeLike",
    "refresh_fixed_step_call",
    "with_fixed_step_monitoring",
]
