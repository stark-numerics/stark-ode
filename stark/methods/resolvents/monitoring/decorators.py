from __future__ import annotations

from stark.methods.resolvents.monitoring.monitor import MonitorResolventLike

def with_resolvent_monitoring(cls):
    """Install the standard resolvent monitor boundary."""

    def assign_monitor(self, monitor: MonitorResolventLike) -> None:
        self._monitor = monitor

    def unassign_monitor(self) -> None:
        self._monitor = None

    def record_solve(
        self,
        block_size: int,
        iteration_count: int,
        error: float,
        scale: float,
        converged: bool,
    ) -> None:
        monitor = self._monitor
        if monitor is None:
            return

        descriptor = getattr(type(self), "descriptor", None)
        resolvent_name = getattr(descriptor, "short_name", type(self).__name__)
        monitor.record_solve(
            resolvent_name,
            self.alpha,
            block_size,
            iteration_count,
            error,
            scale,
            converged,
        )

    cls.assign_monitor = assign_monitor
    cls.unassign_monitor = unassign_monitor
    cls.record_solve = record_solve
    return cls


__all__ = ["with_resolvent_monitoring"]
