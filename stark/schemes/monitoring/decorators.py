from __future__ import annotations

from stark.contracts import IntervalLike, State


def with_fixed_step_monitoring(cls):
    """Install the standard fixed-step monitor wrapper on a scheme class."""

    def call_monitored(
        self,
        interval: IntervalLike,
        state: State,
        executor,
    ) -> float:
        t_start = interval.present
        accepted_dt = self.call_body(interval, state, executor)
        monitor = self.monitor
        if monitor is not None and accepted_dt > 0.0:
            monitor.record_fixed_step(self.short_name, t_start, accepted_dt)
        return accepted_dt

    cls.call_monitored = call_monitored
    return cls


def with_adaptive_step_monitoring(cls):
    """Install the standard adaptive-step monitor wrapper on a scheme class."""

    def call_monitored(
        self,
        interval: IntervalLike,
        state: State,
        executor,
    ) -> float:
        accepted_dt = self.call_body(interval, state, executor)
        report = self.step_control.report()
        monitor = self.monitor

        if monitor is not None:
            monitor.record_adaptive_step(
                self.short_name,
                report.t_start,
                report.proposed_dt,
                report.accepted_dt,
                report.next_dt,
                report.error_ratio,
                report.rejection_count,
            )

        return accepted_dt

    cls.call_monitored = call_monitored
    return cls


__all__ = [
    "with_adaptive_step_monitoring",
    "with_fixed_step_monitoring",
]
