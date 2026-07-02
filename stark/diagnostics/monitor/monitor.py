"""Aggregate monitor for scheme, resolvent, and inverter observations."""

from __future__ import annotations

from dataclasses import dataclass, field

from stark.diagnostics.monitor.inverter import MonitorInverter, MonitorInverterSummary
from stark.diagnostics.monitor.resolvent import MonitorResolvent, MonitorResolventSummary
from stark.diagnostics.monitor.scheme import MonitorScheme, MonitorSchemeSummary


@dataclass(frozen=True, slots=True)
class MonitorSummary:
    """Snapshot summary gathered from all monitor channels."""

    scheme: MonitorSchemeSummary
    resolvent: MonitorResolventSummary
    inverter: MonitorInverterSummary


@dataclass(slots=True)
class Monitor:
    """Collect optional diagnostic evidence during an observed solve.

    Monitoring is intentionally separate from the hot path. Schemes,
    resolvents, and inverters can be decorated or prepared with monitor-aware
    variants when a user asks for diagnostic evidence; ordinary solves do not
    need to pay for recording lists of events.
    """

    scheme: MonitorScheme = field(default_factory=MonitorScheme)
    resolvent: MonitorResolvent = field(default_factory=MonitorResolvent)
    inverter: MonitorInverter = field(default_factory=MonitorInverter)

    def summary(self) -> MonitorSummary:
        return MonitorSummary(
            scheme=self.scheme.summary(),
            resolvent=self.resolvent.summary(),
            inverter=self.inverter.summary(),
        )

    def clear(self) -> None:
        self.scheme.clear()
        self.resolvent.clear()
        self.inverter.clear()


__all__ = [
    "Monitor",
    "MonitorSummary",
]
