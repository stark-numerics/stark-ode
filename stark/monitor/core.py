from __future__ import annotations

from dataclasses import dataclass, field

from stark.monitor.inverter import MonitorInverter, MonitorSummaryInverter
from stark.monitor.resolvent import MonitorResolvent, MonitorSummaryResolvent
from stark.monitor.scheme import MonitorScheme, MonitorSummaryScheme


@dataclass(frozen=True, slots=True)
class MonitorSummary:
    scheme: MonitorSummaryScheme
    resolvent: MonitorSummaryResolvent
    inverter: MonitorSummaryInverter


@dataclass(slots=True)
class Monitor:
    scheme: MonitorScheme = field(default_factory=MonitorScheme)
    resolvent: MonitorResolvent = field(default_factory=MonitorResolvent)
    inverter: MonitorInverter = field(default_factory=MonitorInverter)

    @classmethod
    def with_scheme(
        cls,
        scheme: MonitorScheme | None = None,
        resolvent: MonitorResolvent | None = None,
        inverter: MonitorInverter | None = None,
    ) -> Monitor:
        return cls(
            scheme=MonitorScheme() if scheme is None else scheme,
            resolvent=MonitorResolvent() if resolvent is None else resolvent,
            inverter=MonitorInverter() if inverter is None else inverter,
        )

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
