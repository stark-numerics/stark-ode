from __future__ import annotations

from dataclasses import dataclass, field

from stark.monitor.resolvent import MonitorResolvent, MonitorSummaryResolvent
from stark.monitor.scheme import MonitorScheme, MonitorSummaryScheme


@dataclass(frozen=True, slots=True)
class MonitorSummary:
    scheme: MonitorSummaryScheme
    resolvent: MonitorSummaryResolvent


@dataclass(slots=True)
class Monitor:
    scheme: MonitorScheme = field(default_factory=MonitorScheme)
    resolvent: MonitorResolvent = field(default_factory=MonitorResolvent)

    @classmethod
    def with_scheme(
        cls,
        scheme: MonitorScheme | None = None,
        resolvent: MonitorResolvent | None = None,
    ) -> Monitor:
        return cls(
            scheme=MonitorScheme() if scheme is None else scheme,
            resolvent=MonitorResolvent() if resolvent is None else resolvent,
        )

    def summary(self) -> MonitorSummary:
        return MonitorSummary(
            scheme=self.scheme.summary(),
            resolvent=self.resolvent.summary(),
        )

    def clear(self) -> None:
        self.scheme.clear()
        self.resolvent.clear()


__all__ = [
    "Monitor",
    "MonitorSummary",
]
