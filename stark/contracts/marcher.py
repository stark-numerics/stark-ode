"""Contracts for step-accepting marchers."""

from __future__ import annotations

from typing import Any, Protocol

from stark.contracts.contract_audit import AuditRecorder
from stark.contracts.interval import IntervalLike
from stark.contracts.state import State


class MarcherLike(Protocol):
    """
    Minimal protocol for step-accepting marchers.

    A marcher owns a scheme together with an executor and mutates a bound
    `(interval, state)` pair forward by one accepted step. Snapshot integration
    also asks the marcher for cloned state objects via `snapshot_state(...)`.
    """

    def __call__(self, interval: IntervalLike, state: State) -> None:
        ...

    def snapshot_state(self, state: State) -> State:
        ...


class MarcherAudit:
    """Audit a marcher object."""

    @staticmethod
    def __call__(recorder: AuditRecorder, marcher: Any, *, snapshots: bool) -> None:
        recorder.check(callable(marcher), "Marcher object is callable.", "Provide a marcher(interval, state) callable.")
        if snapshots:
            recorder.check(
                callable(getattr(marcher, "snapshot_state", None)),
                "Marcher provides snapshot_state(state) for snapshot integration.",
                "Use Marcher(...) or add snapshot_state(state) before calling Integrator(...).",
            )


__all__ = ["MarcherAudit", "MarcherLike"]
