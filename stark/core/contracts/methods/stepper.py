"""Contracts for step-accepting steppers."""

from __future__ import annotations

from typing import Any, Protocol

from stark.core.contracts.shared.contract_audit import AuditRecorder
from stark.core.contracts.shared.interval import IntervalLike
from stark.core.contracts.problem.state import State


class IntegratorStepperLike(Protocol):
    """
    Minimal protocol for step-accepting steppers.

    A stepper owns a scheme together with an executor and mutates a bound
    `(interval, state)` pair forward by one accepted step. Snapshot integration
    also asks the stepper for cloned state objects via `snapshot_state(...)`.
    """

    def __call__(self, interval: IntervalLike, state: State) -> None:
        ...

    def snapshot_state(self, state: State) -> State:
        ...


class IntegratorStepperAudit:
    """Audit a stepper object."""

    @staticmethod
    def __call__(recorder: AuditRecorder, stepper: Any, *, snapshots: bool) -> None:
        recorder.check(callable(stepper), "IntegratorStepper object is callable.", "Provide a stepper(interval, state) callable.")
        if snapshots:
            recorder.check(
                callable(getattr(stepper, "snapshot_state", None)),
                "IntegratorStepper provides snapshot_state(state) for snapshot integration.",
                "Use IntegratorStepper(...) or add snapshot_state(state) before calling Integrator(...).",
            )


__all__ = ["IntegratorStepperAudit", "IntegratorStepperLike"]
