from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Protocol

from stark.contracts.audit_support import AuditRecorder
from stark.contracts.intervals import IntervalLike
from stark.contracts.translations import State
if TYPE_CHECKING:
    from stark.execution.executor import Executor
    from stark.schemes.descriptor import SchemeDescriptor


class SchemeLike(Protocol):
    """
    Minimal protocol accepted by STARK for one-step integration schemes.

    A custom scheme does not need to inherit from STARK's internal helper
    classes as long as it satisfies this interface.

    The scheme is responsible for mutating `state` forward by one accepted step
    and returning the step size that was actually taken. Adaptive schemes may
    also update `interval.step` to propose the next step size.
    """

    def __call__(self, interval: IntervalLike, state: State, executor: Executor) -> float:
        ...

    def snapshot_state(self, state: State) -> State:
        ...

    def set_apply_delta_safety(self, enabled: bool) -> None:
        ...


class Scheme(SchemeLike, Protocol):
    """
    Richer scheme protocol for STARK's built-in, tableau-backed schemes.

    This adds the descriptor and tableau metadata used for readable reporting,
    table display, and package-level exports.
    """

    descriptor: SchemeDescriptor
    tableau: Any

    @classmethod
    def display_tableau(cls) -> str:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def __format__(self, format_spec: str) -> str:
        ...


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


class IntegratorLike(Protocol):
    """
    Protocol for trajectory-building workers built on top of a marcher.

    Integrators repeatedly call a marcher until the interval reaches its stop
    time, yielding either snapshot copies or live mutable objects along the
    way.
    """

    def __call__(
        self,
        marcher: MarcherLike,
        interval: IntervalLike,
        state: State,
        checkpoints: Any | None = None,
    ) -> Iterator[tuple[IntervalLike, State]]:
        ...

    def live(
        self,
        marcher: MarcherLike,
        interval: IntervalLike,
        state: State,
        checkpoints: Any | None = None,
    ) -> Iterator[tuple[IntervalLike, State]]:
        ...


class IntegrationAudit:
    @staticmethod
    def scheme(recorder: AuditRecorder, scheme: Any) -> None:
        recorder.check(
            callable(scheme),
            "Scheme provides __call__(interval, state, executor).",
            "Add __call__(interval, state, executor) returning the accepted step size.",
        )
        recorder.check(
            callable(getattr(scheme, "snapshot_state", None)),
            "Scheme provides snapshot_state(state).",
            "Add snapshot_state(state) so snapshot integration can clone the state.",
        )
        recorder.check(
            callable(getattr(scheme, "set_apply_delta_safety", None)),
            "Scheme provides set_apply_delta_safety(enabled).",
            "Add set_apply_delta_safety(enabled) to control alias-safe state updates.",
        )

    @staticmethod
    def marcher(recorder: AuditRecorder, marcher: Any, *, snapshots: bool) -> None:
        recorder.check(callable(marcher), "Marcher object is callable.", "Provide a marcher(interval, state) callable.")
        if snapshots:
            recorder.check(
                callable(getattr(marcher, "snapshot_state", None)),
                "Marcher provides snapshot_state(state) for snapshot integration.",
                "Use Marcher(...) or add snapshot_state(state) before calling Integrator(...).",
            )


__all__ = [
    "IntegrationAudit",
    "IntegratorLike",
    "MarcherLike",
    "Scheme",
    "SchemeLike",
]






