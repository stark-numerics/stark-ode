"""Contracts for right-hand-side dynamics workers."""

from __future__ import annotations

from typing import Any, Protocol

from stark.core.contracts.contract_audit import AuditRecorder
from stark.core.contracts.interval import IntervalLike
from stark.core.contracts.state import State, StateTypeContravariant
from stark.core.contracts.translation import Translation, TranslationTypeContravariant


class DynamicsLike(Protocol[StateTypeContravariant, TranslationTypeContravariant]):
    """
    Fill `out` with the dynamics at `(interval.present, state)`.

    Dynamicss are output-last and mutate the supplied translation. They do
    not allocate, return, or own time-stepping policy. Schemes decide when and
    where the dynamics is sampled.

    The protocol is generic in the concrete state and translation. This is what
    users naturally write: a dynamics for `ScalarState` writes a
    `ScalarTranslation`. The type variables are contravariant because the
    dynamics consumes both objects rather than producing them.
    """

    def __call__(
        self,
        interval: IntervalLike,
        state: StateTypeContravariant,
        out: TranslationTypeContravariant,
    ) -> None:
        """Write the dynamics at `interval` and `state` into `out`."""
        ...


class DynamicsAudit:
    """Audit a right-hand-side dynamics worker."""

    @staticmethod
    def __call__(recorder: AuditRecorder, dynamics: Any) -> None:
        recorder.check(
            callable(dynamics),
            "Dynamics is callable.",
            "Provide a callable dynamics(interval, state, translation).",
        )

    @staticmethod
    def exercise(
        recorder: AuditRecorder,
        dynamics: Any,
        interval: Any,
        state: Any,
        translation: Any,
    ) -> None:
        try:
            dynamics(interval, state, translation)
        except Exception as exc:
            recorder.record_exception("Dynamics(interval, state, translation) can be called.", exc)
        else:
            recorder.check(True, "Dynamics(interval, state, translation) can be called.")


__all__ = ["DynamicsLike", "DynamicsAudit"]
