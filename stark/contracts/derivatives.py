"""Contracts for right-hand-side derivative workers."""

from __future__ import annotations

from typing import Any, Protocol

from stark.contracts.audit_support import AuditRecorder
from stark.contracts.intervals import IntervalLike
from stark.contracts.states import State
from stark.contracts.translations import Translation


class Derivative(Protocol):
    """
    Fill `out` with the derivative at `(interval.present, state)`.

    Derivatives are output-last and mutate the supplied translation. They do
    not allocate, return, or own time-stepping policy. Schemes decide when and
    where the derivative is sampled.
    """

    def __call__(self, interval: IntervalLike, state: State, out: Translation) -> None:
        ...


class DerivativeAudit:
    """Audit a right-hand-side derivative worker."""

    @staticmethod
    def __call__(recorder: AuditRecorder, derivative: Any) -> None:
        recorder.check(
            callable(derivative),
            "Derivative is callable.",
            "Provide a callable derivative(interval, state, translation).",
        )

    @staticmethod
    def exercise(
        recorder: AuditRecorder,
        derivative: Any,
        interval: Any,
        state: Any,
        translation: Any,
    ) -> None:
        try:
            derivative(interval, state, translation)
        except Exception as exc:
            recorder.record_exception("Derivative(interval, state, translation) can be called.", exc)
        else:
            recorder.check(True, "Derivative(interval, state, translation) can be called.")


__all__ = ["Derivative", "DerivativeAudit"]
