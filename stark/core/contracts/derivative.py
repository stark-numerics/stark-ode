"""Contracts for right-hand-side derivative workers."""

from __future__ import annotations

from typing import Any, Protocol

from stark.core.contracts.contract_audit import AuditRecorder
from stark.core.contracts.interval import IntervalLike
from stark.core.contracts.state import State, StateTypeContravariant
from stark.core.contracts.translation import Translation, TranslationTypeContravariant


class DerivativeLike(Protocol[StateTypeContravariant, TranslationTypeContravariant]):
    """
    Fill `out` with the derivative at `(interval.present, state)`.

    Derivatives are output-last and mutate the supplied translation. They do
    not allocate, return, or own time-stepping policy. Schemes decide when and
    where the derivative is sampled.

    The protocol is generic in the concrete state and translation. This is what
    users naturally write: a derivative for `ScalarState` writes a
    `ScalarTranslation`. The type variables are contravariant because the
    derivative consumes both objects rather than producing them.
    """

    def __call__(
        self,
        interval: IntervalLike,
        state: StateTypeContravariant,
        out: TranslationTypeContravariant,
    ) -> None:
        """Write the derivative at `interval` and `state` into `out`."""
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


__all__ = ["DerivativeLike", "DerivativeAudit"]
