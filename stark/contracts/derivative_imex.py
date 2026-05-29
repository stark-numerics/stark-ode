"""Carrier for implicit-explicit derivative splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stark.contracts.contract_audit import AuditRecorder
from stark.contracts.derivative import Derivative


@dataclass(frozen=True, slots=True)
class DerivativeIMEX:
    """
    Split right-hand side for implicit-explicit time stepping.

    IMEX methods treat one part of the derivative explicitly and one part
    implicitly. This small carrier keeps those two workers together in a
    single object that is easy to audit, document, and pass into a scheme.

    The full right-hand side is understood as

        f(t, x) = f_implicit(t, x) + f_explicit(t, x)

    where both parts write into translation objects in place.
    """

    implicit: Derivative
    explicit: Derivative

    @property
    def im(self) -> Derivative:
        return self.implicit

    @property
    def ex(self) -> Derivative:
        return self.explicit


class DerivativeIMEXAudit:
    """Audit an implicit-explicit derivative split."""

    @staticmethod
    def __call__(recorder: AuditRecorder, imex_derivative: Any) -> None:
        implicit = getattr(imex_derivative, "implicit", None)
        explicit = getattr(imex_derivative, "explicit", None)
        recorder.check(
            callable(implicit),
            "DerivativeIMEX provides implicit(interval, state, translation).",
            "Set imex_derivative.implicit to a callable derivative worker.",
        )
        recorder.check(
            callable(explicit),
            "DerivativeIMEX provides explicit(interval, state, translation).",
            "Set imex_derivative.explicit to a callable derivative worker.",
        )

    @staticmethod
    def exercise(
        recorder: AuditRecorder,
        imex_derivative: Any,
        interval: Any,
        state: Any,
        translation: Any,
    ) -> None:
        implicit = getattr(imex_derivative, "implicit", None)
        explicit = getattr(imex_derivative, "explicit", None)
        if callable(implicit):
            try:
                implicit(interval, state, translation)
            except Exception as exc:
                recorder.record_exception("DerivativeIMEX implicit part can be called.", exc)
            else:
                recorder.check(True, "DerivativeIMEX implicit part can be called.")
        if callable(explicit):
            try:
                explicit(interval, state, translation)
            except Exception as exc:
                recorder.record_exception("DerivativeIMEX explicit part can be called.", exc)
            else:
                recorder.check(True, "DerivativeIMEX explicit part can be called.")


__all__ = ["DerivativeIMEX", "DerivativeIMEXAudit"]
