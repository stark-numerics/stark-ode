"""Contracts for problem linearizers."""

from __future__ import annotations

from typing import Any, Protocol

from stark.contracts.contract_audit import AuditRecorder
from stark.contracts.interval import IntervalLike
from stark.contracts.operator import Operator
from stark.contracts.state import State
from stark.contracts.translation import Translation


class _LinearizerOperatorProbe:
    __slots__ = ("apply",)

    def __init__(self) -> None:
        self.apply = self._unset

    @staticmethod
    def _unset(translation: Translation, out: Translation) -> None:
        del translation, out
        raise RuntimeError("Linearizer did not configure the operator probe.")

    def __call__(self, translation: Translation, out: Translation) -> None:
        self.apply(translation, out)


class Linearizer(Protocol):
    """
    Fill `out` with the local Jacobian action of the derivative at `state`.

    This is the contract that asks the user to do some problem-specific maths.
    Given a nonlinear derivative

        x' = f(t, x),

    the linearizer must provide the action of the Jacobian

        J(t, state)[translation]

    as an `Operator`. STARK does not ask for a dense matrix. It asks for a
    callable linear operator that, given an input translation, writes the
    Jacobian image into `out`.
    """

    def __call__(self, interval: IntervalLike, state: State, out: Operator) -> None:
        ...


class LinearizerAudit:
    """Audit a linearizer and the operator it configures."""

    @staticmethod
    def __call__(recorder: AuditRecorder, linearizer: Any) -> None:
        recorder.check(
            callable(linearizer),
            "Linearizer provides __call__(interval, state, out).",
            "Provide a callable linearizer(interval, state, out).",
        )

    @staticmethod
    def exercise(
        recorder: AuditRecorder,
        linearizer: Any,
        interval: Any,
        state: Any,
        translation: Translation,
    ) -> None:
        operator = _LinearizerOperatorProbe()
        try:
            linearizer(interval, state, operator)
        except Exception as exc:
            recorder.record_exception("Linearizer(interval, state, out) can be called.", exc)
            return
        else:
            recorder.check(True, "Linearizer(interval, state, out) can be called.")

        try:
            result = operator(translation, translation)
        except Exception as exc:
            recorder.record_exception("Linearizer configures an operator callable.", exc)
            return
        else:
            recorder.check(True, "Linearizer configures an operator callable.")
            recorder.check(
                result is None,
                "Linearizer-configured operators fill in place and return None.",
                "Make operator(translation, out) mutate out and return None.",
            )


__all__ = ["Linearizer", "LinearizerAudit"]
