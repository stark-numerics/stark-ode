"""Protocol and audit for implicit-explicit derivative splits."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from stark.core.contracts.contract_audit import AuditRecorder
from stark.core.contracts.derivative import DerivativeLike
from stark.core.contracts.state import StateTypeContravariant
from stark.core.contracts.translation import TranslationTypeContravariant


@runtime_checkable
class DerivativeSplitLike(Protocol[StateTypeContravariant, TranslationTypeContravariant]):
    """
    Scheme-facing implicit-explicit derivative split.

    IMEX schemes need two derivative workers: one treated implicitly and one
    treated explicitly. Both workers use the standard derivative call shape
    `worker(interval, state, translation)` and write into the supplied
    translation object.

    Both parts share the same state and translation consumer types. The
    protocol is contravariant for the same reason as `DerivativeLike`: a split
    derivative that can consume a broader state or translation family can stand
    in for one that only needs to consume a narrower family.
    """

    @property
    def implicit(
        self,
    ) -> DerivativeLike[StateTypeContravariant, TranslationTypeContravariant]:
        """Derivative worker used for the implicit part of the split."""
        ...

    @property
    def explicit(
        self,
    ) -> DerivativeLike[StateTypeContravariant, TranslationTypeContravariant]:
        """Derivative worker used for the explicit part of the split."""
        ...


class DerivativeSplitAudit:
    """Audit an implicit-explicit derivative split."""

    @staticmethod
    def __call__(recorder: AuditRecorder, split: Any) -> None:
        implicit = getattr(split, "implicit", None)
        explicit = getattr(split, "explicit", None)
        recorder.check(
            callable(implicit),
            "DerivativeSplit provides implicit(interval, state, translation).",
            "Set split.implicit to a callable derivative worker.",
        )
        recorder.check(
            callable(explicit),
            "DerivativeSplit provides explicit(interval, state, translation).",
            "Set split.explicit to a callable derivative worker.",
        )

    @staticmethod
    def exercise(
        recorder: AuditRecorder,
        split: Any,
        interval: Any,
        state: Any,
        translation: Any,
    ) -> None:
        implicit = getattr(split, "implicit", None)
        explicit = getattr(split, "explicit", None)
        if callable(implicit):
            try:
                implicit(interval, state, translation)
            except Exception as exc:
                recorder.record_exception("DerivativeSplit implicit part can be called.", exc)
            else:
                recorder.check(True, "DerivativeSplit implicit part can be called.")
        if callable(explicit):
            try:
                explicit(interval, state, translation)
            except Exception as exc:
                recorder.record_exception("DerivativeSplit explicit part can be called.", exc)
            else:
                recorder.check(True, "DerivativeSplit explicit part can be called.")


__all__ = ["DerivativeSplitAudit", "DerivativeSplitLike"]
