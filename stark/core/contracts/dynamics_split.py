"""Protocol and audit for implicit-explicit dynamics splits."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from stark.core.contracts.contract_audit import AuditRecorder
from stark.core.contracts.dynamics import DynamicsLike
from stark.core.contracts.state import StateTypeContravariant
from stark.core.contracts.translation import TranslationTypeContravariant


@runtime_checkable
class DynamicsSplitLike(Protocol[StateTypeContravariant, TranslationTypeContravariant]):
    """
    Scheme-facing implicit-explicit dynamics split.

    IMEX schemes need two dynamics workers: one treated implicitly and one
    treated explicitly. Both workers use the standard dynamics call shape
    `worker(interval, state, translation)` and write into the supplied
    translation object.

    Both parts share the same state and translation consumer types. The
    protocol is contravariant for the same reason as `DynamicsLike`: a split
    dynamics that can consume a broader state or translation family can stand
    in for one that only needs to consume a narrower family.
    """

    @property
    def implicit(
        self,
    ) -> DynamicsLike[StateTypeContravariant, TranslationTypeContravariant]:
        """Dynamics worker used for the implicit part of the split."""
        ...

    @property
    def explicit(
        self,
    ) -> DynamicsLike[StateTypeContravariant, TranslationTypeContravariant]:
        """Dynamics worker used for the explicit part of the split."""
        ...


class DynamicsSplitAudit:
    """Audit an implicit-explicit dynamics split."""

    @staticmethod
    def __call__(recorder: AuditRecorder, split: Any) -> None:
        implicit = getattr(split, "implicit", None)
        explicit = getattr(split, "explicit", None)
        recorder.check(
            callable(implicit),
            "DynamicsSplit provides implicit(interval, state, translation).",
            "Set split.implicit to a callable dynamics worker.",
        )
        recorder.check(
            callable(explicit),
            "DynamicsSplit provides explicit(interval, state, translation).",
            "Set split.explicit to a callable dynamics worker.",
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
                recorder.record_exception("DynamicsSplit implicit part can be called.", exc)
            else:
                recorder.check(True, "DynamicsSplit implicit part can be called.")
        if callable(explicit):
            try:
                explicit(interval, state, translation)
            except Exception as exc:
                recorder.record_exception("DynamicsSplit explicit part can be called.", exc)
            else:
                recorder.check(True, "DynamicsSplit explicit part can be called.")


__all__ = ["DynamicsSplitAudit", "DynamicsSplitLike"]
