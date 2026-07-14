"""Contracts for state and translation allocators."""

from __future__ import annotations

from typing import Any, Protocol

from stark.core.contracts.contract_audit import AuditRecorder
from stark.core.contracts.state import State, StateType
from stark.core.contracts.translation import Translation, TranslationTypeCovariant


class AllocatorLike(Protocol[StateType, TranslationTypeCovariant]):
    """
    Factory for reusable scratch objects and state-copy operations.

    This is the main integration point for user-defined state types. A custom
    allocator tells STARK how to:

    - allocate mutable state objects
    - copy one state into another
    - allocate translation objects compatible with that state

    `AllocatorLike` is generic in the state and translation it owns. This lets a
    concrete allocator such as `AllocatorLike[ParticleState, ParticleDelta]` satisfy
    the same public contract without pretending its return values are only the
    broad `State` and `Translation` protocols.

    The state type is invariant because `copy_state` both consumes and writes
    state objects. The translation type is covariant because this protocol only
    produces translations.
    """

    def allocate_state(self) -> StateType:
        """Return a blank mutable state object owned by this allocator."""
        ...

    def copy_state(self, source: StateType, out: StateType) -> Any:
        """Copy `source` into `out` without allocating a replacement state."""
        ...

    def allocate_translation(self) -> TranslationTypeCovariant:
        """Return a blank translation compatible with the allocated state."""
        ...


class AllocatorAudit:
    """Audit state and translation allocation support."""

    @staticmethod
    def __call__(
        recorder: AuditRecorder,
        allocator: Any,
        *,
        exercise: bool = True,
    ) -> tuple[Any | None, Any | None, Any | None]:
        allocate_state = getattr(allocator, "allocate_state", None)
        copy_state = getattr(allocator, "copy_state", None)
        allocate_translation = getattr(allocator, "allocate_translation", None)

        recorder.check(callable(allocate_state), "Allocator provides allocate_state().", "Add allocate_state() returning a blank mutable state.")
        recorder.check(callable(copy_state), "Allocator provides copy_state(source, out).", "Add copy_state(source, out) to support safe updates and snapshots.")
        recorder.check(callable(allocate_translation), "Allocator provides allocate_translation().", "Add allocate_translation() returning a blank translation.")

        if not exercise:
            return None, None, None

        sample_state = None
        second_state = None
        sample_translation = None

        if callable(allocate_state):
            try:
                sample_state = allocate_state()
                second_state = allocate_state()
            except Exception as exc:
                recorder.record_exception("Allocator.allocate_state() succeeds.", exc)
            else:
                recorder.check(True, "Allocator.allocate_state() succeeds.")

        if callable(allocate_translation):
            try:
                sample_translation = allocate_translation()
            except Exception as exc:
                recorder.record_exception("Allocator.allocate_translation() succeeds.", exc)
            else:
                recorder.check(True, "Allocator.allocate_translation() succeeds.")

        return sample_state, second_state, sample_translation

    @staticmethod
    def exercise_copy_state(recorder: AuditRecorder, allocator: Any, source: Any, out: Any) -> None:
        copy_state = getattr(allocator, "copy_state", None)
        if not callable(copy_state):
            return
        try:
            copy_state(source, out)
        except Exception as exc:
            recorder.record_exception("Allocator.copy_state(source, out) can copy a provided state.", exc)
        else:
            recorder.check(True, "Allocator.copy_state(source, out) can copy a provided state.")

    @staticmethod
    def exercise_allocator_copy(recorder: AuditRecorder, allocator: Any, source: Any, out: Any) -> None:
        copy_state = getattr(allocator, "copy_state", None)
        if not callable(copy_state):
            return
        try:
            copy_state(source, out)
        except Exception as exc:
            recorder.record_exception("Allocator.copy_state(source, out) succeeds on blank states.", exc)
        else:
            recorder.check(True, "Allocator.copy_state(source, out) succeeds on blank states.")


__all__ = ["AllocatorLike", "AllocatorAudit"]
