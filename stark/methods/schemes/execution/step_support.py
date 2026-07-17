from __future__ import annotations

from typing import Generic
from warnings import warn

from stark.core.contracts import AllocatorLike, LinearCombine, StateType, TranslationType
from stark.methods.schemes.execution.interval import SchemeShiftedInterval


SCHEME_STEP_SUPPORT_LINEAR_COMBINE_ARITY = 12


class SchemeStepSupport(Generic[StateType, TranslationType]):
    """Step-local workspace for one concrete state and translation family.

    Built-in schemes all need the same boring machinery: allocate scratch
    states, allocate scratch translations, copy a state snapshot, shift an
    interval to a stage time, and apply linear-combination kernels. This class
    keeps that machinery in one object while preserving the exact state and
    translation types supplied by the engine allocator.

    The methods are intentionally small. Concrete schemes copy the hot-path
    callables they need from this object and then run their tableau algorithm
    directly, without asking the allocator or engine questions inside each
    stage.
    """

    __slots__ = (
        "allocate_state",
        "allocate_translation",
        "copy_state",
        "interval_at",
        "scale",
        "combine2",
        "combine3",
        "combine4",
        "combine5",
        "combine6",
        "combine7",
        "combine8",
        "combine9",
        "combine10",
        "combine11",
        "combine12",
    )

    def __init__(
        self,
        allocator: AllocatorLike[StateType, TranslationType],
        linear_combine: LinearCombine,
    ) -> None:
        if len(linear_combine) < SCHEME_STEP_SUPPORT_LINEAR_COMBINE_ARITY:
            message = (
                "SchemeStepSupport requires a prepared linear_combine table "
                "covering translation arities 1..12; got "
                f"{len(linear_combine)} kernel(s). Decorate the allocator "
                "class with @Allocator.runtime before constructing it so STARK "
                "can prepare a complete table, or pass a complete custom "
                "linear_combine tuple."
            )
            warn(message, RuntimeWarning, stacklevel=2)
            raise ValueError(message)
        self.allocate_state = allocator.allocate_state
        self.allocate_translation = allocator.allocate_translation
        self.copy_state = allocator.copy_state
        self.interval_at = SchemeShiftedInterval()
        (
            self.scale,
            self.combine2,
            self.combine3,
            self.combine4,
            self.combine5,
            self.combine6,
            self.combine7,
            self.combine8,
            self.combine9,
            self.combine10,
            self.combine11,
            self.combine12,
        ) = linear_combine[:SCHEME_STEP_SUPPORT_LINEAR_COMBINE_ARITY]

    def allocate_state_buffer(self) -> StateType:
        return self.allocate_state()

    def allocate_translation_buffers(self, count: int) -> tuple[TranslationType, ...]:
        if count < 0:
            raise ValueError("Translation buffer count must be non-negative.")
        return tuple(self.allocate_translation() for _ in range(count))

    @staticmethod
    def apply_delta(delta: TranslationType, state: StateType) -> None:
        delta(state, state)

    def snapshot_state(self, state: StateType) -> StateType:
        snapshot = self.allocate_state()
        self.copy_state(state, snapshot)
        return snapshot


__all__ = ["SchemeStepSupport"]
