from __future__ import annotations

from stark.algebraist.runtime import AlgebraistRuntimeLinearCombine
from stark.contracts import Allocator, State, Translation
from stark.schemes.execution.interval import SchemeShiftedInterval


class SchemeStepSupport:
    """Step-local allocation and arithmetic helpers for built-in schemes."""

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

    def __init__(self, allocator: Allocator, translation: Translation) -> None:
        self.allocate_state = allocator.allocate_state
        self.allocate_translation = allocator.allocate_translation
        self.copy_state = allocator.copy_state
        self.interval_at = SchemeShiftedInterval()
        algebraist = AlgebraistRuntimeLinearCombine(
            translation,
            allocator=allocator,
        )
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
        ) = algebraist.as_tuple(12)

    def allocate_state_buffer(self) -> State:
        return self.allocate_state()

    def allocate_translation_buffers(self, count: int) -> tuple[Translation, ...]:
        if count < 0:
            raise ValueError("Translation buffer count must be non-negative.")
        return tuple(self.allocate_translation() for _ in range(count))

    @staticmethod
    def apply_delta(delta: Translation, state: State) -> None:
        delta(state, state)

    def snapshot_state(self, state: State) -> State:
        snapshot = self.allocate_state()
        self.copy_state(state, snapshot)
        return snapshot


__all__ = ["SchemeStepSupport"]
