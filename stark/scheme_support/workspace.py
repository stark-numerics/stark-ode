from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from stark.contracts import Translation, Workbench
from stark.scheme_support.linear_combine import (
    complete_linear_combine,
    resolve_linear_combine,
)


@dataclass(slots=True, init=False)
class SchemeWorkspace:
    """
    Reusable non-mathematical support for hand-written STARK schemes.

    A scheme owns one `SchemeWorkspace` object and pulls out only the bits it
    needs: scratch allocation, state application, snapshots, and resolved
    linear-combination kernels.
    """

    allocate_state: Callable[[], object]
    allocate_translation: Callable[[], Translation]
    copy_state: Callable[[object, object], None]
    state_buffer: object
    apply_delta: Callable[[Translation, object], None]
    scale: Callable[..., Translation]
    combine2: Callable[..., Translation]
    combine3: Callable[..., Translation]
    combine4: Callable[..., Translation]
    combine5: Callable[..., Translation]
    combine6: Callable[..., Translation]
    combine7: Callable[..., Translation]

    def __init__(self, workbench: Workbench, translation: Translation) -> None:
        self.allocate_state = workbench.allocate_state
        self.allocate_translation = workbench.allocate_translation
        self.copy_state = workbench.copy_state
        self.state_buffer = workbench.allocate_state()
        self.apply_delta = self.apply_delta_safe
        linear_combine = resolve_linear_combine(translation)
        (
            self.scale,
            self.combine2,
            self.combine3,
            self.combine4,
            self.combine5,
            self.combine6,
            self.combine7,
        ) = complete_linear_combine(linear_combine, workbench.allocate_translation)

    def __repr__(self) -> str:
        allocate_state_name = getattr(self.allocate_state, "__qualname__", type(self.allocate_state).__name__)
        allocate_translation_name = getattr(
            self.allocate_translation,
            "__qualname__",
            type(self.allocate_translation).__name__,
        )
        copy_state_name = getattr(self.copy_state, "__qualname__", type(self.copy_state).__name__)
        return (
            "SchemeWorkspace("
            f"allocate_state={allocate_state_name!r}, "
            f"allocate_translation={allocate_translation_name!r}, "
            f"copy_state={copy_state_name!r})"
        )

    def __str__(self) -> str:
        return "scheme workspace"

    def allocate_state_buffer(self) -> object:
        return self.allocate_state()

    def allocate_translation_buffers(self, count: int) -> tuple[Translation, ...]:
        if count < 0:
            raise ValueError("Translation buffer count must be non-negative.")
        return tuple(self.allocate_translation() for _ in range(count))

    def apply_delta_safe(self, delta: Translation, state: object) -> None:
        delta(state, self.state_buffer)
        self.copy_state(state, self.state_buffer)

    @staticmethod
    def apply_delta_in_place(delta: Translation, state: object) -> None:
        delta(state, state)

    def set_apply_delta_safety(self, enabled: bool) -> None:
        self.apply_delta = self.apply_delta_safe if enabled else self.apply_delta_in_place

    def snapshot_state(self, state: object) -> object:
        snapshot = self.allocate_state()
        self.copy_state(snapshot, state)
        return snapshot
