from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from stark.contracts.intervals import IntervalLike
from stark.contracts import Translation, Workbench
from stark.algebraist.combine import AlgebraistCombineResolver


class _StageInterval:
    """Reusable interval snapshot placed at a stage time within the current step."""

    __slots__ = ("interval",)

    def __init__(self) -> None:
        self.interval: IntervalLike | None = None

    def __call__(self, interval: IntervalLike, step: float, shift: float) -> IntervalLike:
        stage = self.interval
        if stage is None:
            stage = interval.copy()
            self.interval = stage
        stage.present = interval.present + shift
        stage.step = step
        stage.stop = interval.stop
        return stage


@dataclass(slots=True, init=False)
class SchemeWorkspace:
    """
    Reusable stage-bound support for built-in schemes and resolvents.

    This worker owns scratch allocation, state snapshots, reusable stage-interval
    placement, and resolved translation linear-combination kernels.
    """

    allocate_state: Callable[[], object]
    allocate_translation: Callable[[], Translation]
    copy_state: Callable[[object, object], None]
    state_buffer: object
    stage_interval: _StageInterval
    apply_delta: Callable[[Translation, object], None]
    scale: Callable[..., Translation]
    combine2: Callable[..., Translation]
    combine3: Callable[..., Translation]
    combine4: Callable[..., Translation]
    combine5: Callable[..., Translation]
    combine6: Callable[..., Translation]
    combine7: Callable[..., Translation]
    combine8: Callable[..., Translation]
    combine9: Callable[..., Translation]
    combine10: Callable[..., Translation]
    combine11: Callable[..., Translation]
    combine12: Callable[..., Translation]

    def __init__(self, workbench: Workbench, translation: Translation) -> None:
        self.allocate_state = workbench.allocate_state
        self.allocate_translation = workbench.allocate_translation
        self.copy_state = workbench.copy_state
        self.state_buffer = workbench.allocate_state()
        self.stage_interval = _StageInterval()
        self.apply_delta = self.apply_delta_safe
        combine_resolver = AlgebraistCombineResolver.from_translation(
            translation,
            workbench.allocate_translation,
            )
        self.scale = combine_resolver.scale
        self.combine2 = combine_resolver.combine2
        self.combine3 = combine_resolver.combine3
        self.combine4 = combine_resolver.combine4
        self.combine5 = combine_resolver.combine5
        self.combine6 = combine_resolver.combine6
        self.combine7 = combine_resolver.combine7
        self.combine8 = combine_resolver.combine8
        self.combine9 = combine_resolver.combine9
        self.combine10 = combine_resolver.combine10
        self.combine11 = combine_resolver.combine11
        self.combine12 = combine_resolver.combine12

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

    def stage_at(self, interval: IntervalLike, step: float, shift: float) -> IntervalLike:
        return self.stage_interval(interval, step, shift)

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


__all__ = ["SchemeWorkspace"]


