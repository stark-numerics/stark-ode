from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from stark.contracts import Block, Translation, Workbench
from stark.scheme_support.linear_combine import complete_linear_combine, resolve_linear_combine


@dataclass(slots=True, init=False)
class ResolverWorkspace:
    """Reusable scratch support for nonlinear resolvers."""

    allocate_translation: Callable[[], Translation]
    scale: Callable[..., Translation]
    combine2: Callable[..., Translation]
    combine3: Callable[..., Translation]
    combine4: Callable[..., Translation]
    combine5: Callable[..., Translation]
    combine6: Callable[..., Translation]
    combine7: Callable[..., Translation]

    def __init__(self, workbench: Workbench, translation: Translation) -> None:
        self.allocate_translation = workbench.allocate_translation
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
        allocate_translation_name = getattr(
            self.allocate_translation,
            "__qualname__",
            type(self.allocate_translation).__name__,
        )
        return f"ResolverWorkspace(allocate_translation={allocate_translation_name!r})"

    def __str__(self) -> str:
        return "resolver workspace"

    def allocate_block(self, count: int) -> Block:
        if count < 0:
            raise ValueError("Block size must be non-negative.")
        return Block([self.allocate_translation() for _ in range(count)])

    def zero_block(self, block: Block) -> None:
        for index, item in enumerate(block):
            block.items[index] = self.scale(item, 0.0, item)

    def copy_block(self, dst: Block, src: Block) -> None:
        self._check_size(dst, src)
        for index, (dst_item, src_item) in enumerate(zip(dst, src, strict=True)):
            dst.items[index] = self.scale(dst_item, 1.0, src_item)

    def combine2_block(self, out: Block, a0: float, x0: Block, a1: float, x1: Block) -> None:
        self._check_size(out, x0, x1)
        for index, (out_item, x0_item, x1_item) in enumerate(zip(out, x0, x1, strict=True)):
            out.items[index] = self.combine2(out_item, a0, x0_item, a1, x1_item)

    @staticmethod
    def _check_size(*blocks: Block) -> None:
        if not blocks:
            return
        size = len(blocks[0])
        if any(len(block) != size for block in blocks[1:]):
            raise ValueError("Block sizes must match.")


__all__ = ["ResolverWorkspace"]
