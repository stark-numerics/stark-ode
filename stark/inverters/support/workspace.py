from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Callable

from stark.accelerators import AcceleratorAbsent
from stark.contracts import AcceleratorLike, Block, InnerProduct, Translation, Workbench
from stark.execution.safety import Safety
from stark.machinery.translation_algebra.linear_combine import Combiner, resolve_linear_combine


@dataclass(slots=True, init=False)
class InverterWorkspace:
    """Reusable scratch support for bundle-level linear inversion."""

    accelerator: AcceleratorLike
    allocate_translation: Callable[[], Translation]
    inner_product_translation: InnerProduct
    linear_combine: tuple[Callable[..., object], ...]
    scale: Callable[..., Translation]
    combine2: Callable[..., Translation]
    combine3: Callable[..., Translation]
    safety: Safety
    _check: Callable[..., None]

    def __init__(
        self,
        workbench: Workbench,
        translation: Translation,
        inner_product: InnerProduct,
        safety: Safety | None = None,
        accelerator: AcceleratorLike | None = None,
    ) -> None:
        self.accelerator = accelerator if accelerator is not None else AcceleratorAbsent()
        self.allocate_translation = workbench.allocate_translation
        self.inner_product_translation = inner_product
        self.linear_combine = resolve_linear_combine(translation)
        self.safety = safety if safety is not None else Safety()
        self._check = self._check_size if self.safety.block_sizes else self._skip_check
        self.bind_accelerator(self.accelerator)

    def bind_accelerator(self, accelerator: AcceleratorLike) -> None:
        self.accelerator = accelerator
        combiner = accelerator.resolve_support(Combiner(self.linear_combine, self.allocate_translation), label="inverter_combiner")
        self.scale = combiner.scale
        self.combine2 = combiner.combine2
        self.combine3 = combiner.combine3

    def __repr__(self) -> str:
        allocate_translation_name = getattr(
            self.allocate_translation,
            "__qualname__",
            type(self.allocate_translation).__name__,
        )
        inner_product_name = getattr(
            self.inner_product_translation,
            "__qualname__",
            type(self.inner_product_translation).__name__,
        )
        return (
            "InverterWorkspace("
            f"allocate_translation={allocate_translation_name!r}, "
            f"inner_product={inner_product_name!r}, "
            f"accelerator={self.accelerator!r})"
        )

    def __str__(self) -> str:
        return "inverter workspace"

    def allocate_block(self, count: int) -> Block:
        if count < 0:
            raise ValueError("Block size must be non-negative.")
        return Block([self.allocate_translation() for _ in range(count)])

    def zero_block(self, block: Block) -> None:
        for index, item in enumerate(block):
            block.items[index] = self.scale(item, 0.0, item)

    def copy_block(self, dst: Block, src: Block) -> None:
        self._check(dst, src)
        for index, (dst_item, src_item) in enumerate(zip(dst, src, strict=True)):
            dst.items[index] = self.scale(dst_item, 1.0, src_item)

    def scale_block(self, out: Block, a: float, block: Block) -> None:
        self._check(out, block)
        for index, (out_item, block_item) in enumerate(zip(out, block, strict=True)):
            out.items[index] = self.scale(out_item, a, block_item)

    def combine2_block(self, out: Block, a0: float, x0: Block, a1: float, x1: Block) -> None:
        self._check(out, x0, x1)
        for index, (out_item, x0_item, x1_item) in enumerate(zip(out, x0, x1, strict=True)):
            out.items[index] = self.combine2(out_item, a0, x0_item, a1, x1_item)

    def combine3_block(
        self,
        out: Block,
        a0: float,
        x0: Block,
        a1: float,
        x1: Block,
        a2: float,
        x2: Block,
    ) -> None:
        self._check(out, x0, x1, x2)
        for index, (out_item, x0_item, x1_item, x2_item) in enumerate(zip(out, x0, x1, x2, strict=True)):
            out.items[index] = self.combine3(out_item, a0, x0_item, a1, x1_item, a2, x2_item)

    def inner_product(self, left: Block, right: Block) -> float:
        self._check(left, right)
        return sum(
            self.inner_product_translation(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )

    def norm(self, block: Block) -> float:
        return sqrt(max(0.0, self.inner_product(block, block)))

    @staticmethod
    def _check_size(*blocks: Block) -> None:
        if not blocks:
            return
        size = len(blocks[0])
        if any(len(block) != size for block in blocks[1:]):
            raise ValueError("Block sizes must match.")

    @staticmethod
    def _skip_check(*blocks: Block) -> None:
        del blocks


__all__ = ["InverterWorkspace"]












