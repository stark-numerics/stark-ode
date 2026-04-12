from __future__ import annotations

from dataclasses import dataclass

from stark.contracts import Block, Operator


@dataclass(slots=True)
class BlockOperator:
    """Lift per-translation operators componentwise onto blocks."""

    operators: list[Operator]

    def __repr__(self) -> str:
        return f"BlockOperator(size={len(self.operators)!r})"

    def __str__(self) -> str:
        return f"block operator[{len(self.operators)}]"

    def __call__(self, out: Block, block: Block) -> None:
        if len(self.operators) != len(out) or len(out) != len(block):
            raise ValueError("BlockOperator sizes must match the input and output blocks.")
        for index, (operator, out_item, block_item) in enumerate(zip(self.operators, out, block, strict=True)):
            result = operator(out_item, block_item)
            out.items[index] = out_item if result is None else result


__all__ = ["BlockOperator"]
