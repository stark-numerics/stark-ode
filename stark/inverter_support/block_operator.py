from __future__ import annotations

from stark.contracts import Block, Operator


class BlockOperator:
    """Lift per-translation operators componentwise onto blocks."""

    __slots__ = ("operators", "_call")

    def __init__(self, operators: list[Operator], check_sizes: bool = True) -> None:
        self.operators = operators
        self._call = self._call_safe if check_sizes else self._call_fast

    def __repr__(self) -> str:
        return f"BlockOperator(size={len(self.operators)!r})"

    def __str__(self) -> str:
        return f"block operator[{len(self.operators)}]"

    def __call__(self, out: Block, block: Block) -> None:
        self._call(out, block)

    def reset(self) -> None:
        for index in range(len(self.operators)):
            self.operators[index] = None  # type: ignore[list-item]

    def _call_safe(self, out: Block, block: Block) -> None:
        if len(self.operators) != len(out) or len(out) != len(block):
            raise ValueError("BlockOperator sizes must match the input and output blocks.")
        self._call_fast(out, block)

    def _call_fast(self, out: Block, block: Block) -> None:
        for index, (operator, out_item, block_item) in enumerate(zip(self.operators, out, block, strict=True)):
            operator(out_item, block_item)
            out.items[index] = out_item


__all__ = ["BlockOperator"]
