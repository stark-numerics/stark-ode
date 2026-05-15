from __future__ import annotations

from stark.contracts import Block, Operator


class BlockOperator:
    """Lift per-translation operators componentwise onto blocks."""

    __slots__ = ("operators", "redirect_call")

    def __init__(self, operators: list[Operator], check_sizes: bool = True) -> None:
        self.operators = operators
        self.redirect_call = self.call_checked if check_sizes else self.call_unchecked

    def __repr__(self) -> str:
        return f"BlockOperator(size={len(self.operators)!r})"

    def __str__(self) -> str:
        return f"block operator[{len(self.operators)}]"

    def __call__(self, block: Block, out: Block) -> None:
        self.redirect_call(block, out)

    def reset(self) -> None:
        for index in range(len(self.operators)):
            self.operators[index] = None  # type: ignore[list-item]

    def call_checked(self, block: Block, out: Block) -> None:
        if len(self.operators) != len(out) or len(out) != len(block):
            raise ValueError("BlockOperator sizes must match the input and output blocks.")
        self.call_unchecked(block, out)

    def call_unchecked(self, block: Block, out: Block) -> None:
        for index, (operator, out_item, block_item) in enumerate(zip(self.operators, out, block, strict=True)):
            operator(block_item, out_item)
            out.items[index] = out_item


__all__ = ["BlockOperator"]









