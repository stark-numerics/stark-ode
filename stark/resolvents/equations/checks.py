from __future__ import annotations

from stark.block import Block


def check_one_stage_block(name: str, block: Block) -> None:
    if len(block) != 1:
        raise ValueError(f"{name} must be a one-item block for this resolvent.")


__all__ = ["check_one_stage_block"]
