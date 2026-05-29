"""Contract-level block container for product-space solver objects."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from stark.contracts.translations import Translation


@dataclass(slots=True)
class Block:
    """
    A grouped solver-space object built from one or more translations.

    Implicit schemes, resolvents, and inverters work in a product space of
    translations rather than on single translations alone. A one-stage implicit
    method therefore uses a one-item block, while multi-stage methods and
    quasi-Newton histories can use larger blocks.
    """

    items: list[Translation]

    def __repr__(self) -> str:
        return f"Block(size={len(self.items)!r})"

    def __str__(self) -> str:
        return f"block[{len(self.items)}]"

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, index: int) -> Translation:
        return self.items[index]

    def norm(self) -> float:
        if not self.items:
            return 0.0
        return sqrt(sum(item.norm() ** 2 for item in self.items))


__all__ = ["Block"]
