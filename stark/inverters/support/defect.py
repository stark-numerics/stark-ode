from __future__ import annotations

from dataclasses import dataclass
from typing import Generic

from stark.contracts import BlockLike, InverterRequest, TranslationType


@dataclass(slots=True, init=False)
class InverterDefect(Generic[TranslationType]):
    """
    Reusable defect block for an inverter request.

    For the linear problem

        request.operator(output) = request.residual

    the defect is

        request.residual - request.operator(output)

    Calling the worker stores the current defect in `block` and returns its
    norm. Algorithms may then use `block` directly without recomputing the
    operator image.
    """

    image: BlockLike[TranslationType] | None
    block: BlockLike[TranslationType] | None
    size: int

    def __init__(self) -> None:
        self.image = None
        self.block = None
        self.size = -1

    def prepare(self, output: BlockLike[TranslationType]) -> None:
        """Allocate scratch blocks matching the current output size."""

        size = len(output)
        if self.size == size:
            return

        self.image = 0.0 * output  # type: ignore[operator]
        self.block = 0.0 * output  # type: ignore[operator]
        self.size = size

    def __call__(
        self,
        request: InverterRequest[TranslationType],
        output: BlockLike[TranslationType],
    ) -> float:
        """Store the current defect and return its norm."""

        self.prepare(output)

        image = self.image
        block = self.block
        assert image is not None
        assert block is not None

        request.operator(output, image)
        block.replace(request.residual - image)  # type: ignore[operator]
        return block.norm()


__all__ = ["InverterDefect"]
