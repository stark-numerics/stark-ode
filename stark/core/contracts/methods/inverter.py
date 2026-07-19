"""Contracts for linear inverter and preconditioner workers."""

from __future__ import annotations

from enum import Enum
from typing import ClassVar, Protocol

from stark.core.contracts.methods.block import BlockLike, BlockOperatorLike
from stark.core.contracts.problem.translation import TranslationType


class InverterOutputMode(Enum):
    """
    Contract for how an inverter treats the supplied output block.

    overwrite:
        The inverter computes a fresh solution and replaces output contents.

    improve:
        The inverter treats output as the current guess and improves it in place.
    """

    overwrite = "overwrite"
    improve = "improve"


class InverterRequest(Protocol[TranslationType]):
    """
    Structural contract for a linear solve request consumed by an inverter.

    An inverter request represents the block-valued equation

        operator(solution) = residual

    The request exposes the linear block action to invert and the right-hand
    side block. Calling the operator as `operator(output, image)` should write
    the image of `output` into `image`.
    """

    @property
    def operator(self) -> BlockOperatorLike[TranslationType]:
        ...

    @property
    def residual(self) -> BlockLike[TranslationType]:
        ...


class Inverter(Protocol[TranslationType]):
    """
    Linear problem solver used by resolvents.

    An inverter receives a linear problem request and improves `output` in
    place so that

        request.operator(output) approximately equals request.residual
    """
    output_mode: ClassVar[InverterOutputMode]

    def __call__(
        self,
        request: InverterRequest[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        ...


class InverterInstance(Protocol[TranslationType]):
    """
    Block-operator-bound linear solve action.

    Some resolvents reuse the same linearized operator for several right-hand
    sides. An inverter can expose `instance(operator)` to do operator-specific
    preparation once and return a callable that solves

        operator(output) = residual

    for each later residual block.
    """

    def __call__(
        self,
        residual: BlockLike[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        ...


class InverterInstancing(Protocol[TranslationType]):
    """
    Optional inverter capability for operator-bound solve instances."""

    def instance(
        self,
        operator: BlockOperatorLike[TranslationType],
    ) -> InverterInstance[TranslationType]:
        ...


__all__ = [
    "InverterOutputMode",
    "InverterRequest",
    "Inverter",
    "InverterInstance",
    "InverterInstancing",
]
