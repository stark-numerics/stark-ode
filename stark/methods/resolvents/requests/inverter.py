from __future__ import annotations

from dataclasses import dataclass
from typing import Generic

from stark.core.contracts.block import BlockLike, BlockOperatorLike
from stark.core.contracts.translation import TranslationType


@dataclass(slots=True)
class ResolventInverterRequest(Generic[TranslationType]):
    """
    Linear correction request emitted by a resolvent.

    Represents the linear equation

        operator(solution) = residual

    produced during a nonlinear or implicit solve. The operator is the
    linearized residual action assembled at the current trial state. The
    residual is the right-hand side requested by the resolvent, typically the
    negated nonlinear residual for Newton-style correction equations.
    """

    operator: BlockOperatorLike[TranslationType]
    residual: BlockLike[TranslationType]


__all__ = ["ResolventInverterRequest"]
