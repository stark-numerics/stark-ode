from __future__ import annotations

"""Picard-backed resolvent for one-stage shifted implicit solves."""

from typing import Any

from stark.contracts import AcceleratorLike, Block, Derivative, Workbench
from stark.resolvents.base import ResolventBaseFixedPoint
from stark.resolvents.descriptor import ResolventDescriptor
from stark.resolvents.policy import ResolventPolicy
from stark.resolvents.support.stage_residual import StageResidual
from stark.execution.safety import Safety
from stark.execution.tolerance import Tolerance


class ResolventPicard(ResolventBaseFixedPoint):
    """Picard-driven resolvent for one-stage shifted implicit equations."""

    __slots__ = ()

    descriptor = ResolventDescriptor("Picard", "Picard Iteration")

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        tolerance: Tolerance | None = None,
        policy: ResolventPolicy | None = None,
        safety: Safety | None = None,
        accelerator: AcceleratorLike | None = None,
        tableau: Any | None = None,
    ) -> None:
        super().initialise_fixed_point(
            derivative,
            workbench,
            residual_factory=lambda workspace: StageResidual("ResolventPicard", derivative, workspace),
            tolerance=tolerance,
            policy=policy,
            safety=safety,
            accelerator=accelerator,
            tableau=tableau,
        )

    def check_block_sizes(self, out: Block, rhs: Block | None = None) -> None:
        self.check_one_stage_block("out", out)
        if rhs is not None:
            self.check_one_stage_block("rhs", rhs)


__all__ = ["ResolventPicard"]












