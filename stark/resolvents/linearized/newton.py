from __future__ import annotations

"""Newton-backed resolvent for one-stage shifted implicit solves."""

from typing import Any

from stark.contracts import AcceleratorLike, Block, Derivative, InverterLike, Linearizer, Workbench
from stark.resolvents.base import ResolventBaseLinearized
from stark.resolvents.descriptor import ResolventDescriptor
from stark.resolvents.policy import ResolventPolicy
from stark.resolvents.support.stage_residual import StageResidual
from stark.execution.safety import Safety
from stark.execution.tolerance import Tolerance


class ResolventNewton(ResolventBaseLinearized):
    """Newton-driven resolvent for one-stage shifted implicit equations."""

    __slots__ = ()

    descriptor = ResolventDescriptor("Newton", "Newton Iteration")

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        linearizer: Linearizer,
        inverter: InverterLike,
        tolerance: Tolerance | None = None,
        policy: ResolventPolicy | None = None,
        safety: Safety | None = None,
        accelerator: AcceleratorLike | None = None,
        tableau: Any | None = None,
    ) -> None:
        super().initialise_linearized(
            derivative,
            workbench,
            linearizer,
            inverter,
            residual_factory=lambda workspace: StageResidual(
                "ResolventNewton",
                derivative,
                workspace,
                linearizer=linearizer,
            ),
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


__all__ = ["ResolventNewton"]












