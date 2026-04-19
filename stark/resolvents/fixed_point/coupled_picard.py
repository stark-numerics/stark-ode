from __future__ import annotations

"""Picard-backed resolvent for fully coupled implicit RK stage systems."""

from typing import TYPE_CHECKING

from stark.contracts import AcceleratorLike, Block, Derivative, Workbench
from stark.resolvents.base import ResolventBaseFixedPoint
from stark.resolvents.descriptor import ResolventDescriptor
from stark.resolvents.policy import ResolventPolicy
from stark.resolvents.support.stage_residual import CoupledStageResidual
from stark.execution.safety import Safety
from stark.execution.tolerance import Tolerance

if TYPE_CHECKING:
    from stark.schemes.tableau import ButcherTableau


class ResolventCoupledPicard(ResolventBaseFixedPoint):
    """Picard-driven resolvent for fully coupled implicit Runge-Kutta stages."""

    __slots__ = (
        "stage_count",
    )

    descriptor = ResolventDescriptor("Picard", "Picard Iteration")

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        tableau: ButcherTableau,
        tolerance: Tolerance | None = None,
        policy: ResolventPolicy | None = None,
        safety: Safety | None = None,
        accelerator: AcceleratorLike | None = None,
    ) -> None:
        self.stage_count = len(tableau.c)
        super().initialise_fixed_point(
            derivative,
            workbench,
            residual_factory=lambda workspace: CoupledStageResidual(
                "ResolventCoupledPicard",
                derivative,
                workspace,
                stage_shifts=tableau.c,
                matrix=tableau.a,
            ),
            tolerance=tolerance,
            policy=policy,
            safety=safety,
            accelerator=accelerator,
            tableau=tableau,
        )

    def check_block_sizes(self, out: Block, rhs: Block | None = None) -> None:
        if len(out) != self.stage_count:
            raise ValueError(f"out must be a {self.stage_count}-item block for this resolvent.")
        if rhs is not None and len(rhs) != self.stage_count:
            raise ValueError(f"rhs must be a {self.stage_count}-item block for this resolvent.")


__all__ = ["ResolventCoupledPicard"]












