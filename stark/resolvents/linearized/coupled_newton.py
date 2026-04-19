from __future__ import annotations

"""Newton-backed resolvent for fully coupled implicit RK stage systems."""

from stark.schemes.tableau import ButcherTableau
from stark.contracts import AcceleratorLike, Block, Derivative, InverterLike, Linearizer, Workbench
from stark.resolvents.base import ResolventBaseLinearized
from stark.resolvents.descriptor import ResolventDescriptor
from stark.resolvents.policy import ResolventPolicy
from stark.resolvents.support.stage_residual import CoupledStageResidual
from stark.execution.safety import Safety
from stark.execution.tolerance import Tolerance


class ResolventCoupledNewton(ResolventBaseLinearized):
    """Newton-driven resolvent for fully coupled implicit Runge-Kutta stages."""

    __slots__ = (
        "stage_count",
    )

    descriptor = ResolventDescriptor("Newton", "Newton Iteration")

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        tableau: ButcherTableau,
        linearizer: Linearizer,
        inverter: InverterLike,
        tolerance: Tolerance | None = None,
        policy: ResolventPolicy | None = None,
        safety: Safety | None = None,
        accelerator: AcceleratorLike | None = None,
    ) -> None:
        self.stage_count = len(tableau.c)
        super().initialise_linearized(
            derivative,
            workbench,
            linearizer,
            inverter,
            residual_factory=lambda workspace: CoupledStageResidual(
                "ResolventCoupledNewton",
                derivative,
                workspace,
                stage_shifts=tableau.c,
                matrix=tableau.a,
                linearizer=linearizer,
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


__all__ = ["ResolventCoupledNewton"]












