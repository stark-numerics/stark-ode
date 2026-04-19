from __future__ import annotations

"""Anderson-backed resolvent for one-stage shifted implicit solves."""

from typing import Any

from stark.contracts import AcceleratorLike, Block, Derivative, InnerProduct, Workbench
from stark.resolvents.base import ResolventBaseSecant
from stark.resolvents.descriptor import ResolventDescriptor
from stark.resolvents.failure import ResolventError
from stark.resolvents.policy import ResolventPolicy
from stark.resolvents.support.stage_residual import StageResidual
from stark.execution.safety import Safety
from stark.execution.tolerance import Tolerance


class ResolventAnderson(ResolventBaseSecant):
    """Anderson-driven resolvent for one-stage shifted implicit equations."""

    __slots__ = ("previous_residual", "fixed_point", "previous_fixed_point", "correction")

    descriptor = ResolventDescriptor("Anderson", "Anderson Acceleration")

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        inner_product: InnerProduct,
        tolerance: Tolerance | None = None,
        policy: ResolventPolicy | None = None,
        depth: int = 4,
        safety: Safety | None = None,
        accelerator: AcceleratorLike | None = None,
        tableau: Any | None = None,
    ) -> None:
        super().initialise_secant(
            derivative,
            workbench,
            inner_product,
            residual_factory=lambda workspace: StageResidual("ResolventAnderson", derivative, workspace),
            tolerance=tolerance,
            policy=policy,
            depth=depth,
            safety=safety,
            accelerator=accelerator,
            tableau=tableau,
        )

    def prepare(self, size: int) -> None:
        if self.size == size:
            return
        self.size = size
        self.residual_buffer = self.resolvent_workspace.allocate_block(size)
        self.previous_residual = self.resolvent_workspace.allocate_block(size)
        self.fixed_point = self.resolvent_workspace.allocate_block(size)
        self.previous_fixed_point = self.resolvent_workspace.allocate_block(size)
        self.correction = self.resolvent_workspace.allocate_block(size)
        self.history.ensure_size(size)

    def resolve(self, block: Block) -> None:
        if self.policy.max_iterations < 1:
            raise ValueError("ResolventPolicy.max_iterations must be at least 1.")

        self.prepare(len(block))
        residual_buffer = self.residual_buffer
        previous_residual = self.previous_residual
        fixed_point = self.fixed_point
        previous_fixed_point = self.previous_fixed_point
        correction = self.correction
        assert residual_buffer is not None
        assert previous_residual is not None
        assert fixed_point is not None
        assert previous_fixed_point is not None
        assert correction is not None

        history = self.history
        workspace = history.workspace
        combine2_block = workspace.combine2_block
        copy_block = workspace.copy_block
        history.clear()
        have_previous = False

        for _ in range(self.policy.max_iterations):
            self.residual(residual_buffer, block)
            error = residual_buffer.norm()
            scale = block.norm()
            if self.tolerance.accepts(error, scale):
                return

            combine2_block(fixed_point, 1.0, block, -1.0, residual_buffer)
            if have_previous:
                history.append_difference(fixed_point, previous_fixed_point, residual_buffer, previous_residual)

            if len(history) > 0:
                coefficients = history.solve_right_least_squares(residual_buffer)
                history.combine_left(correction, coefficients)
                combine2_block(block, 1.0, fixed_point, -1.0, correction)
            else:
                copy_block(block, fixed_point)

            copy_block(previous_fixed_point, fixed_point)
            copy_block(previous_residual, residual_buffer)
            have_previous = True

        self.residual(residual_buffer, block)
        error = residual_buffer.norm()
        scale = block.norm()
        if self.tolerance.accepts(error, scale):
            return
        raise ResolventError(
            f"{self.short_name} failed to resolve the residual within "
            f"{self.policy.max_iterations} iterations (error={error:g})."
        )


__all__ = ["ResolventAnderson"]












