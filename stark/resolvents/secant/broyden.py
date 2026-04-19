from __future__ import annotations

"""Broyden-backed resolvent for one-stage shifted implicit solves."""

from typing import Any

from stark.contracts import AcceleratorLike, Block, Derivative, InnerProduct, Workbench
from stark.resolvents.base import ResolventBaseSecant
from stark.resolvents.descriptor import ResolventDescriptor
from stark.resolvents.failure import ResolventError
from stark.resolvents.policy import ResolventPolicy
from stark.resolvents.support.stage_residual import StageResidual
from stark.execution.safety import Safety
from stark.execution.tolerance import Tolerance


class ResolventBroyden(ResolventBaseSecant):
    """Broyden-driven resolvent for one-stage shifted implicit equations."""

    __slots__ = (
        "next_residual",
        "correction",
        "trial",
        "residual_delta",
        "inverse_residual_delta",
        "scaled_update",
        "history_correction",
    )

    descriptor = ResolventDescriptor("Broyden", "Inverse Broyden")

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        inner_product: InnerProduct,
        tolerance: Tolerance | None = None,
        policy: ResolventPolicy | None = None,
        depth: int = 8,
        safety: Safety | None = None,
        accelerator: AcceleratorLike | None = None,
        tableau: Any | None = None,
    ) -> None:
        super().initialise_secant(
            derivative,
            workbench,
            inner_product,
            residual_factory=lambda workspace: StageResidual("ResolventBroyden", derivative, workspace),
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
        self.next_residual = self.resolvent_workspace.allocate_block(size)
        self.correction = self.resolvent_workspace.allocate_block(size)
        self.trial = self.resolvent_workspace.allocate_block(size)
        self.residual_delta = self.resolvent_workspace.allocate_block(size)
        self.inverse_residual_delta = self.resolvent_workspace.allocate_block(size)
        self.scaled_update = self.resolvent_workspace.allocate_block(size)
        self.history_correction = self.resolvent_workspace.allocate_block(size)
        self.history.ensure_size(size)

    def resolve(self, block: Block) -> None:
        if self.policy.max_iterations < 1:
            raise ValueError("ResolventPolicy.max_iterations must be at least 1.")

        self.prepare(len(block))
        residual_buffer = self.residual_buffer
        next_residual = self.next_residual
        correction = self.correction
        trial = self.trial
        residual_delta = self.residual_delta
        inverse_residual_delta = self.inverse_residual_delta
        scaled_update = self.scaled_update
        history_correction = self.history_correction
        assert residual_buffer is not None
        assert next_residual is not None
        assert correction is not None
        assert trial is not None
        assert residual_delta is not None
        assert inverse_residual_delta is not None
        assert scaled_update is not None
        assert history_correction is not None

        history = self.history
        workspace = history.workspace
        scale_block = workspace.scale_block
        combine2_block = workspace.combine2_block
        copy_block = workspace.copy_block
        inner_product = workspace.inner_product
        history.clear()

        for _ in range(self.policy.max_iterations):
            self.residual(residual_buffer, block)
            error = residual_buffer.norm()
            scale = block.norm()
            if self.tolerance.accepts(error, scale):
                return

            self.apply_inverse(correction, residual_buffer, history_correction)
            scale_block(correction, -1.0, correction)
            combine2_block(trial, 1.0, block, 1.0, correction)
            self.residual(next_residual, trial)

            combine2_block(residual_delta, 1.0, next_residual, -1.0, residual_buffer)
            denominator = inner_product(residual_delta, residual_delta)
            if denominator > 0.0:
                self.apply_inverse(inverse_residual_delta, residual_delta, history_correction)
                combine2_block(scaled_update, 1.0, correction, -1.0, inverse_residual_delta)
                scale_block(scaled_update, 1.0 / denominator, scaled_update)
                history.append(scaled_update, residual_delta)

            copy_block(block, trial)

        self.residual(residual_buffer, block)
        error = residual_buffer.norm()
        scale = block.norm()
        if self.tolerance.accepts(error, scale):
            return
        raise ResolventError(
            f"{self.short_name} failed to resolve the residual within "
            f"{self.policy.max_iterations} iterations (error={error:g})."
        )

    def apply_inverse(self, out: Block, block: Block, history_correction: Block) -> None:
        history = self.history
        workspace = history.workspace
        copy_block = workspace.copy_block
        combine2_block = workspace.combine2_block
        copy_block(out, block)
        if len(history) == 0:
            return
        coefficients = history.project_right(block)
        history.combine_left(history_correction, coefficients)
        combine2_block(out, 1.0, out, 1.0, history_correction)


__all__ = ["ResolventBroyden"]












