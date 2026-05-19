from __future__ import annotations

"""Broyden-backed resolvent for one-stage shifted implicit solves."""

from typing import TYPE_CHECKING, Any, cast

from stark.auditor import Auditor
from stark.contracts import AcceleratorLike, Block, Derivative, InnerProduct, IntervalLike, State, Workbench
from stark.execution.safety import Safety
from stark.execution.tolerance import Tolerance
from stark.machinery.stage_solve.workspace import SchemeWorkspace
from stark.resolvents.descriptor import ResolventDescriptor
from stark.resolvents.failure import ResolventError
from stark.resolvents.policy import ResolventPolicy
from stark.resolvents.support import (
    MonitorResolventLike,
    ResolventStageResidual,
    SecantHistory,
    check_one_stage_block,
    initialise_resolvent_runtime,
    with_resolvent_binding_methods,
    with_resolvent_display_methods,
    with_resolvent_monitoring_methods,
)
from stark.resolvents.support.workspace import ResolventWorkspace
from stark.resolvents.tolerance import ResolventTolerance


@with_resolvent_display_methods
@with_resolvent_binding_methods
@with_resolvent_monitoring_methods
class ResolventBroyden:
    """
    Inverse-Broyden resolvent for one-stage shifted implicit equations.

    The method updates an approximate inverse residual Jacobian from recent
    secant pairs, then applies that inverse approximation to produce the next
    nonlinear correction.

    Further reading:
    https://en.wikipedia.org/wiki/Broyden%27s_method
    """

    __slots__ = (
        "_monitor",
        "alpha",
        "accelerator",
        "correction",
        "depth",
        "history",
        "history_correction",
        "interval",
        "inverse_residual_delta",
        "next_residual",
        "policy",
        "redirect_call",
        "residual",
        "residual_buffer",
        "residual_delta",
        "resolvent_workspace",
        "safety",
        "scaled_update",
        "scheme_workspace",
        "size",
        "state",
        "tableau",
        "tolerance",
        "trial",
    )

    descriptor = ResolventDescriptor("Broyden", "Inverse Broyden")

    if TYPE_CHECKING:
        def bind(self, interval: IntervalLike, state: State) -> None: ...

        def bind_accelerator(self, accelerator: AcceleratorLike) -> None: ...

        def assign_monitor(self, monitor: MonitorResolventLike) -> None: ...

        def unassign_monitor(self) -> None: ...

        def call_unbound(self, alpha: float, rhs: Block | None, out: Block) -> None: ...

        def record_solve(
            self,
            block_size: int,
            iteration_count: int,
            error: float,
            scale: float,
            converged: bool,
        ) -> None: ...

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
        translation_probe = workbench.allocate_translation()
        Auditor.require_scheme_inputs(derivative, workbench, translation_probe)

        self.tableau = tableau
        initialise_resolvent_runtime(self, safety, accelerator)

        self.scheme_workspace = SchemeWorkspace(workbench, translation_probe)
        self.tolerance = tolerance if tolerance is not None else ResolventTolerance(atol=1.0e-9, rtol=1.0e-9)
        self.policy = policy if policy is not None else ResolventPolicy()
        self.depth = depth
        self.resolvent_workspace = ResolventWorkspace(
            workbench,
            translation_probe,
            self.safety,
            inner_product=inner_product,
            accelerator=self.accelerator,
        )
        self.history = SecantHistory(self.resolvent_workspace, depth, accelerator=self.accelerator)
        self.residual = ResolventStageResidual("ResolventBroyden", derivative, self.scheme_workspace)
        self.residual_buffer = None
        self.next_residual = None
        self.correction = None
        self.trial = None
        self.residual_delta = None
        self.inverse_residual_delta = None
        self.scaled_update = None
        self.history_correction = None
        self.size = -1

    def call_checked(self, alpha: float, rhs: Block | None, out: Block) -> None:
        check_one_stage_block("out", out)
        if rhs is not None:
            check_one_stage_block("rhs", rhs)
        self.call_unchecked(alpha, rhs, out)

    def call_unchecked(self, alpha: float, rhs: Block | None, out: Block) -> None:
        interval = cast(IntervalLike, self.interval)
        state = cast(State, self.state)

        self.alpha = alpha
        self.residual.configure(interval, state, alpha, rhs=rhs)
        self.resolvent_workspace.zero_block(out)
        self.resolve(out)

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
        residual_buffer = cast(Block, self.residual_buffer)
        next_residual = cast(Block, self.next_residual)
        correction = cast(Block, self.correction)
        trial = cast(Block, self.trial)
        residual_delta = cast(Block, self.residual_delta)
        inverse_residual_delta = cast(Block, self.inverse_residual_delta)
        scaled_update = cast(Block, self.scaled_update)
        history_correction = cast(Block, self.history_correction)

        history = self.history
        workspace = history.workspace
        scale_block = workspace.scale_block
        combine2_block = workspace.combine2_block
        copy_block = workspace.copy_block
        inner_product = workspace.inner_product
        history.clear()
        block_size = len(block)
        iteration_count = 0

        for _ in range(self.policy.max_iterations):
            self.residual(block, residual_buffer)
            error = residual_buffer.norm()
            scale = block.norm()
            if self.tolerance.accepts(error, scale):
                self.record_solve(block_size, iteration_count, error, scale, True)
                return

            # Apply the current inverse-Jacobian approximation to the residual
            # to obtain a nonlinear correction.
            self.apply_inverse(residual_buffer, history_correction, correction)
            scale_block(-1.0, correction, correction)
            combine2_block(1.0, block, 1.0, correction, trial)
            self.residual(trial, next_residual)

            # Use the trial residual change to add one inverse-Broyden secant
            # pair to the rolling history.
            combine2_block(1.0, next_residual, -1.0, residual_buffer, residual_delta)
            denominator = inner_product(residual_delta, residual_delta)
            if denominator > 0.0:
                self.apply_inverse(residual_delta, history_correction, inverse_residual_delta)
                combine2_block(1.0, correction, -1.0, inverse_residual_delta, scaled_update)
                scale_block(1.0 / denominator, scaled_update, scaled_update)
                history.append(scaled_update, residual_delta)

            copy_block(block, trial)
            iteration_count += 1

        self.residual(block, residual_buffer)
        error = residual_buffer.norm()
        scale = block.norm()
        if self.tolerance.accepts(error, scale):
            self.record_solve(block_size, iteration_count, error, scale, True)
            return
        self.record_solve(block_size, iteration_count, error, scale, False)
        raise ResolventError(
            f"{type(self).__name__} failed to resolve the residual within "
            f"{self.policy.max_iterations} iterations (error={error:g})."
        )

    def apply_inverse(self, block: Block, history_correction: Block, out: Block) -> None:
        history = self.history
        workspace = history.workspace
        copy_block = workspace.copy_block
        combine2_block = workspace.combine2_block
        copy_block(out, block)
        if len(history) == 0:
            return
        coefficients = history.project_right(block)
        history.combine_left(history_correction, coefficients)
        combine2_block(1.0, out, 1.0, history_correction, out)


__all__ = ["ResolventBroyden"]
