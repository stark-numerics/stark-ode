from __future__ import annotations

"""Picard-backed resolvent for one-stage shifted implicit solves."""

from typing import TYPE_CHECKING, Any, cast

from stark.auditor import Auditor
from stark.contracts import AcceleratorLike, Block, Derivative, IntervalLike, State, Workbench
from stark.execution.safety import Safety
from stark.execution.tolerance import Tolerance
from stark.machinery.stage_solve.workspace import SchemeWorkspace
from stark.resolvents.support.descriptor import ResolventDescriptor
from stark.resolvents.support.failure import ResolventError
from stark.resolvents.support.policy import ResolventPolicy
from stark.resolvents.support import (
    MonitorResolventLike,
    ResolventStageResidual,
    check_one_stage_block,
    initialise_resolvent_runtime,
    with_resolvent_binding_methods,
    with_resolvent_display_methods,
    with_resolvent_monitoring_methods,
)
from stark.resolvents.support.workspace import ResolventWorkspace
from stark.resolvents.support.tolerance import ResolventTolerance


@with_resolvent_display_methods
@with_resolvent_binding_methods
@with_resolvent_monitoring_methods
class ResolventPicard:
    """
    Picard-driven resolvent for one-stage shifted implicit equations.

    This solves the one-stage nonlinear residual

        delta - rhs - alpha * f(t, state + delta) = 0

    by fixed-point iteration. Starting from a zero stage increment, each Picard
    iteration evaluates the residual and applies the simple correction

        delta <- delta - residual(delta)

    which is equivalent to iterating

        delta <- rhs + alpha * f(t, state + delta).

    Picard iteration is deliberately simple: it is easy to customize and cheap
    per iteration, but less robust than Newton or secant-family resolvents when
    the implicit problem is strongly nonlinear or the step is large.

    Further reading:
    https://en.wikipedia.org/wiki/Fixed-point_iteration
    https://en.wikipedia.org/wiki/Picard%E2%80%93Lindel%C3%B6f_theorem
    """

    __slots__ = (
        "_monitor",
        "alpha",
        "accelerator",
        "interval",
        "policy",
        "redirect_call",
        "residual",
        "residual_buffer",
        "resolvent_workspace",
        "safety",
        "scheme_workspace",
        "size",
        "state",
        "tableau",
        "tolerance",
    )

    descriptor = ResolventDescriptor("Picard", "Picard Iteration")

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
        tolerance: Tolerance | None = None,
        policy: ResolventPolicy | None = None,
        safety: Safety | None = None,
        accelerator: AcceleratorLike | None = None,
        tableau: Any | None = None,
    ) -> None:
        translation_probe = workbench.allocate_translation()
        Auditor.require_scheme_inputs(derivative, workbench, translation_probe)

        self.tableau = tableau
        initialise_resolvent_runtime(self, safety, accelerator)

        self.scheme_workspace = SchemeWorkspace(workbench, translation_probe)
        self.resolvent_workspace = ResolventWorkspace(
            workbench,
            translation_probe,
            self.safety,
            accelerator=self.accelerator,
        )
        self.tolerance = tolerance if tolerance is not None else ResolventTolerance(atol=1.0e-9, rtol=1.0e-9)
        self.policy = policy if policy is not None else ResolventPolicy()
        self.residual = ResolventStageResidual("ResolventPicard", derivative, self.scheme_workspace)
        self.residual_buffer = None
        self.size = -1

    def check_block_sizes(self, out: Block, rhs: Block | None = None) -> None:
        check_one_stage_block("out", out)
        if rhs is not None:
            check_one_stage_block("rhs", rhs)

    def call_checked(self, alpha: float, rhs: Block | None, out: Block) -> None:
        self.check_block_sizes(out, rhs)
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

    def resolve(self, block: Block) -> None:
        if self.policy.max_iterations < 1:
            raise ValueError("ResolventPolicy.max_iterations must be at least 1.")

        self.prepare(len(block))
        residual_buffer = cast(Block, self.residual_buffer)

        # Picard iterates by measuring the residual, then subtracting it from
        # the current stage increment to form the next fixed-point guess.
        block_size = len(block)
        iteration_count = 0

        for _ in range(self.policy.max_iterations):
            self.residual(block, residual_buffer)
            error = residual_buffer.norm()
            scale = block.norm()
            if self.tolerance.accepts(error, scale):
                self.record_solve(block_size, iteration_count, error, scale, True)
                return
            self.resolvent_workspace.combine2_block(1.0, block, -1.0, residual_buffer, block)
            iteration_count += 1

        # Recheck once after the final update so the last Picard correction can
        # be accepted without requiring an extra user-visible iteration.
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


__all__ = ["ResolventPicard"]


