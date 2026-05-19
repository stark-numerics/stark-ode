from __future__ import annotations

"""Anderson-backed resolvent for one-stage shifted implicit solves."""

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
class ResolventAnderson:
    """
    Anderson-accelerated resolvent for one-stage shifted implicit equations.

    Anderson acceleration starts from the same fixed-point map as Picard, then
    uses a rolling history of fixed-point and residual differences to project a
    better next guess.

    Further reading:
    https://en.wikipedia.org/wiki/Anderson_acceleration
    """

    __slots__ = (
        "_monitor",
        "alpha",
        "accelerator",
        "correction",
        "depth",
        "fixed_point",
        "history",
        "interval",
        "policy",
        "previous_fixed_point",
        "previous_residual",
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

    descriptor = ResolventDescriptor("Anderson", "Anderson Acceleration")

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
        depth: int = 4,
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
        self.residual = ResolventStageResidual("ResolventAnderson", derivative, self.scheme_workspace)
        self.residual_buffer = None
        self.previous_residual = None
        self.fixed_point = None
        self.previous_fixed_point = None
        self.correction = None
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
        self.previous_residual = self.resolvent_workspace.allocate_block(size)
        self.fixed_point = self.resolvent_workspace.allocate_block(size)
        self.previous_fixed_point = self.resolvent_workspace.allocate_block(size)
        self.correction = self.resolvent_workspace.allocate_block(size)
        self.history.ensure_size(size)

    def resolve(self, block: Block) -> None:
        if self.policy.max_iterations < 1:
            raise ValueError("ResolventPolicy.max_iterations must be at least 1.")

        self.prepare(len(block))
        residual_buffer = cast(Block, self.residual_buffer)
        previous_residual = cast(Block, self.previous_residual)
        fixed_point = cast(Block, self.fixed_point)
        previous_fixed_point = cast(Block, self.previous_fixed_point)
        correction = cast(Block, self.correction)

        history = self.history
        workspace = history.workspace
        combine2_block = workspace.combine2_block
        copy_block = workspace.copy_block
        history.clear()
        have_previous = False
        block_size = len(block)
        iteration_count = 0

        for _ in range(self.policy.max_iterations):
            self.residual(block, residual_buffer)
            error = residual_buffer.norm()
            scale = block.norm()
            if self.tolerance.accepts(error, scale):
                self.record_solve(block_size, iteration_count, error, scale, True)
                return

            # Build the plain Picard fixed-point candidate first; Anderson only
            # changes how that candidate is mixed with recent history.
            combine2_block(1.0, block, -1.0, residual_buffer, fixed_point)
            if have_previous:
                history.append_difference(fixed_point, previous_fixed_point, residual_buffer, previous_residual)

            if len(history) > 0:
                # Solve the small history least-squares problem and subtract
                # the projected correction from the fixed-point candidate.
                coefficients = history.solve_right_least_squares(residual_buffer)
                history.combine_left(correction, coefficients)
                combine2_block(1.0, fixed_point, -1.0, correction, block)
            else:
                copy_block(block, fixed_point)

            copy_block(previous_fixed_point, fixed_point)
            copy_block(previous_residual, residual_buffer)
            have_previous = True
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


__all__ = ["ResolventAnderson"]
