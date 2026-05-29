from __future__ import annotations

"""Newton-backed resolvent for fully coupled implicit RK stage systems."""

from typing import TYPE_CHECKING, Any, cast

from stark.block import Block, BlockAllocator, BlockOperator
from stark.contracts import (
    AcceleratorLike,
    LegacyInverterLike,
    Linearizer,
    Translation,
    Allocator,
)
from stark.executor.tolerance import ExecutorTolerance
from stark.resolvents.support import (
    MonitorResolventLike,
    ResolventCoupledStageProblem,
    ResolventCoupledStageResidual,
    ResolventError,
    ResolventPolicy,
    ResolventSafety,
    ResolventSpecialist,
    ResolventStencilBlock,
    initialise_resolvent_runtime,
    refresh_resolvent_call,
    with_resolvent_call_methods,
    with_resolvent_display,
    with_resolvent_monitoring,
)
from stark.resolvents.support.descriptor import ResolventDescriptor
from stark.resolvents.support.tolerance import ResolventTolerance


@with_resolvent_display
@with_resolvent_call_methods
@with_resolvent_monitoring
class ResolventCoupledNewton:
    """Newton iteration for coupled implicit RK stage systems.

    Algorithm sketch:

        1. Compute the coupled residual F(delta).
        2. Accept if ||F(delta)|| is within ExecutorTolerance.
        3. Build the coupled differential DF(delta).
        4. Solve DF(delta) correction = -F(delta).
        5. Apply delta <- delta + correction.
        6. Recheck once after the final correction.
    """

    __slots__ = (
        "_monitor",
        "accelerator",
        "alpha",
        "allocator",
        "call_pure",
        "correction",
        "inverter",
        "newton_update",
        "operator",
        "policy",
        "redirect_call",
        "residual",
        "residual_buffer",
        "rhs_buffer",
        "safety",
        "size",
        "tableau",
        "tolerance",
    )

    descriptor = ResolventDescriptor("Newton", "Newton Iteration")

    if TYPE_CHECKING:
        def assign_monitor(self, monitor: MonitorResolventLike) -> None: ...
        def unassign_monitor(self) -> None: ...
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
        allocator: Allocator,
        linearizer: Linearizer,
        inverter: LegacyInverterLike,
        ExecutorTolerance: ExecutorTolerance | None = None,
        policy: ResolventPolicy | None = None,
        safety: ResolventSafety | None = None,
        accelerator: AcceleratorLike | None = None,
        specialist: ResolventSpecialist[Translation] | None = None,
        tableau: Any | None = None,
    ) -> None:
        self.tableau = tableau
        initialise_resolvent_runtime(self, safety, accelerator)

        self.allocator = BlockAllocator(allocator)
        self.tolerance = (
            ExecutorTolerance
            if ExecutorTolerance is not None
            else ResolventTolerance(atol=1.0e-9, rtol=1.0e-9)
        )
        self.policy = policy if policy is not None else ResolventPolicy()
        self.inverter = inverter
        self.residual = ResolventCoupledStageResidual(
            "ResolventCoupledNewton",
            allocator,
            linearizer=linearizer,
            accelerator=self.accelerator,
        )
        self.correction = None
        self.residual_buffer = None
        self.rhs_buffer = None
        self.operator = None
        self.newton_update = None
        self.size = -1

        if specialist is not None:
            self.prepare_specialized_kernels(specialist)
            self.call_pure = self.call_specialized
            refresh_resolvent_call(self)

    def prepare_specialized_kernels(
        self,
        specialist: ResolventSpecialist[Translation],
    ) -> None:
        # Step 5: Newton update delta <- delta + correction.
        self.newton_update = specialist.provide(
            ResolventStencilBlock((1.0, 1.0))
        )

    def prepare_buffers(self, delta: Block[Translation]) -> None:
        size = len(delta)
        if self.size == size:
            return

        self.size = size
        self.correction = self.allocator.allocate_like(delta)
        self.residual_buffer = self.allocator.allocate_like(delta)
        self.rhs_buffer = self.allocator.allocate_like(delta)
        self.operator = BlockOperator(None for _ in range(size))

    def operator_for(self, delta: Block[Translation]) -> BlockOperator[Translation]:
        """Return the operator object that will receive the coupled Jacobian.

        A coupled stage residual owns a block operator when the Jacobian action
        is naturally expressed as one coupled matrix action. Other residual
        workers can fall back to the ordinary per-stage BlockOperator buffer.
        """

        residual_owned_operator = self.residual.block_operator
        if residual_owned_operator is not None:
            return residual_owned_operator

        self.prepare_buffers(delta)
        return cast(BlockOperator[Translation], self.operator)

    def call_inline(
        self,
        problem: ResolventCoupledStageProblem,
        delta: Block[Translation],
    ) -> Block[Translation]:
        if self.policy.max_iterations < 1:
            raise ValueError("ResolventPolicy.max_iterations must be at least 1.")

        self.alpha = problem.step
        self.prepare_buffers(delta)

        F = self.residual
        F.configure(problem)
        correction = cast(Block[Translation], self.correction)
        residual = cast(Block[Translation], self.residual_buffer)
        rhs = cast(Block[Translation], self.rhs_buffer)

        block_size = len(delta)
        iteration_count = 0

        for _ in range(self.policy.max_iterations):
            # 1. Compute the coupled residual F(delta).
            F(delta, residual)

            # 2. Accept if ||F(delta)|| is within ExecutorTolerance.
            error = residual.norm()
            scale = delta.norm()
            if self.tolerance.accepts(error, scale):
                self.record_solve(block_size, iteration_count, error, scale, True)
                return delta

            # 3. Build the coupled differential DF(delta).
            operator = self.operator_for(delta)
            operator.reset()
            F.differential(delta, operator)

            # 4. Solve DF(delta) correction = -F(delta).
            rhs.replace(-1.0 * residual)
            correction.replace(0.0 * correction)
            self.inverter.bind(operator)
            self.inverter(rhs, correction)

            # 5. Newton update: delta <- delta + correction.
            delta += correction
            iteration_count += 1

        # 6. Recheck once after the final correction.
        F(delta, residual)

        error = residual.norm()
        scale = delta.norm()
        if self.tolerance.accepts(error, scale):
            self.record_solve(block_size, iteration_count, error, scale, True)
            return delta

        self.record_solve(block_size, iteration_count, error, scale, False)
        raise ResolventError(
            f"{type(self).__name__} failed to resolve the residual within "
            f"{self.policy.max_iterations} iterations (error={error:g})."
        )

    def call_specialized(
        self,
        problem: ResolventCoupledStageProblem,
        delta: Block[Translation],
    ) -> Block[Translation]:
        if self.policy.max_iterations < 1:
            raise ValueError("ResolventPolicy.max_iterations must be at least 1.")

        self.alpha = problem.step
        self.prepare_buffers(delta)

        F = self.residual
        F.configure(problem)
        correction = cast(Block[Translation], self.correction)
        residual = cast(Block[Translation], self.residual_buffer)
        rhs = cast(Block[Translation], self.rhs_buffer)
        newton_update = self.newton_update
        assert newton_update is not None

        block_size = len(delta)
        iteration_count = 0

        for _ in range(self.policy.max_iterations):
            # 1. Compute the coupled residual F(delta).
            F(delta, residual)

            # 2. Accept if ||F(delta)|| is within ExecutorTolerance.
            error = residual.norm()
            scale = delta.norm()
            if self.tolerance.accepts(error, scale):
                self.record_solve(block_size, iteration_count, error, scale, True)
                return delta

            # 3. Build the coupled differential DF(delta).
            operator = self.operator_for(delta)
            operator.reset()
            F.differential(delta, operator)

            # 4. Solve DF(delta) correction = -F(delta).
            rhs.replace(-1.0 * residual)
            correction.replace(0.0 * correction)
            self.inverter.bind(operator)
            self.inverter(rhs, correction)

            # 5. Newton update: delta <- delta + correction.
            newton_update(1.0, delta, delta, correction)
            iteration_count += 1

        # 6. Recheck once after the final correction.
        F(delta, residual)

        error = residual.norm()
        scale = delta.norm()
        if self.tolerance.accepts(error, scale):
            self.record_solve(block_size, iteration_count, error, scale, True)
            return delta

        self.record_solve(block_size, iteration_count, error, scale, False)
        raise ResolventError(
            f"{type(self).__name__} failed to resolve the residual within "
            f"{self.policy.max_iterations} iterations (error={error:g})."
        )


__all__ = ["ResolventCoupledNewton"]
