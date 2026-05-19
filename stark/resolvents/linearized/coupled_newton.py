from __future__ import annotations

"""Newton-backed resolvent for fully coupled implicit RK stage systems."""

from typing import TYPE_CHECKING, cast

from stark.auditor import Auditor
from stark.block.operator import BlockOperator
from stark.contracts import AcceleratorLike, Block, Derivative, IntervalLike, InverterLike, Linearizer, State, Workbench
from stark.execution.safety import Safety
from stark.execution.tolerance import Tolerance
from stark.machinery.stage_solve.workspace import SchemeWorkspace
from stark.resolvents.descriptor import ResolventDescriptor
from stark.resolvents.failure import ResolventError
from stark.resolvents.policy import ResolventPolicy
from stark.resolvents.support import (
    MonitorResolventLike,
    ResolventCoupledStageResidual,
    initialise_resolvent_runtime,
    with_resolvent_binding_methods,
    with_resolvent_display_methods,
    with_resolvent_monitoring_methods,
)
from stark.resolvents.support.workspace import ResolventWorkspace
from stark.resolvents.tolerance import ResolventTolerance
from stark.schemes.tableau import ButcherTableau


@with_resolvent_display_methods
@with_resolvent_binding_methods
@with_resolvent_monitoring_methods
class ResolventCoupledNewton:
    """
    Newton-driven resolvent for fully coupled implicit Runge-Kutta stages.

    The residual and linearized operator are block-valued: each Newton update
    solves for all stage increments at once. This is the natural fully coupled
    form for collocation and other implicit RK methods with dense tableau rows.

    Further reading:
    https://en.wikipedia.org/wiki/Newton%27s_method
    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    """

    __slots__ = (
        "_monitor",
        "alpha",
        "accelerator",
        "correction",
        "interval",
        "inverter",
        "operator",
        "policy",
        "redirect_call",
        "residual",
        "residual_buffer",
        "resolvent_workspace",
        "rhs_buffer",
        "safety",
        "scheme_workspace",
        "size",
        "stage_count",
        "state",
        "tableau",
        "tolerance",
    )

    descriptor = ResolventDescriptor("Newton", "Newton Iteration")

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
        tableau: ButcherTableau,
        linearizer: Linearizer,
        inverter: InverterLike,
        tolerance: Tolerance | None = None,
        policy: ResolventPolicy | None = None,
        safety: Safety | None = None,
        accelerator: AcceleratorLike | None = None,
    ) -> None:
        translation_probe = workbench.allocate_translation()
        Auditor.require_scheme_inputs(derivative, workbench, translation_probe)
        Auditor.require_linearizer_inputs(linearizer, workbench, translation_probe)

        self.stage_count = len(tableau.c)
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
        self.inverter = inverter
        self.residual = ResolventCoupledStageResidual(
            "ResolventCoupledNewton",
            derivative,
            self.scheme_workspace,
            stage_shifts=tableau.c,
            matrix=tableau.a,
            linearizer=linearizer,
        )
        self.correction = None
        self.residual_buffer = None
        self.rhs_buffer = None
        self.operator = None
        self.size = -1

    def check_block_sizes(self, out: Block, rhs: Block | None = None) -> None:
        if len(out) != self.stage_count:
            raise ValueError(f"out must be a {self.stage_count}-item block for this resolvent.")
        if rhs is not None and len(rhs) != self.stage_count:
            raise ValueError(f"rhs must be a {self.stage_count}-item block for this resolvent.")

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
        self.correction = self.resolvent_workspace.allocate_block(size)
        self.residual_buffer = self.resolvent_workspace.allocate_block(size)
        self.rhs_buffer = self.resolvent_workspace.allocate_block(size)
        self.operator = BlockOperator([None for _ in range(size)], check_sizes=self.safety.block_sizes)  # type: ignore[list-item]

    def resolve_operator(self, size: int):
        custom_operator = getattr(self.residual, "block_operator", None)
        if custom_operator is not None:
            return custom_operator
        self.prepare(size)
        return cast(BlockOperator, self.operator)

    def resolve(self, block: Block) -> None:
        if self.policy.max_iterations < 1:
            raise ValueError("ResolventPolicy.max_iterations must be at least 1.")

        self.prepare(len(block))
        correction = cast(Block, self.correction)
        residual_buffer = cast(Block, self.residual_buffer)
        rhs_buffer = cast(Block, self.rhs_buffer)
        operator = self.resolve_operator(len(block))

        block_size = len(block)
        iteration_count = 0

        for _ in range(self.policy.max_iterations):
            self.residual(block, residual_buffer)
            error = residual_buffer.norm()
            scale = block.norm()
            if self.tolerance.accepts(error, scale):
                self.record_solve(block_size, iteration_count, error, scale, True)
                return

            operator.reset()
            self.residual.linearize(block, operator)
            self.resolvent_workspace.combine2_block(0.0, residual_buffer, -1.0, residual_buffer, rhs_buffer)
            self.resolvent_workspace.zero_block(correction)
            self.inverter.bind(operator)
            self.inverter(rhs_buffer, correction)
            self.resolvent_workspace.combine2_block(1.0, block, 1.0, correction, block)
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


__all__ = ["ResolventCoupledNewton"]
