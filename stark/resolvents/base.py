from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from stark.accelerators import AcceleratorAbsent
from stark.auditor import Auditor
from stark.block.operator import BlockOperator
from stark.contracts import AcceleratorLike, Block, Derivative, InnerProduct, IntervalLike, InverterLike, Linearizer, State, Workbench
from stark.execution.safety import Safety
from stark.execution.tolerance import Tolerance
from stark.machinery.stage_solve.workspace import SchemeWorkspace
from stark.resolvents.failure import ResolventError
from stark.resolvents.policy import ResolventPolicy
from stark.resolvents.support.secant import SecantHistory
from stark.resolvents.support.workspace import ResolventWorkspace
from stark.resolvents.tolerance import ResolventTolerance


class ResolventBase(ABC):
    __slots__ = ("interval", "state", "safety", "accelerator", "redirect_call")

    descriptor: object

    def initialise_binding(self, safety: Safety | None = None, accelerator: AcceleratorLike | None = None) -> None:
        self.safety = safety if safety is not None else Safety()
        self.accelerator = accelerator if accelerator is not None else AcceleratorAbsent()
        self.interval = None
        self.state = None
        self.redirect_call = self.call_unbound

    @property
    def short_name(self) -> str:
        return self.descriptor.short_name

    def bind(self, interval: IntervalLike, state: State) -> None:
        self.interval = interval
        self.state = state
        self.redirect_call = self.call_checked if self.safety.block_sizes else self.call_unchecked

    def bind_accelerator(self, accelerator: AcceleratorLike) -> None:
        self.accelerator = accelerator

    def __call__(self, alpha: float, rhs: Block | None, out: Block) -> None:
        self.redirect_call(alpha, rhs, out)

    def call_unbound(self, alpha: float, rhs: Block | None, out: Block) -> None:
        del alpha, rhs, out
        raise RuntimeError(f"{type(self).__name__} must be bound before use.")

    @staticmethod
    def check_one_stage_block(name: str, block: Block) -> None:
        if len(block) != 1:
            raise ValueError(f"{name} must be a one-item block for this resolvent.")

    @abstractmethod
    def call_checked(self, alpha: float, rhs: Block | None, out: Block) -> None:
        """Validate input sizes, then dispatch to the unchecked hot path."""

    @abstractmethod
    def call_unchecked(self, alpha: float, rhs: Block | None, out: Block) -> None:
        """Unchecked hot path for a bound resolvent."""


class ResolventBaseFixedPoint(ResolventBase):
    __slots__ = (
        "tableau",
        "scheme_workspace",
        "resolvent_workspace",
        "residual",
        "tolerance",
        "policy",
        "residual_buffer",
        "size",
    )

    def initialise_fixed_point(
        self,
        derivative: Derivative,
        workbench: Workbench,
        *,
        residual_factory: Callable[[SchemeWorkspace], object],
        tolerance: Tolerance | None = None,
        policy: ResolventPolicy | None = None,
        safety: Safety | None = None,
        accelerator: AcceleratorLike | None = None,
        tableau: Any | None = None,
    ) -> None:
        translation_probe = workbench.allocate_translation()
        Auditor.require_scheme_inputs(derivative, workbench, translation_probe)
        self.tableau = tableau
        self.initialise_binding(safety, accelerator)
        self.scheme_workspace = SchemeWorkspace(workbench, translation_probe)
        self.resolvent_workspace = ResolventWorkspace(
            workbench,
            translation_probe,
            self.safety,
            accelerator=self.accelerator,
        )
        self.tolerance = tolerance if tolerance is not None else ResolventTolerance(atol=1.0e-9, rtol=1.0e-9)
        self.policy = policy if policy is not None else ResolventPolicy()
        self.residual = residual_factory(self.scheme_workspace)
        self.residual_buffer = None
        self.size = -1

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"tolerance={self.tolerance!r}, "
            f"policy={self.policy!r}, "
            f"accelerator={self.accelerator!r}, "
            f"tableau={self.tableau!r})"
        )

    def __str__(self) -> str:
        return type(self).__name__

    def call_checked(self, alpha: float, rhs: Block | None, out: Block) -> None:
        self.check_block_sizes(out, rhs)
        self.call_unchecked(alpha, rhs, out)

    @abstractmethod
    def check_block_sizes(self, out: Block, rhs: Block | None = None) -> None:
        """Validate block sizes for this fixed-point resolvent."""

    def call_unchecked(self, alpha: float, rhs: Block | None, out: Block) -> None:
        interval = self.interval
        state = self.state
        assert interval is not None
        assert state is not None
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
        residual_buffer = self.residual_buffer
        assert residual_buffer is not None

        for _ in range(self.policy.max_iterations):
            self.residual(block, residual_buffer)
            error = residual_buffer.norm()
            scale = block.norm()
            if self.tolerance.accepts(error, scale):
                return
            self.resolvent_workspace.combine2_block(block, 1.0, block, -1.0, residual_buffer)

        self.residual(block, residual_buffer)
        error = residual_buffer.norm()
        scale = block.norm()
        if self.tolerance.accepts(error, scale):
            return
        raise ResolventError(
            f"{self.short_name} failed to resolve the residual within "
            f"{self.policy.max_iterations} iterations (error={error:g})."
        )


class ResolventBaseSecant(ResolventBase):
    __slots__ = (
        "tableau",
        "scheme_workspace",
        "resolvent_workspace",
        "residual",
        "tolerance",
        "policy",
        "depth",
        "history",
        "residual_buffer",
        "size",
    )

    def initialise_secant(
        self,
        derivative: Derivative,
        workbench: Workbench,
        inner_product: InnerProduct,
        *,
        residual_factory: Callable[[SchemeWorkspace], object],
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
        self.initialise_binding(safety, accelerator)
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
        self.residual = residual_factory(self.scheme_workspace)
        self.residual_buffer = None
        self.size = -1

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"tolerance={self.tolerance!r}, "
            f"policy={self.policy!r}, "
            f"depth={self.depth!r}, "
            f"accelerator={self.accelerator!r}, "
            f"tableau={self.tableau!r})"
        )

    def __str__(self) -> str:
        return type(self).__name__

    def call_checked(self, alpha: float, rhs: Block | None, out: Block) -> None:
        self.check_one_stage_block("out", out)
        if rhs is not None:
            self.check_one_stage_block("rhs", rhs)
        self.call_unchecked(alpha, rhs, out)

    def call_unchecked(self, alpha: float, rhs: Block | None, out: Block) -> None:
        interval = self.interval
        state = self.state
        assert interval is not None
        assert state is not None
        self.residual.configure(interval, state, alpha, rhs=rhs)
        self.resolvent_workspace.zero_block(out)
        self.resolve(out)

    @abstractmethod
    def prepare(self, size: int) -> None:
        """Allocate family-specific secant scratch buffers."""

    @abstractmethod
    def resolve(self, block: Block) -> None:
        """Apply the concrete secant-family solve."""


class ResolventBaseLinearized(ResolventBase):
    __slots__ = (
        "tableau",
        "scheme_workspace",
        "resolvent_workspace",
        "residual",
        "tolerance",
        "policy",
        "inverter",
        "correction",
        "residual_buffer",
        "rhs_buffer",
        "operator",
        "size",
    )

    def initialise_linearized(
        self,
        derivative: Derivative,
        workbench: Workbench,
        linearizer: Linearizer,
        inverter: InverterLike,
        *,
        residual_factory: Callable[[SchemeWorkspace], object],
        tolerance: Tolerance | None = None,
        policy: ResolventPolicy | None = None,
        safety: Safety | None = None,
        accelerator: AcceleratorLike | None = None,
        tableau: Any | None = None,
    ) -> None:
        translation_probe = workbench.allocate_translation()
        Auditor.require_scheme_inputs(derivative, workbench, translation_probe)
        Auditor.require_linearizer_inputs(linearizer, workbench, translation_probe)
        self.tableau = tableau
        self.initialise_binding(safety, accelerator)
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
        self.residual = residual_factory(self.scheme_workspace)
        self.correction = None
        self.residual_buffer = None
        self.rhs_buffer = None
        self.operator = None
        self.size = -1

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"tolerance={self.tolerance!r}, "
            f"policy={self.policy!r}, "
            f"inverter={self.inverter!r}, "
            f"accelerator={self.accelerator!r}, "
            f"tableau={self.tableau!r})"
        )

    def __str__(self) -> str:
        return type(self).__name__

    def call_checked(self, alpha: float, rhs: Block | None, out: Block) -> None:
        self.check_block_sizes(out, rhs)
        self.call_unchecked(alpha, rhs, out)

    @abstractmethod
    def check_block_sizes(self, out: Block, rhs: Block | None = None) -> None:
        """Validate block sizes for this Newton-family resolvent."""

    def call_unchecked(self, alpha: float, rhs: Block | None, out: Block) -> None:
        interval = self.interval
        state = self.state
        assert interval is not None
        assert state is not None
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
        operator = self.operator
        assert operator is not None
        return operator

    def resolve(self, block: Block) -> None:
        if self.policy.max_iterations < 1:
            raise ValueError("ResolventPolicy.max_iterations must be at least 1.")

        self.prepare(len(block))
        correction = self.correction
        residual_buffer = self.residual_buffer
        rhs_buffer = self.rhs_buffer
        assert correction is not None
        assert residual_buffer is not None
        assert rhs_buffer is not None
        operator = self.resolve_operator(len(block))

        for _ in range(self.policy.max_iterations):
            self.residual(block, residual_buffer)
            error = residual_buffer.norm()
            scale = block.norm()
            if self.tolerance.accepts(error, scale):
                return

            operator.reset()
            self.residual.linearize(block, operator)
            self.resolvent_workspace.combine2_block(rhs_buffer, 0.0, residual_buffer, -1.0, residual_buffer)
            self.resolvent_workspace.zero_block(correction)
            self.inverter.bind(operator)
            self.inverter(rhs_buffer, correction)
            self.resolvent_workspace.combine2_block(block, 1.0, block, 1.0, correction)

        self.residual(block, residual_buffer)
        error = residual_buffer.norm()
        scale = block.norm()
        if self.tolerance.accepts(error, scale):
            return
        raise ResolventError(
            f"{self.short_name} failed to resolve the residual within "
            f"{self.policy.max_iterations} iterations (error={error:g})."
        )


__all__ = [
    "ResolventBase",
    "ResolventBaseFixedPoint",
    "ResolventBaseLinearized",
    "ResolventBaseSecant",
]
