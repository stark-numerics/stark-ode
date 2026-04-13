from __future__ import annotations

"""
Newton residual resolution for STARK blocks.

This resolver applies Newton's method to a nonlinear residual equation

    F(x) = 0

by repeatedly solving linearized correction problems

    J(x_k) delta_k = -F(x_k)

and updating

    x_{k+1} = x_k + delta_k.

The residual object must therefore support both evaluation and linearization;
in STARK that contract is represented by `LinearResidual`.

See:
    https://en.wikipedia.org/wiki/Newton%27s_method
"""

from stark.audit import Auditor
from stark.inverter_support.block_operator import BlockOperator
from stark.contracts import Block, InverterLike, LinearResidual, Workbench
from stark.safety import Safety
from stark.resolver_support.policy import ResolverPolicy
from stark.resolver_support.descriptor import ResolverDescriptor
from stark.resolver_support.failure import ResolutionError
from stark.resolver_support.tolerance import ResolverTolerance
from stark.resolver_support.workspace import ResolverWorkspace
from stark.tolerance import Tolerance


class ResolverNewton:
    """Newton iteration driven by a residual linearization and an inverter."""

    __slots__ = ("tolerance", "policy", "workspace", "inverter", "correction", "residual_buffer", "rhs_buffer", "operator", "size", "safety")

    descriptor = ResolverDescriptor("Newton", "Newton Iteration")

    def __init__(
        self,
        workbench: Workbench,
        inverter: InverterLike,
        tolerance: Tolerance | None = None,
        policy: ResolverPolicy | None = None,
        safety: Safety | None = None,
    ) -> None:
        translation_probe = workbench.allocate_translation()
        self.tolerance = tolerance if tolerance is not None else ResolverTolerance(atol=1.0e-9, rtol=1.0e-9)
        self.policy = policy if policy is not None else ResolverPolicy()
        self.safety = safety if safety is not None else Safety()
        self.workspace = ResolverWorkspace(workbench, translation_probe, self.safety)
        self.inverter = inverter
        self.correction: Block | None = None
        self.residual_buffer: Block | None = None
        self.rhs_buffer: Block | None = None
        self.operator: BlockOperator | None = None
        self.size = -1

    @property
    def short_name(self) -> str:
        return self.descriptor.short_name

    @property
    def full_name(self) -> str:
        return self.descriptor.full_name

    def __repr__(self) -> str:
        return f"ResolverNewton(tolerance={self.tolerance!r}, policy={self.policy!r}, inverter={self.inverter!r})"

    def __str__(self) -> str:
        return f"{self.short_name} with {self.tolerance}, {self.policy}"

    def prepare(self, size: int) -> None:
        """Allocate cached solver storage for blocks of a specific length."""
        if self.size == size:
            return
        self.size = size
        self.correction = self.workspace.allocate_block(size)
        self.residual_buffer = self.workspace.allocate_block(size)
        self.rhs_buffer = self.workspace.allocate_block(size)
        self.operator = BlockOperator([None for _ in range(size)], check_sizes=self.safety.block_sizes)  # type: ignore[list-item]

    def __call__(self, block: Block, residual: LinearResidual) -> None:
        if self.policy.max_iterations < 1:
            raise ValueError("ResolverPolicy.max_iterations must be at least 1.")
        Auditor.require_linear_residual(residual)

        self.prepare(len(block))
        correction = self.correction
        residual_buffer = self.residual_buffer
        rhs_buffer = self.rhs_buffer
        operator = self.operator
        assert correction is not None
        assert residual_buffer is not None
        assert rhs_buffer is not None
        assert operator is not None
        tolerance = self.tolerance
        policy = self.policy
        workspace = self.workspace

        for _ in range(policy.max_iterations):
            residual(residual_buffer, block)
            error = residual_buffer.norm()
            scale = block.norm()
            if tolerance.accepts(error, scale):
                return

            operator.reset()
            residual.linearize(operator, block)
            workspace.combine2_block(rhs_buffer, 0.0, residual_buffer, -1.0, residual_buffer)
            workspace.zero_block(correction)
            self.inverter.bind(operator)
            self.inverter(correction, rhs_buffer)
            workspace.combine2_block(block, 1.0, block, 1.0, correction)

        residual(residual_buffer, block)
        error = residual_buffer.norm()
        scale = block.norm()
        if tolerance.accepts(error, scale):
            return
        raise ResolutionError(
            f"{self.short_name} failed to resolve the residual within "
            f"{policy.max_iterations} iterations (error={error:g})."
        )


__all__ = ["ResolverNewton"]
