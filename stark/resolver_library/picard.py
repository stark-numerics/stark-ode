from __future__ import annotations

"""
Picard-style fixed-point residual resolution for STARK blocks.

This resolver applies the classical fixed-point update

    x_{k+1} = x_k - F(x_k)

to a nonlinear residual equation `F(x) = 0`. In the ODE schemes currently in
STARK, the residual is arranged so that subtracting the residual produces the
next Picard iterate directly.

This is the simplest nonlinear resolver in the package: it requires only
residual evaluation, no linearization and no inverter.

See:
    https://en.wikipedia.org/wiki/Fixed-point_iteration
"""

from stark.contracts import Block, Residual, ResolverLike, Workbench
from stark.safety import Safety
from stark.resolver_support.policy import ResolverPolicy
from stark.resolver_support.descriptor import ResolverDescriptor
from stark.resolver_support.failure import ResolutionError
from stark.resolver_support.tolerance import ResolverTolerance
from stark.resolver_support.workspace import ResolverWorkspace
from stark.tolerance import Tolerance


class ResolverPicard:
    """Residual relaxation driven by repeated Picard-style updates."""

    __slots__ = ("tolerance", "policy", "workspace", "residual_buffer", "size", "safety")

    descriptor = ResolverDescriptor("Picard", "Picard Iteration")

    def __init__(
        self,
        workbench: Workbench,
        tolerance: Tolerance | None = None,
        policy: ResolverPolicy | None = None,
        safety: Safety | None = None,
    ) -> None:
        translation_probe = workbench.allocate_translation()
        self.tolerance = tolerance if tolerance is not None else ResolverTolerance(atol=1.0e-9, rtol=1.0e-9)
        self.policy = policy if policy is not None else ResolverPolicy()
        self.safety = safety if safety is not None else Safety()
        self.workspace = ResolverWorkspace(workbench, translation_probe, self.safety)
        self.residual_buffer: Block | None = None
        self.size = -1

    @property
    def short_name(self) -> str:
        return self.descriptor.short_name

    @property
    def full_name(self) -> str:
        return self.descriptor.full_name

    def __repr__(self) -> str:
        return f"ResolverPicard(tolerance={self.tolerance!r}, policy={self.policy!r})"

    def __str__(self) -> str:
        return f"{self.short_name} with {self.tolerance}, {self.policy}"

    def prepare(self, size: int) -> None:
        """Allocate cached residual storage for blocks of a specific length."""
        if self.size == size:
            return
        self.size = size
        self.residual_buffer = self.workspace.allocate_block(size)

    def __call__(self, block: Block, residual: Residual) -> None:
        if self.policy.max_iterations < 1:
            raise ValueError("ResolverPolicy.max_iterations must be at least 1.")

        self.prepare(len(block))
        residual_buffer = self.residual_buffer
        assert residual_buffer is not None
        tolerance = self.tolerance
        policy = self.policy
        workspace = self.workspace

        for _ in range(policy.max_iterations):
            residual(residual_buffer, block)
            error = residual_buffer.norm()
            scale = block.norm()
            if tolerance.accepts(error, scale):
                return
            workspace.combine2_block(block, 1.0, block, -1.0, residual_buffer)

        residual(residual_buffer, block)
        error = residual_buffer.norm()
        scale = block.norm()
        if tolerance.accepts(error, scale):
            return
        raise ResolutionError(
            f"{self.short_name} failed to resolve the residual within "
            f"{policy.max_iterations} iterations (error={error:g})."
        )


__all__ = ["ResolverPicard"]
