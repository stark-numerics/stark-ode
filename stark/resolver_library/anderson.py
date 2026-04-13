from __future__ import annotations

"""
Anderson-accelerated fixed-point resolution for STARK blocks.

Anderson acceleration starts from the same fixed-point map as Picard iteration,
but mixes recent fixed-point updates to reduce the current residual over a small
history window. In STARK terms, with residual `F(x)`, the underlying map is

    G(x) = x - F(x)

and the next iterate is chosen by combining recent differences of `G(x)` and
`F(x)`.

See:
    https://en.wikipedia.org/wiki/Anderson_acceleration
"""

from stark.contracts import Block, InnerProduct, Residual, Workbench
from stark.safety import Safety
from stark.resolver_support.descriptor import ResolverDescriptor
from stark.resolver_support.failure import ResolutionError
from stark.resolver_support.policy import ResolverPolicy
from stark.resolver_support.secant import SecantHistory
from stark.resolver_support.tolerance import ResolverTolerance
from stark.resolver_support.workspace import ResolverWorkspace
from stark.tolerance import Tolerance


class ResolverAnderson:
    """Anderson-accelerated fixed-point iteration on STARK blocks."""

    __slots__ = (
        "tolerance",
        "policy",
        "depth",
        "workspace",
        "history",
        "residual_buffer",
        "previous_residual",
        "fixed_point",
        "previous_fixed_point",
        "correction",
        "size",
        "safety",
    )

    descriptor = ResolverDescriptor("Anderson", "Anderson Acceleration")

    def __init__(
        self,
        workbench: Workbench,
        inner_product: InnerProduct,
        tolerance: Tolerance | None = None,
        policy: ResolverPolicy | None = None,
        depth: int = 4,
        safety: Safety | None = None,
    ) -> None:
        if depth < 1:
            raise ValueError("Anderson depth must be at least 1.")
        translation_probe = workbench.allocate_translation()
        self.tolerance = tolerance if tolerance is not None else ResolverTolerance(atol=1.0e-9, rtol=1.0e-9)
        self.policy = policy if policy is not None else ResolverPolicy()
        self.depth = depth
        self.safety = safety if safety is not None else Safety()
        self.workspace = ResolverWorkspace(workbench, translation_probe, self.safety, inner_product=inner_product)
        self.history = SecantHistory(self.workspace, depth)
        self.residual_buffer: Block | None = None
        self.previous_residual: Block | None = None
        self.fixed_point: Block | None = None
        self.previous_fixed_point: Block | None = None
        self.correction: Block | None = None
        self.size = -1

    @property
    def short_name(self) -> str:
        return self.descriptor.short_name

    @property
    def full_name(self) -> str:
        return self.descriptor.full_name

    def __repr__(self) -> str:
        return f"ResolverAnderson(tolerance={self.tolerance!r}, policy={self.policy!r}, depth={self.depth!r})"

    def __str__(self) -> str:
        return f"{self.short_name} with {self.tolerance}, {self.policy}"

    def prepare(self, size: int) -> None:
        """Allocate cached solver storage for blocks of a specific length."""
        if self.size == size:
            return
        workspace = self.workspace
        self.size = size
        self.residual_buffer = workspace.allocate_block(size)
        self.previous_residual = workspace.allocate_block(size)
        self.fixed_point = workspace.allocate_block(size)
        self.previous_fixed_point = workspace.allocate_block(size)
        self.correction = workspace.allocate_block(size)
        self.history.ensure_size(size)

    def __call__(self, block: Block, residual: Residual) -> None:
        if self.policy.max_iterations < 1:
            raise ValueError("ResolverPolicy.max_iterations must be at least 1.")

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

        workspace = self.workspace
        tolerance = self.tolerance
        policy = self.policy
        history = self.history
        history.clear()
        have_previous = False

        for _ in range(policy.max_iterations):
            residual(residual_buffer, block)
            error = residual_buffer.norm()
            scale = block.norm()
            if tolerance.accepts(error, scale):
                return

            workspace.combine2_block(fixed_point, 1.0, block, -1.0, residual_buffer)
            if have_previous:
                history.append_difference(fixed_point, previous_fixed_point, residual_buffer, previous_residual)

            if len(history) > 0:
                coefficients = history.solve_right_least_squares(residual_buffer)
                history.combine_left(correction, coefficients)
                workspace.combine2_block(block, 1.0, fixed_point, -1.0, correction)
            else:
                workspace.copy_block(block, fixed_point)

            workspace.copy_block(previous_fixed_point, fixed_point)
            workspace.copy_block(previous_residual, residual_buffer)
            have_previous = True

        residual(residual_buffer, block)
        error = residual_buffer.norm()
        scale = block.norm()
        if tolerance.accepts(error, scale):
            return
        raise ResolutionError(
            f"{self.short_name} failed to resolve the residual within "
            f"{policy.max_iterations} iterations (error={error:g})."
        )


__all__ = ["ResolverAnderson"]
