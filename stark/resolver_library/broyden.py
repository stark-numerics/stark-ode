from __future__ import annotations

"""
Inverse-Broyden residual resolution for STARK blocks.

This resolver keeps a low-rank approximation to the inverse Jacobian using
secant updates. The implementation here follows the inverse form of Broyden's
second method:

    H_{k+1} = H_k + ((s_k - H_k y_k) y_k^T) / (y_k^T y_k)

with

    s_k = x_{k+1} - x_k
    y_k = F(x_{k+1}) - F(x_k).

That update is convenient in STARK because it only needs block inner products
and low-rank history terms; it does not require a linearizer or inverter.

See:
    https://en.wikipedia.org/wiki/Broyden%27s_method
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


class ResolverBroyden:
    """Limited-memory inverse-Broyden resolution on STARK blocks."""

    __slots__ = (
        "tolerance",
        "policy",
        "depth",
        "workspace",
        "history",
        "residual_buffer",
        "next_residual",
        "correction",
        "trial",
        "residual_delta",
        "inverse_residual_delta",
        "scaled_update",
        "history_correction",
        "size",
        "safety",
    )

    descriptor = ResolverDescriptor("Broyden", "Inverse Broyden")

    def __init__(
        self,
        workbench: Workbench,
        inner_product: InnerProduct,
        tolerance: Tolerance | None = None,
        policy: ResolverPolicy | None = None,
        depth: int = 8,
        safety: Safety | None = None,
    ) -> None:
        if depth < 1:
            raise ValueError("Broyden depth must be at least 1.")
        translation_probe = workbench.allocate_translation()
        self.tolerance = tolerance if tolerance is not None else ResolverTolerance(atol=1.0e-9, rtol=1.0e-9)
        self.policy = policy if policy is not None else ResolverPolicy()
        self.depth = depth
        self.safety = safety if safety is not None else Safety()
        self.workspace = ResolverWorkspace(workbench, translation_probe, self.safety, inner_product=inner_product)
        self.history = SecantHistory(self.workspace, depth)
        self.residual_buffer: Block | None = None
        self.next_residual: Block | None = None
        self.correction: Block | None = None
        self.trial: Block | None = None
        self.residual_delta: Block | None = None
        self.inverse_residual_delta: Block | None = None
        self.scaled_update: Block | None = None
        self.history_correction: Block | None = None
        self.size = -1

    @property
    def short_name(self) -> str:
        return self.descriptor.short_name

    @property
    def full_name(self) -> str:
        return self.descriptor.full_name

    def __repr__(self) -> str:
        return f"ResolverBroyden(tolerance={self.tolerance!r}, policy={self.policy!r}, depth={self.depth!r})"

    def __str__(self) -> str:
        return f"{self.short_name} with {self.tolerance}, {self.policy}"

    def prepare(self, size: int) -> None:
        """Allocate cached solver storage for blocks of a specific length."""
        if self.size == size:
            return
        workspace = self.workspace
        self.size = size
        self.residual_buffer = workspace.allocate_block(size)
        self.next_residual = workspace.allocate_block(size)
        self.correction = workspace.allocate_block(size)
        self.trial = workspace.allocate_block(size)
        self.residual_delta = workspace.allocate_block(size)
        self.inverse_residual_delta = workspace.allocate_block(size)
        self.scaled_update = workspace.allocate_block(size)
        self.history_correction = workspace.allocate_block(size)
        self.history.ensure_size(size)

    def __call__(self, block: Block, residual: Residual) -> None:
        if self.policy.max_iterations < 1:
            raise ValueError("ResolverPolicy.max_iterations must be at least 1.")

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

        workspace = self.workspace
        tolerance = self.tolerance
        policy = self.policy
        history = self.history
        history.clear()

        for _ in range(policy.max_iterations):
            residual(residual_buffer, block)
            error = residual_buffer.norm()
            scale = block.norm()
            if tolerance.accepts(error, scale):
                return

            self._apply_inverse(correction, residual_buffer, history_correction)
            workspace.scale_block(correction, -1.0, correction)
            workspace.combine2_block(trial, 1.0, block, 1.0, correction)
            residual(next_residual, trial)

            workspace.combine2_block(residual_delta, 1.0, next_residual, -1.0, residual_buffer)
            denominator = workspace.inner_product(residual_delta, residual_delta)
            if denominator > 0.0:
                self._apply_inverse(inverse_residual_delta, residual_delta, history_correction)
                workspace.combine2_block(scaled_update, 1.0, correction, -1.0, inverse_residual_delta)
                workspace.scale_block(scaled_update, 1.0 / denominator, scaled_update)
                history.append(scaled_update, residual_delta)

            workspace.copy_block(block, trial)

        residual(residual_buffer, block)
        error = residual_buffer.norm()
        scale = block.norm()
        if tolerance.accepts(error, scale):
            return
        raise ResolutionError(
            f"{self.short_name} failed to resolve the residual within "
            f"{policy.max_iterations} iterations (error={error:g})."
        )

    def _apply_inverse(self, out: Block, block: Block, history_correction: Block) -> None:
        """Apply the current low-rank inverse approximation to `block`."""
        workspace = self.workspace
        workspace.copy_block(out, block)
        if len(self.history) == 0:
            return
        coefficients = self.history.project_right(block)
        self.history.combine_left(history_correction, coefficients)
        workspace.combine2_block(out, 1.0, out, 1.0, history_correction)


__all__ = ["ResolverBroyden"]
