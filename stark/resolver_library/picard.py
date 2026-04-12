from __future__ import annotations

from stark.contracts import Block, Residual, ResolverLike, Workbench
from stark.resolver_support.resolution import Resolution
from stark.resolver_support.descriptor import ResolverDescriptor
from stark.resolver_support.workspace import ResolverWorkspace


class ResolverPicard:
    """Residual relaxation driven by repeated Picard-style updates."""

    __slots__ = ("resolution", "workspace", "residual_buffer")

    descriptor = ResolverDescriptor("Picard", "Picard Iteration")

    def __init__(self, workbench: Workbench, resolution: Resolution | None = None) -> None:
        translation_probe = workbench.allocate_translation()
        self.resolution = resolution if resolution is not None else Resolution()
        self.workspace = ResolverWorkspace(workbench, translation_probe)
        self.residual_buffer: Block | None = None

    @property
    def short_name(self) -> str:
        return self.descriptor.short_name

    @property
    def full_name(self) -> str:
        return self.descriptor.full_name

    def __repr__(self) -> str:
        return (
            "ResolverPicard("
            f"resolution={self.resolution!r})"
        )

    def __str__(self) -> str:
        return f"{self.short_name} with {self.resolution}"

    def __call__(self, block: Block, residual: Residual) -> None:
        if self.resolution.max_iterations < 1:
            raise ValueError("Resolution.max_iterations must be at least 1.")

        residual_buffer = self._require_residual_buffer(len(block))
        resolution = self.resolution
        workspace = self.workspace

        for _ in range(resolution.max_iterations):
            residual(residual_buffer, block)
            error = residual_buffer.norm()
            scale = block.norm()
            if resolution.accepts(error, scale):
                return
            workspace.combine2_block(block, 1.0, block, -1.0, residual_buffer)

        residual(residual_buffer, block)
        error = residual_buffer.norm()
        raise RuntimeError(
            f"{self.short_name} failed to resolve the residual within "
            f"{resolution.max_iterations} iterations (error={error:g})."
        )

    def _require_residual_buffer(self, size: int) -> Block:
        if self.residual_buffer is None or len(self.residual_buffer) != size:
            self.residual_buffer = self.workspace.allocate_block(size)
        return self.residual_buffer


__all__ = ["ResolverPicard"]
