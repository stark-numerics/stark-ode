from __future__ import annotations

from stark.inverter_support.block_operator import BlockOperator
from stark.contracts import Block, InverterLike, LinearResidual, Workbench
from stark.resolver_support.resolution import Resolution
from stark.resolver_support.descriptor import ResolverDescriptor
from stark.resolver_support.workspace import ResolverWorkspace


class ResolverNewton:
    """Newton iteration driven by a residual linearization and an inverter."""

    __slots__ = ("resolution", "workspace", "inverter", "correction", "residual_buffer", "rhs_buffer", "operator")

    descriptor = ResolverDescriptor("Newton", "Newton Iteration")

    def __init__(
        self,
        workbench: Workbench,
        inverter: InverterLike,
        resolution: Resolution | None = None,
    ) -> None:
        translation_probe = workbench.allocate_translation()
        self.resolution = resolution if resolution is not None else Resolution()
        self.workspace = ResolverWorkspace(workbench, translation_probe)
        self.inverter = inverter
        self.correction: Block | None = None
        self.residual_buffer: Block | None = None
        self.rhs_buffer: Block | None = None
        self.operator: BlockOperator | None = None

    @property
    def short_name(self) -> str:
        return self.descriptor.short_name

    @property
    def full_name(self) -> str:
        return self.descriptor.full_name

    def __repr__(self) -> str:
        return f"ResolverNewton(resolution={self.resolution!r}, inverter={self.inverter!r})"

    def __str__(self) -> str:
        return f"{self.short_name} with {self.resolution}"

    def __call__(self, block: Block, residual: LinearResidual) -> None:
        if self.resolution.max_iterations < 1:
            raise ValueError("Resolution.max_iterations must be at least 1.")
        if not callable(getattr(residual, "linearize", None)):
            raise TypeError(f"{self.short_name} requires a residual with linearize(out, block).")

        correction = self._require_correction(len(block))
        residual_buffer = self._require_residual_buffer(len(block))
        rhs_buffer = self._require_rhs_buffer(len(block))
        operator = self._require_operator(len(block))
        resolution = self.resolution
        workspace = self.workspace

        for _ in range(resolution.max_iterations):
            residual(residual_buffer, block)
            error = residual_buffer.norm()
            scale = block.norm()
            if resolution.accepts(error, scale):
                return

            residual.linearize(operator, block)
            workspace.combine2_block(rhs_buffer, 0.0, residual_buffer, -1.0, residual_buffer)
            workspace.zero_block(correction)
            self.inverter.bind(operator)
            self.inverter(correction, rhs_buffer)
            workspace.combine2_block(block, 1.0, block, 1.0, correction)

        residual(residual_buffer, block)
        error = residual_buffer.norm()
        raise RuntimeError(
            f"{self.short_name} failed to resolve the residual within "
            f"{resolution.max_iterations} iterations (error={error:g})."
        )

    def _require_correction(self, size: int) -> Block:
        if self.correction is None or len(self.correction) != size:
            self.correction = self.workspace.allocate_block(size)
        return self.correction

    def _require_residual_buffer(self, size: int) -> Block:
        if self.residual_buffer is None or len(self.residual_buffer) != size:
            self.residual_buffer = self.workspace.allocate_block(size)
        return self.residual_buffer

    def _require_rhs_buffer(self, size: int) -> Block:
        if self.rhs_buffer is None or len(self.rhs_buffer) != size:
            self.rhs_buffer = self.workspace.allocate_block(size)
        return self.rhs_buffer

    def _require_operator(self, size: int) -> BlockOperator:
        if self.operator is None or len(self.operator.operators) != size:
            self.operator = BlockOperator([_UnsetOperator() for _ in range(size)])
        return self.operator


class _UnsetOperator:
    def __call__(self, out, translation) -> None:
        del out, translation
        raise RuntimeError("Newton linear operator was used before residual.linearize(...) configured it.")


__all__ = ["ResolverNewton"]
