from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import sqrt
from typing import ClassVar, Generic

from stark.core import Configuration
from stark.core.block import Block, BlockAllocator
from stark.core.contracts import (
    Accelerator,
    Allocator,
    BlockLike,
    BlockOperatorLike,
    InnerProduct,
    InverterOutputMode,
    InverterRequest,
    Translation,
    TranslationType,
)
from stark.engines.shared.accelerators import AcceleratorNone
from stark.engines.shared.algebraist.runtime import AlgebraistRuntimeLinearCombine
from stark.methods.inverters.configuration import InverterConfiguration
from stark.methods.inverters.krylov.basis import InverterKrylovBasis
from stark.methods.inverters.krylov.projection import InverterKrylovProjection
from stark.methods.inverters.krylov.preconditioner import InverterKrylovPreconditionerLike
from stark.methods.inverters.support import (
    InverterDefect,
    InverterDescriptor,
    MonitorInverterLike,
    with_inverter_monitoring,
)


@dataclass(slots=True)
class InverterKrylovArnoldiInstance(Generic[TranslationType]):
    """Operator-bound Arnoldi solve action."""

    inverter: InverterKrylovArnoldi[TranslationType]
    operator: BlockOperatorLike[TranslationType]

    def __call__(
        self,
        residual: BlockLike[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        self.inverter.call(self.operator, residual, output)


# Optional extension: records inverter monitor events.
# Provides: record_solve.
@with_inverter_monitoring
@dataclass(slots=True)
class InverterKrylovArnoldi(Generic[TranslationType]):
    """
    Restarted Arnoldi/GMRES inverter for matrix-free block linear systems.

    Problem:
        Improve output so that operator(output) = residual.

    Algorithm:
        1. Compute the current defect residual - operator(output).
        2. Build an Arnoldi basis using repeated operator actions.
        3. Solve the small Hessenberg least-squares problem.
        4. Accumulate the Krylov correction into output.
        5. Restart until accepted or the configured step budget is exhausted.

    The implementation deliberately follows the current inverter contract.  It
    does not expose the legacy bind-then-solve API; repeated operator-specific
    use goes through instance(operator).
    """

    allocator: Allocator
    inner_product: InnerProduct
    restart: int = 16
    breakdown_tolerance: float = 1.0e-30
    configuration: InverterConfiguration | None = None
    monitor: MonitorInverterLike | None = None
    accelerator: Accelerator | None = None
    preconditioner: InverterKrylovPreconditionerLike[TranslationType] | None = None

    descriptor: ClassVar[InverterDescriptor] = InverterDescriptor("Krylov", "Arnoldi Krylov")
    output_mode: ClassVar[InverterOutputMode] = InverterOutputMode.improve

    block_allocator: BlockAllocator[TranslationType] = field(init=False, repr=False)
    sample: Translation = field(init=False, repr=False)
    scale: Callable[..., TranslationType] = field(init=False, repr=False)
    combine2: Callable[..., TranslationType] = field(init=False, repr=False)
    defect: InverterDefect[TranslationType] = field(init=False, repr=False)
    basis: InverterKrylovBasis[TranslationType] = field(init=False, repr=False)
    projection: InverterKrylovProjection = field(init=False, repr=False)
    correction: BlockLike[TranslationType] | None = field(init=False, default=None, repr=False)
    temporary: BlockLike[TranslationType] | None = field(init=False, default=None, repr=False)
    preconditioned_residual: BlockLike[TranslationType] | None = field(init=False, default=None, repr=False)
    size: int = field(init=False, default=-1)
    tolerance: object = field(init=False, repr=False)
    maximum_steps: int = field(init=False)
    call: Callable[[BlockOperatorLike[TranslationType], BlockLike[TranslationType], BlockLike[TranslationType]], None] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.restart < 1:
            raise ValueError("InverterKrylovArnoldi.restart must be at least 1.")
        if self.breakdown_tolerance < 0.0:
            raise ValueError("InverterKrylovArnoldi.breakdown_tolerance must be non-negative.")

        configuration = self.configuration if self.configuration is not None else Configuration()
        accelerator = self.accelerator if self.accelerator is not None else AcceleratorNone()
        self.accelerator = accelerator
        self.tolerance = configuration.inverter_tolerance
        self.maximum_steps = configuration.inverter_maximum_steps
        self.block_allocator = BlockAllocator(self.allocator)
        self.sample = self.allocator.allocate_translation()
        kernels = AlgebraistRuntimeLinearCombine(
            translation=self.sample,
            allocator=self.allocator,
            accelerator=accelerator,
        ).as_tuple(2)
        self.scale = kernels[0]
        self.combine2 = kernels[1]
        self.defect = InverterDefect()
        self.basis = InverterKrylovBasis(self, self.restart)
        self.projection = InverterKrylovProjection(self.restart)
        self.call = self.call_monitored if self.monitor is not None else self.call_body

    def __call__(
        self,
        request: InverterRequest[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        self.call(request.operator, request.residual, output)

    def instance(
        self,
        operator: BlockOperatorLike[TranslationType],
    ) -> InverterKrylovArnoldiInstance[TranslationType]:
        return InverterKrylovArnoldiInstance(self, operator)

    def allocate_block(self, size: int) -> Block[TranslationType]:
        return self.block_allocator.allocate(size)

    def copy_block(
        self,
        source: BlockLike[TranslationType],
        target: BlockLike[TranslationType],
    ) -> None:
        scale = self.scale
        for index in range(len(source)):
            target[index] = scale(1.0, source[index], target[index])

    def zero_block(self, target: BlockLike[TranslationType]) -> None:
        scale = self.scale
        for index in range(len(target)):
            target[index] = scale(0.0, target[index], target[index])

    def scale_block(
        self,
        coefficient: float,
        source: BlockLike[TranslationType],
        target: BlockLike[TranslationType],
    ) -> None:
        scale = self.scale
        for index in range(len(source)):
            target[index] = scale(coefficient, source[index], target[index])

    def combine2_block(
        self,
        left_coefficient: float,
        left: BlockLike[TranslationType],
        right_coefficient: float,
        right: BlockLike[TranslationType],
        target: BlockLike[TranslationType],
    ) -> None:
        combine2 = self.combine2
        for index in range(len(left)):
            target[index] = combine2(
                left_coefficient,
                left[index],
                right_coefficient,
                right[index],
                target[index],
            )


    def precondition_block(
        self,
        operator: BlockOperatorLike[TranslationType],
        source: BlockLike[TranslationType],
        target: BlockLike[TranslationType],
    ) -> None:
        preconditioner = self.preconditioner
        if preconditioner is None:
            self.copy_block(source, target)
            return
        preconditioner(operator, source, target)

    def inner_product_block(
        self,
        left: BlockLike[TranslationType],
        right: BlockLike[TranslationType],
    ) -> float:
        return float(
            sum(
                self.inner_product(left[index], right[index])
                for index in range(len(left))
            )
        )

    def norm_block(self, block: BlockLike[TranslationType]) -> float:
        return sqrt(max(0.0, self.inner_product_block(block, block)))

    def prepare(self, output: BlockLike[TranslationType]) -> None:
        size = len(output)
        if self.size == size:
            return
        self.size = size
        self.basis.prepare(size)
        self.correction = self.allocate_block(size)
        self.temporary = self.allocate_block(size)
        self.preconditioned_residual = self.allocate_block(size)

    def defect_norm(
        self,
        operator: BlockOperatorLike[TranslationType],
        residual: BlockLike[TranslationType],
        output: BlockLike[TranslationType],
    ) -> float:
        """Store residual - operator(output) in the reusable defect block."""

        self.defect.prepare(output)
        image = self.defect.image
        block = self.defect.block
        assert image is not None
        assert block is not None
        operator(output, image)
        block.replace(residual - image)  # type: ignore[operator]
        return block.norm()

    def call_body(
        self,
        operator: BlockOperatorLike[TranslationType],
        residual: BlockLike[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        self.solve(operator, residual, output)

    def call_monitored(
        self,
        operator: BlockOperatorLike[TranslationType],
        residual: BlockLike[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        converged, iterations, initial, final = self.solve(operator, residual, output)
        self.record_solve(
            converged=converged,
            iteration_count=iterations,
            initial_residual=initial,
            final_residual=final,
            failure_reason=None if converged else "maximum steps reached",
        )

    def solve(
        self,
        operator: BlockOperatorLike[TranslationType],
        residual: BlockLike[TranslationType],
        output: BlockLike[TranslationType],
    ) -> tuple[bool, int, float, float]:
        self.prepare(output)
        rhs_norm = residual.norm()
        initial = self.defect_norm(operator, residual, output)
        final = initial
        if self.tolerance.accepts(initial, rhs_norm):  # type: ignore[attr-defined]
            return True, 0, initial, final

        iterations = 0
        while iterations < self.maximum_steps:
            used, final = self.run_window(
                operator,
                residual,
                output,
                rhs_norm,
                remaining=self.maximum_steps - iterations,
            )
            iterations += used
            if self.tolerance.accepts(final, rhs_norm):  # type: ignore[attr-defined]
                return True, iterations, initial, final
            if used == 0:
                break

        return False, iterations, initial, final

    def run_window(
        self,
        operator: BlockOperatorLike[TranslationType],
        residual: BlockLike[TranslationType],
        output: BlockLike[TranslationType],
        rhs_norm: float,
        *,
        remaining: int,
    ) -> tuple[int, float]:
        residual_block = self.defect.block
        preconditioned_residual = self.preconditioned_residual
        assert residual_block is not None
        assert preconditioned_residual is not None
        self.precondition_block(operator, residual_block, preconditioned_residual)
        beta = self.norm_block(preconditioned_residual)
        if beta <= self.breakdown_tolerance:
            return 0, beta

        width = min(self.restart, remaining)
        self.projection.reset(beta)
        self.basis.start(preconditioned_residual, beta)

        last_column = -1
        residual_estimate = beta
        for column in range(width):
            broke_down = self.basis.build_column(
                column,
                operator,
                self.projection,
                breakdown_tolerance=self.breakdown_tolerance,
            )
            residual_estimate = self.projection.apply_new(column)
            last_column = column
            if broke_down or self.tolerance.accepts(residual_estimate, rhs_norm):  # type: ignore[attr-defined]
                break

        used = last_column + 1
        self.apply_correction(output, used)
        final = self.defect_norm(operator, residual, output)
        return used, final

    def apply_correction(
        self,
        output: BlockLike[TranslationType],
        width: int,
    ) -> None:
        if width <= 0:
            return
        correction = self.correction
        temporary = self.temporary
        assert correction is not None
        assert temporary is not None

        self.zero_block(correction)
        coefficients = self.projection.solve(width)
        for index in range(width):
            self.combine2_block(
                1.0,
                correction,
                coefficients[index],
                self.basis.vectors[index],
                temporary,
            )
            self.copy_block(temporary, correction)

        self.combine2_block(1.0, output, 1.0, correction, temporary)
        output.replace(temporary)


__all__ = [
    "InverterKrylovArnoldi",
    "InverterKrylovArnoldiInstance",
]
