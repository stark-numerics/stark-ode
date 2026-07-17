from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar, Generic, cast

from stark.core.contracts import (
    BlockOperatorEntryLike,
    BlockOperatorDiagonalLike,
    BlockLike,
    InverterOutputMode,
    InverterRequest,
    TranslationType,
)
from stark.methods.inverters.configuration import (
    InverterConfiguration,
    InverterConfigurationDefault,
)
from stark.methods.inverters.relaxation.linear_fixed import InverterRelaxationLinearFixed
from stark.methods.inverters.relaxation.stencil import InverterRelaxationStencilUpdate
from stark.methods.inverters.support import (
    InverterDefect,
    InverterDescriptor,
    InverterRecordSolve,
    MonitorInverterLike,
    with_inverter_monitoring,
)

InverterRelaxationJacobiInverse = Callable[
    [BlockOperatorEntryLike[TranslationType], TranslationType, TranslationType],
    None,
]


# Optional extension: records inverter monitor events.
# Provides: record_solve.
@with_inverter_monitoring
class InverterRelaxationJacobi(Generic[TranslationType]):
    """
    Jacobi relaxation inverter.

    Jacobi relaxation improves an output block by applying an approximate
    inverse of each diagonal block-operator entry to the current defect.

    Problem:
        Improve output so that request.operator(output) = request.residual.

    Algorithm:
        1. Compute defect = request.residual - request.operator(output).
        2. Accept if the defect norm is within tolerance.
        3. Apply each diagonal inverse to the corresponding defect entry.
        4. Apply output <- output + damping * update.
        5. Repeat until accepted or the step budget is exhausted.
    """

    __slots__ = (
        "call_body",
        "damping",
        "defect",
        "diagonal_inverse",
        "maximum_steps",
        "monitor",
        "output_buffer",
        "output_size",
        "redirect_call",
        "tolerance",
        "update",
        "update_output",
        "update_size",
    )

    descriptor: ClassVar[InverterDescriptor] = InverterDescriptor("Jacobi", "Jacobi relaxation")
    output_mode: ClassVar[InverterOutputMode] = InverterOutputMode.improve
    # Installed by with_inverter_monitoring from stark.methods.inverters.support.
    record_solve: ClassVar[InverterRecordSolve]

    def __init__(
        self,
        diagonal_inverse: InverterRelaxationJacobiInverse[TranslationType],
        *,
        damping: float = 1.0,
        configuration: InverterConfiguration | None = None,
        monitor: MonitorInverterLike | None = None,
        linear_fixed: InverterRelaxationLinearFixed[TranslationType] | None = None,
    ) -> None:
        if damping <= 0.0:
            raise ValueError("InverterRelaxationJacobi.damping must be positive.")

        configuration = configuration if configuration is not None else InverterConfigurationDefault()
        self.diagonal_inverse = diagonal_inverse
        self.damping = damping
        self.tolerance = configuration.inverter_tolerance
        self.maximum_steps = configuration.inverter_maximum_steps
        self.monitor = monitor
        self.defect = InverterDefect[TranslationType]()
        self.output_buffer = None
        self.output_size = -1
        self.update: BlockLike[TranslationType] | None = None
        self.update_output = None
        self.update_size = -1
        self.call_body = self.call_inline

        if linear_fixed is not None:
            self.prepare_specialized_kernels(linear_fixed)
            self.call_body = self.call_specialized
        self.redirect_call = self.call_monitored if monitor is not None else self.call_unmonitored

    def __call__(
        self,
        request: InverterRequest[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        return self.redirect_call(request, output)

    def call_unmonitored(
        self,
        request: InverterRequest[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        self.call_body(request, output)

    def call_monitored(
        self,
        request: InverterRequest[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        converged, iteration_count, initial_residual, final_residual = self.call_body(
            request,
            output,
        )
        self.record_solve(
            converged=converged,
            iteration_count=iteration_count,
            initial_residual=initial_residual,
            final_residual=final_residual,
            failure_reason=None if converged else "maximum steps reached",
        )

    def prepare_specialized_kernels(
        self,
        linear_fixed: InverterRelaxationLinearFixed[TranslationType],
    ) -> None:
        """Prepare the relaxation update kernel for the specialized path."""

        # Step 4 applies output <- output + damping * update.
        self.update_output = linear_fixed(
            InverterRelaxationStencilUpdate(self.damping)
        )

    def prepare_update(self, output: BlockLike[TranslationType]) -> None:
        """Allocate the update block matching the current output size."""

        size = len(output)
        if self.update_size == size:
            return

        self.update = 0.0 * output  # type: ignore[operator]
        self.update_size = size

    def prepare_output_buffer(self, output: BlockLike[TranslationType]) -> None:
        """Allocate the specialized output buffer matching the current output size."""

        size = len(output)
        if self.output_size == size:
            return

        self.output_buffer = 0.0 * output  # type: ignore[operator]
        self.output_size = size

    def apply_diagonal_inverse(
        self,
        request: InverterRequest[TranslationType],
    ) -> None:
        """Store the diagonal inverse action on the current defect."""

        defect = self.defect.block
        update = self.update
        assert defect is not None
        assert update is not None
        operator = cast(BlockOperatorDiagonalLike[TranslationType], request.operator)

        if len(operator) != len(defect):
            raise ValueError("Jacobi operator size must match the defect size.")

        for index in range(len(defect)):
            entry = operator[index]
            if entry is None:
                raise RuntimeError(
                    f"Jacobi operator diagonal entry {index} is not configured."
                )
            self.diagonal_inverse(entry, defect[index], update[index])

    def call_inline(
        self,
        request: InverterRequest[TranslationType],
        output: BlockLike[TranslationType],
    ) -> tuple[bool, int, float, float]:
        self.prepare_update(output)
        residual_norm = request.residual.norm()

        # 1. Compute defect = request.residual - request.operator(output).
        initial_defect = self.defect(request, output)

        # 2. Accept if the defect norm is within tolerance.
        if self.tolerance.accepts(initial_defect, residual_norm):
            return True, 0, initial_defect, initial_defect

        final_defect = initial_defect
        for step in range(1, self.maximum_steps + 1):
            # 3. Apply each diagonal inverse to the corresponding defect entry.
            self.apply_diagonal_inverse(request)
            update = self.update
            assert update is not None

            # 4. Apply output <- output + damping * update.
            output.replace(output + self.damping * update)  # type: ignore[operator]

            # 5. Repeat until accepted or the step budget is exhausted.
            final_defect = self.defect(request, output)
            if self.tolerance.accepts(final_defect, residual_norm):
                return True, step, initial_defect, final_defect

        return False, self.maximum_steps, initial_defect, final_defect

    def call_specialized(
        self,
        request: InverterRequest[TranslationType],
        output: BlockLike[TranslationType],
    ) -> tuple[bool, int, float, float]:
        self.prepare_update(output)
        self.prepare_output_buffer(output)
        residual_norm = request.residual.norm()
        update_output = self.update_output
        output_buffer = self.output_buffer
        assert update_output is not None
        assert output_buffer is not None

        # 1. Compute defect = request.residual - request.operator(output).
        initial_defect = self.defect(request, output)

        # 2. Accept if the defect norm is within tolerance.
        if self.tolerance.accepts(initial_defect, residual_norm):
            return True, 0, initial_defect, initial_defect

        final_defect = initial_defect
        for step in range(1, self.maximum_steps + 1):
            # 3. Apply each diagonal inverse to the corresponding defect entry.
            self.apply_diagonal_inverse(request)
            update = self.update
            assert update is not None

            # 4. Apply output <- output + damping * update.
            update_output(1.0, output, update, output_buffer)
            output.replace(output_buffer)

            # 5. Repeat until accepted or the step budget is exhausted.
            final_defect = self.defect(request, output)
            if self.tolerance.accepts(final_defect, residual_norm):
                return True, step, initial_defect, final_defect

        return False, self.maximum_steps, initial_defect, final_defect


__all__ = ["InverterRelaxationJacobi", "InverterRelaxationJacobiInverse"]
