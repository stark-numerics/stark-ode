from __future__ import annotations

from typing import Generic

from stark.contracts import BlockLike, InverterRequest, TranslationType
from stark.inverters.relaxation.specialist import InverterRelaxationSpecialist
from stark.inverters.relaxation.stencil import InverterRelaxationStencilUpdate
from stark.inverters.support import (
    InverterBudget,
    InverterDefect,
    InverterDescriptor,
    InverterTolerance,
    MonitorInverterLike,
    with_inverter_monitoring,
)


# Optional extension: records inverter monitor events.
# Provides: record_solve.
@with_inverter_monitoring
class InverterRelaxationRichardson(Generic[TranslationType]):
    """
    Richardson relaxation inverter.

    Richardson relaxation is the simplest stationary inverter for a linear
    correction request. It repeatedly corrects the current output by a damped
    multiple of the current defect.

    Problem:
        Improve output so that request.operator(output) = request.residual.

    Algorithm:
        1. Compute defect = request.residual - request.operator(output).
        2. Accept if the defect norm is within tolerance.
        3. Apply output <- output + damping * defect.
        4. Repeat until accepted or the step budget is exhausted.
    """

    __slots__ = (
        "budget",
        "call_body",
        "damping",
        "defect",
        "monitor",
        "output_buffer",
        "output_size",
        "tolerance",
        "update_output",
    )

    descriptor = InverterDescriptor("Richardson", "Richardson relaxation")

    def __init__(
        self,
        *,
        damping: float = 1.0,
        tolerance: InverterTolerance | None = None,
        budget: InverterBudget | None = None,
        monitor: MonitorInverterLike | None = None,
        specialist: InverterRelaxationSpecialist[TranslationType] | None = None,
    ) -> None:
        if damping <= 0.0:
            raise ValueError("InverterRelaxationRichardson.damping must be positive.")

        self.damping = damping
        self.tolerance = tolerance if tolerance is not None else InverterTolerance()
        self.budget = budget if budget is not None else InverterBudget()
        self.monitor = monitor
        self.defect = InverterDefect[TranslationType]()
        self.output_buffer = None
        self.output_size = -1
        self.update_output = None
        self.call_body = self.call_inline

        if specialist is not None:
            self.prepare_specialized_kernels(specialist)
            self.call_body = self.call_specialized

    def __call__(
        self,
        request: InverterRequest[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        return self.call_body(request, output)

    def prepare_specialized_kernels(
        self,
        specialist: InverterRelaxationSpecialist[TranslationType],
    ) -> None:
        """Prepare the relaxation update kernel for the specialized path."""

        # Step 3 applies output <- output + damping * defect.
        self.update_output = specialist.provide(
            InverterRelaxationStencilUpdate(self.damping)
        )

    def prepare_output_buffer(self, output: BlockLike[TranslationType]) -> None:
        """Allocate the specialized output buffer matching the current output size."""

        size = len(output)
        if self.output_size == size:
            return

        self.output_buffer = 0.0 * output  # type: ignore[operator]
        self.output_size = size

    def call_inline(
        self,
        request: InverterRequest[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        residual_norm = request.residual.norm()

        # 1. Compute defect = request.residual - request.operator(output).
        initial_defect = self.defect(request, output)

        # 2. Accept if the defect norm is within tolerance.
        if self.tolerance.accepts(initial_defect, residual_norm):
            self.record_solve(
                converged=True,
                iteration_count=0,
                initial_residual=initial_defect,
                final_residual=initial_defect,
            )
            return

        final_defect = initial_defect
        for step in range(1, self.budget.maximum_steps + 1):
            defect_block = self.defect.block
            assert defect_block is not None

            # 3. Apply output <- output + damping * defect.
            output.replace(output + self.damping * defect_block)  # type: ignore[operator]

            # 4. Repeat until accepted or the step budget is exhausted.
            final_defect = self.defect(request, output)
            if self.tolerance.accepts(final_defect, residual_norm):
                self.record_solve(
                    converged=True,
                    iteration_count=step,
                    initial_residual=initial_defect,
                    final_residual=final_defect,
                )
                return

        self.record_solve(
            converged=False,
            iteration_count=self.budget.maximum_steps,
            initial_residual=initial_defect,
            final_residual=final_defect,
            failure_reason="maximum steps reached",
        )

    def call_specialized(
        self,
        request: InverterRequest[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
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
            self.record_solve(
                converged=True,
                iteration_count=0,
                initial_residual=initial_defect,
                final_residual=initial_defect,
            )
            return

        final_defect = initial_defect
        for step in range(1, self.budget.maximum_steps + 1):
            defect_block = self.defect.block
            assert defect_block is not None

            # 3. Apply output <- output + damping * defect.
            update_output(1.0, output, defect_block, output_buffer)
            output.replace(output_buffer)

            # 4. Repeat until accepted or the step budget is exhausted.
            final_defect = self.defect(request, output)
            if self.tolerance.accepts(final_defect, residual_norm):
                self.record_solve(
                    converged=True,
                    iteration_count=step,
                    initial_residual=initial_defect,
                    final_residual=final_defect,
                )
                return

        self.record_solve(
            converged=False,
            iteration_count=self.budget.maximum_steps,
            initial_residual=initial_defect,
            final_residual=final_defect,
            failure_reason="maximum steps reached",
        )


__all__ = ["InverterRelaxationRichardson"]
