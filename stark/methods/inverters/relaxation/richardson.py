from __future__ import annotations

from typing import ClassVar, Generic

from stark.core.contracts import (
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
        "call_body",
        "damping",
        "defect",
        "maximum_steps",
        "monitor",
        "output_buffer",
        "output_size",
        "redirect_call",
        "tolerance",
        "update_output",
    )

    descriptor: ClassVar[InverterDescriptor] = InverterDescriptor("Richardson", "Richardson relaxation")
    output_mode: ClassVar[InverterOutputMode] = InverterOutputMode.improve
    # Installed by with_inverter_monitoring from stark.methods.inverters.support.
    record_solve: ClassVar[InverterRecordSolve]

    def __init__(
        self,
        *,
        damping: float = 1.0,
        configuration: InverterConfiguration | None = None,
        monitor: MonitorInverterLike | None = None,
        linear_fixed: InverterRelaxationLinearFixed[TranslationType] | None = None,
    ) -> None:
        if damping <= 0.0:
            raise ValueError("InverterRelaxationRichardson.damping must be positive.")

        configuration = configuration if configuration is not None else InverterConfigurationDefault()
        self.damping = damping
        self.tolerance = configuration.inverter_tolerance
        self.maximum_steps = configuration.inverter_maximum_steps
        self.monitor = monitor
        self.defect = InverterDefect[TranslationType]()
        self.output_buffer = None
        self.output_size = -1
        self.update_output = None
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

        # Step 3 applies output <- output + damping * defect.
        self.update_output = linear_fixed(
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
    ) -> tuple[bool, int, float, float]:
        residual_norm = request.residual.norm()

        # 1. Compute defect = request.residual - request.operator(output).
        initial_defect = self.defect(request, output)

        # 2. Accept if the defect norm is within tolerance.
        if self.tolerance.accepts(initial_defect, residual_norm):
            return True, 0, initial_defect, initial_defect

        final_defect = initial_defect
        for step in range(1, self.maximum_steps + 1):
            defect_block = self.defect.block
            assert defect_block is not None

            # 3. Apply output <- output + damping * defect.
            output.replace(output + self.damping * defect_block)  # type: ignore[operator]

            # 4. Repeat until accepted or the step budget is exhausted.
            final_defect = self.defect(request, output)
            if self.tolerance.accepts(final_defect, residual_norm):
                return True, step, initial_defect, final_defect

        return False, self.maximum_steps, initial_defect, final_defect

    def call_specialized(
        self,
        request: InverterRequest[TranslationType],
        output: BlockLike[TranslationType],
    ) -> tuple[bool, int, float, float]:
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
            defect_block = self.defect.block
            assert defect_block is not None

            # 3. Apply output <- output + damping * defect.
            update_output(1.0, output, defect_block, output_buffer)
            output.replace(output_buffer)

            # 4. Repeat until accepted or the step budget is exhausted.
            final_defect = self.defect(request, output)
            if self.tolerance.accepts(final_defect, residual_norm):
                return True, step, initial_defect, final_defect

        return False, self.maximum_steps, initial_defect, final_defect


__all__ = ["InverterRelaxationRichardson"]
