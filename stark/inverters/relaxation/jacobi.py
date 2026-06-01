from __future__ import annotations

from collections.abc import Callable
from typing import Generic

from stark.contracts import (
    BlockOperatorEntryLike,
    BlockLike,
    InverterRequest,
    TranslationType,
)
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
        "budget",
        "call_body",
        "damping",
        "defect",
        "diagonal_inverse",
        "monitor",
        "output_buffer",
        "output_size",
        "tolerance",
        "update",
        "update_output",
        "update_size",
    )

    descriptor = InverterDescriptor("Jacobi", "Jacobi relaxation")

    def __init__(
        self,
        diagonal_inverse: InverterRelaxationJacobiInverse[TranslationType],
        *,
        damping: float = 1.0,
        tolerance: InverterTolerance | None = None,
        budget: InverterBudget | None = None,
        monitor: MonitorInverterLike | None = None,
        specialist: InverterRelaxationSpecialist[TranslationType] | None = None,
    ) -> None:
        if damping <= 0.0:
            raise ValueError("InverterRelaxationJacobi.damping must be positive.")

        self.diagonal_inverse = diagonal_inverse
        self.damping = damping
        self.tolerance = tolerance if tolerance is not None else InverterTolerance()
        self.budget = budget if budget is not None else InverterBudget()
        self.monitor = monitor
        self.defect = InverterDefect[TranslationType]()
        self.output_buffer = None
        self.output_size = -1
        self.update: BlockLike[TranslationType] | None = None
        self.update_output = None
        self.update_size = -1
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

        # Step 4 applies output <- output + damping * update.
        self.update_output = specialist.provide(
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

        if len(request.operator) != len(defect):
            raise ValueError("Jacobi operator size must match the defect size.")

        for index in range(len(defect)):
            operator = request.operator[index]
            if operator is None:
                raise RuntimeError(
                    f"Jacobi operator diagonal entry {index} is not configured."
                )
            self.diagonal_inverse(operator, defect[index], update[index])

    def call_inline(
        self,
        request: InverterRequest[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        self.prepare_update(output)
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
            # 3. Apply each diagonal inverse to the corresponding defect entry.
            self.apply_diagonal_inverse(request)
            update = self.update
            assert update is not None

            # 4. Apply output <- output + damping * update.
            output.replace(output + self.damping * update)  # type: ignore[operator]

            # 5. Repeat until accepted or the step budget is exhausted.
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
            self.record_solve(
                converged=True,
                iteration_count=0,
                initial_residual=initial_defect,
                final_residual=initial_defect,
            )
            return

        final_defect = initial_defect
        for step in range(1, self.budget.maximum_steps + 1):
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


__all__ = ["InverterRelaxationJacobi", "InverterRelaxationJacobiInverse"]
