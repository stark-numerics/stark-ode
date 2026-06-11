from __future__ import annotations

from collections.abc import MutableSequence
from typing import cast

from stark.algebraist.arity import AlgebraistArity
from stark.algebraist.runtime.linear_combine import AlgebraistRuntimeLinearCombine
from stark.block import Block, BlockOperatorDiagonal
from stark.contracts import Accelerator, DerivativeLike, IntervalLike, Linearizer, State, Translation, Allocator
from stark.contracts.translation_basis import TranslationBasis
from stark.methods.resolvents.requests.resolvent import (
    ResolventRequestCoupled,
    ResolventRequest,
)
from stark.methods.resolvents.equations.workers import ResolventDerivative, ResolventLinearizer


class ResolventImplicitEquationJacobian:
    """Mutable Jacobian action configured by a linearizer."""

    __slots__ = ("apply", "dense_fill", "method_name")

    def __init__(self, method_name: str) -> None:
        self.method_name = method_name
        self.apply = self._unconfigured
        self.dense_fill = None

    def __call__(self, translation: Translation, out: Translation) -> None:
        self.apply(translation, out)

    def _unconfigured(self, translation: Translation, out: Translation) -> None:
        del translation, out
        raise RuntimeError(
            f"{self.method_name} Jacobian operator was used before the "
            "linearizer configured it."
        )


class ResolventImplicitEquationDifferential:
    """Linearized one-stage implicit equation action ``I - alpha J``."""

    __slots__ = (
        "combine2",
        "dense_fill",
        "dense_fill_direct",
        "jacobian_buffer",
        "jacobian",
        "alpha",
    )

    def __init__(self, combine2, allocate_translation, jacobian) -> None:
        self.combine2 = combine2
        self.dense_fill = None
        self.dense_fill_direct = self.dense_fill_configured
        self.jacobian_buffer = allocate_translation()
        self.jacobian = jacobian
        self.alpha = 0.0

    def __call__(self, translation: Translation, out: Translation) -> None:
        self.jacobian(translation, self.jacobian_buffer)
        self.combine2(1.0, translation, -self.alpha, self.jacobian_buffer, out)

    def dense_fill_configured(
        self,
        basis: TranslationBasis[Translation],
        matrix: MutableSequence[float],
        row_offset: int,
        column_offset: int,
        stride: int,
    ) -> None:
        jacobian_dense_fill = self.jacobian.dense_fill
        assert jacobian_dense_fill is not None

        jacobian_dense_fill(basis, matrix, row_offset, column_offset, stride)

        dimension = basis.dimension
        alpha = self.alpha
        for local_column in range(dimension):
            column = column_offset + local_column
            for local_row in range(dimension):
                row = row_offset + local_row
                matrix[row * stride + column] *= -alpha

        for local_index in range(dimension):
            row = row_offset + local_index
            column = column_offset + local_index
            matrix[row * stride + column] += 1.0


class ResolventImplicitEquationDifferentialCoupled:
    """Linearized block action for coupled implicit equations."""

    __slots__ = (
        "scale",
        "combine2",
        "jacobian_buffer",
        "jacobians",
        "matrix",
        "step",
    )

    def __init__(
        self,
        scale,
        combine2,
        allocate_translation,
        jacobians: list[ResolventImplicitEquationJacobian],
        matrix: tuple[tuple[float, ...], ...],
    ) -> None:
        self.scale = scale
        self.combine2 = combine2
        self.jacobian_buffer = allocate_translation()
        self.jacobians = jacobians
        self.matrix = matrix
        self.step = 0.0

    def reset(self) -> None:
        self.step = 0.0
        for jacobian in self.jacobians:
            jacobian.apply = jacobian._unconfigured
            jacobian.dense_fill = None

    def __call__(self, block: Block[Translation], out: Block[Translation]) -> None:
        for row_index, row in enumerate(self.matrix):
            out_item = self.scale(0.0, out[row_index], out[row_index])

            for column_index, coefficient in enumerate(row):
                if row_index == column_index:
                    out_item = self.combine2(
                        1.0,
                        out_item,
                        1.0,
                        block[column_index],
                        out_item,
                    )

                if coefficient == 0.0:
                    continue

                self.jacobians[column_index](
                    block[column_index],
                    self.jacobian_buffer,
                )
                out_item = self.combine2(
                    1.0,
                    out_item,
                    -self.step * coefficient,
                    self.jacobian_buffer,
                    out_item,
                )

            out[row_index] = out_item


class ResolventImplicitEquation:
    """Reusable one-stage implicit equation evaluator.

    For a prepared ``ResolventRequest`` this evaluates:

        F(delta) = delta - rhs - alpha * f(interval, origin + delta)
    """

    __slots__ = (
        "method_name",
        "scale",
        "combine2",
        "combine3",
        "copy_state",
        "base_state",
        "interval",
        "trial_state",
        "rhs",
        "derivative",
        "derivative_buffer",
        "alpha",
        "linearizer",
        "jacobian_operator",
        "residual_operator",
        "_differential",
    )

    def __init__(
        self,
        method_name: str,
        allocator: Allocator,
        linearizer: Linearizer | None = None,
        accelerator: Accelerator | None = None,
    ) -> None:
        self.method_name = method_name
        translation_probe = allocator.allocate_translation()
        general = AlgebraistRuntimeLinearCombine(
            translation=translation_probe,
            allocator=allocator,
            accelerator=accelerator,
        )
        self.scale = general.provide(AlgebraistArity(1))
        self.combine2 = general.provide(AlgebraistArity(2))
        self.combine3 = general.provide(AlgebraistArity(3))

        self.copy_state = allocator.copy_state
        self.base_state = allocator.allocate_state()
        self.interval: IntervalLike | None = None
        self.trial_state = allocator.allocate_state()
        self.rhs = allocator.allocate_translation()
        self.derivative: DerivativeLike | None = None
        self.derivative_buffer = allocator.allocate_translation()
        self.alpha = 0.0

        self.linearizer = (
            ResolventLinearizer(linearizer) if linearizer is not None else None
        )
        self.jacobian_operator = ResolventImplicitEquationJacobian(method_name)
        self.residual_operator = ResolventImplicitEquationDifferential(
            self.combine2,
            allocator.allocate_translation,
            self.jacobian_operator,
        )
        self._differential = (
            self._differential_configured
            if linearizer is not None
            else self._differential_missing
        )

    def prepare(self, problem: ResolventRequest) -> "ResolventImplicitEquation":
        self.interval = problem.interval
        self.copy_state(problem.origin, self.base_state)
        self.alpha = problem.alpha
        self.derivative = ResolventDerivative(problem.derivative)

        if problem.rhs is None:
            self.rhs = self.scale(0.0, self.rhs, self.rhs)
            return self

        self.rhs = self.combine2(0.0, self.rhs, 1.0, problem.rhs[0], self.rhs)
        return self

    def __call__(
        self,
        block: Block[Translation],
        out: Block[Translation],
    ) -> None:
        interval = self.interval
        derivative = self.derivative
        assert interval is not None
        assert derivative is not None

        delta = block[0]
        delta(self.base_state, self.trial_state)
        derivative(interval, self.trial_state, self.derivative_buffer)
        out[0] = self.combine3(
            1.0,
            delta,
            -1.0,
            self.rhs,
            -self.alpha,
            self.derivative_buffer,
            out[0],
        )

    def differential(self, block: Block[Translation], out) -> None:
        self._differential(block, out)

    # Backward-compatible alias while Newton callers are migrated.
    def linearize(self, block: Block[Translation], out) -> None:
        self.differential(block, out)

    def _differential_missing(self, block: Block[Translation], out) -> None:
        del block, out
        raise RuntimeError(
            f"{self.method_name} Newton resolution requires a linearizer."
        )

    def _differential_configured(self, block: Block[Translation], out) -> None:
        linearizer = self.linearizer
        interval = self.interval
        assert linearizer is not None
        assert interval is not None

        block[0](self.base_state, self.trial_state)
        self.jacobian_operator.dense_fill = None
        linearizer(interval, self.trial_state, self.jacobian_operator)
        self.residual_operator.alpha = self.alpha
        self.residual_operator.dense_fill = (
            self.residual_operator.dense_fill_direct
            if self.jacobian_operator.dense_fill is not None
            else None
        )
        out[0] = self.residual_operator


class ResolventImplicitEquationCoupled:
    """Reusable coupled implicit equation evaluator."""

    __slots__ = (
        "method_name",
        "scale",
        "combine2",
        "copy_state",
        "allocator",
        "base_state",
        "stage_count",
        "stage_shifts",
        "matrix",
        "stage_states",
        "stage_intervals",
        "rhs_block",
        "derivative",
        "derivative_buffers",
        "linearizer",
        "jacobian_operators",
        "residual_operator",
        "block_operator",
        "_differential",
        "step",
    )

    def __init__(
        self,
        method_name: str,
        allocator: Allocator,
        linearizer: Linearizer | None = None,
        accelerator: Accelerator | None = None,
    ) -> None:
        self.method_name = method_name
        translation_probe = allocator.allocate_translation()
        general = AlgebraistRuntimeLinearCombine(
            translation=translation_probe,
            allocator=allocator,
            accelerator=accelerator,
        )
        self.scale = general.provide(AlgebraistArity(1))
        self.combine2 = general.provide(AlgebraistArity(2))
        self.copy_state = allocator.copy_state
        self.allocator = allocator
        self.base_state = allocator.allocate_state()

        self.stage_count = 0
        self.stage_shifts: tuple[float, ...] = ()
        self.matrix: tuple[tuple[float, ...], ...] = ()
        self.stage_states: list[State] = []
        self.stage_intervals: list[IntervalLike | None] = []
        self.rhs_block = Block([])
        self.derivative: DerivativeLike | None = None
        self.derivative_buffers: list[Translation] = []
        self.linearizer = (
            ResolventLinearizer(linearizer) if linearizer is not None else None
        )
        self.jacobian_operators: list[ResolventImplicitEquationJacobian] = []
        self.residual_operator: ResolventImplicitEquationDifferentialCoupled | None = None
        self.block_operator = None
        self._differential = (
            self._differential_configured
            if linearizer is not None
            else self._differential_missing
        )
        self.step = 0.0

    def prepare(self, problem: ResolventRequestCoupled) -> "ResolventImplicitEquationCoupled":
        self._ensure_stage_count(len(problem.stage_shifts))

        self.copy_state(problem.origin, self.base_state)
        self.stage_shifts = problem.stage_shifts
        self.matrix = problem.matrix
        self.step = problem.step
        self.derivative = ResolventDerivative(problem.derivative)

        residual_operator = self.residual_operator
        assert residual_operator is not None
        residual_operator.matrix = problem.matrix
        residual_operator.step = problem.step

        for index, shift in enumerate(problem.stage_shifts):
            stage_interval = self.stage_intervals[index]
            if stage_interval is None:
                stage_interval = problem.interval.copy()
                self.stage_intervals[index] = stage_interval

            stage_interval.present = problem.interval.present + shift * problem.step
            stage_interval.step = problem.step
            stage_interval.stop = problem.interval.stop

        if problem.rhs is None:
            for index, item in enumerate(self.rhs_block):
                self.rhs_block[index] = self.scale(0.0, item, item)
            return self

        if len(problem.rhs) != self.stage_count:
            raise ValueError(
                f"rhs must have {self.stage_count} items for {self.method_name}."
            )

        for index, item in enumerate(self.rhs_block):
            self.rhs_block[index] = self.combine2(
                0.0,
                item,
                1.0,
                problem.rhs[index],
                item,
            )
        return self

    def __call__(
        self,
        block: Block[Translation],
        out: Block[Translation],
    ) -> None:
        if len(block) != self.stage_count or len(out) != self.stage_count:
            raise ValueError(
                f"{self.method_name} expects {self.stage_count}-item stage blocks."
            )

        derivative = self.derivative
        assert derivative is not None

        for index, delta in enumerate(block):
            delta(self.base_state, self.stage_states[index])
            interval = self.stage_intervals[index]
            assert interval is not None
            derivative(interval, self.stage_states[index], self.derivative_buffers[index])

        for row_index, row in enumerate(self.matrix):
            out_item = self.combine2(
                1.0,
                block[row_index],
                -1.0,
                self.rhs_block[row_index],
                out[row_index],
            )

            for column_index, coefficient in enumerate(row):
                if coefficient == 0.0:
                    continue

                out_item = self.combine2(
                    1.0,
                    out_item,
                    -self.step * coefficient,
                    self.derivative_buffers[column_index],
                    out_item,
                )

            out[row_index] = out_item

    def differential(self, block: Block[Translation], out) -> None:
        self._differential(block, out)

    def linearize(self, block: Block[Translation], out) -> None:
        self.differential(block, out)

    def _differential_missing(self, block: Block[Translation], out) -> None:
        del block, out
        raise RuntimeError(
            f"{self.method_name} Newton resolution requires a linearizer."
        )

    def _differential_configured(self, block: Block[Translation], out) -> None:
        linearizer = self.linearizer
        assert linearizer is not None

        if len(block) != self.stage_count:
            raise ValueError(
                f"{self.method_name} expects {self.stage_count}-item stage blocks."
            )

        out.reset()
        residual_operator = self.residual_operator
        assert residual_operator is not None
        residual_operator.step = self.step

        for index, delta in enumerate(block):
            delta(self.base_state, self.stage_states[index])
            interval = self.stage_intervals[index]
            assert interval is not None
            self.jacobian_operators[index].dense_fill = None
            linearizer(interval, self.stage_states[index], self.jacobian_operators[index])

    def _ensure_stage_count(self, stage_count: int) -> None:
        if self.stage_count == stage_count:
            return

        allocator = self.allocator
        self.stage_count = stage_count
        self.stage_states = [allocator.allocate_state() for _ in range(stage_count)]
        self.stage_intervals = [None for _ in range(stage_count)]
        self.rhs_block = Block(
            [allocator.allocate_translation() for _ in range(stage_count)]
        )
        self.derivative_buffers = [
            allocator.allocate_translation() for _ in range(stage_count)
        ]
        self.jacobian_operators = [
            ResolventImplicitEquationJacobian(f"{self.method_name}[stage {index}]")
            for index in range(stage_count)
        ]
        self.residual_operator = ResolventImplicitEquationDifferentialCoupled(
            self.scale,
            self.combine2,
            allocator.allocate_translation,
            self.jacobian_operators,
            self.matrix,
        )
        self.block_operator = self.residual_operator


__all__ = [
    "ResolventImplicitEquationCoupled",
    "ResolventImplicitEquationDifferentialCoupled",
    "ResolventImplicitEquationJacobian",
    "ResolventImplicitEquation",
    "ResolventImplicitEquationDifferential",
]
