from __future__ import annotations

from stark.accelerators.binding import BoundDerivative, BoundLinearizer
from stark.contracts import Block, Derivative, IntervalLike, Linearizer, State
from stark.machinery.stage_solve.workspace import SchemeWorkspace


class StageJacobianOperator:
    """Mutable Jacobian action configured by a linearizer for one stage solve."""

    __slots__ = ("apply", "method_name")

    def __init__(self, method_name: str) -> None:
        self.method_name = method_name
        self.apply = self._unconfigured

    def __call__(self, translation, out) -> None:
        self.apply(translation, out)

    def _unconfigured(self, translation, out) -> None:
        del translation, out
        raise RuntimeError(f"{self.method_name} Jacobian operator was used before the linearizer configured it.")


class StageResidualOperator:
    """Linearized stage operator `I - alpha J` for one stage residual."""

    __slots__ = ("combine2", "jacobian_buffer", "jacobian", "alpha")

    def __init__(self, workspace: SchemeWorkspace, jacobian: StageJacobianOperator) -> None:
        self.combine2 = workspace.combine2
        self.jacobian_buffer = workspace.allocate_translation()
        self.jacobian = jacobian
        self.alpha = 0.0

    def __call__(self, translation, out) -> None:
        self.jacobian(translation, self.jacobian_buffer)
        self.combine2(out, 1.0, translation, -self.alpha, self.jacobian_buffer)


class CoupledStageResidualOperator:
    """Linearized block operator for a fully coupled implicit RK stage system."""

    __slots__ = ("scale", "combine2", "jacobian_buffer", "jacobians", "matrix", "step")

    def __init__(self, workspace: SchemeWorkspace, jacobians: list[StageJacobianOperator], matrix: tuple[tuple[float, ...], ...]) -> None:
        self.scale = workspace.scale
        self.combine2 = workspace.combine2
        self.jacobian_buffer = workspace.allocate_translation()
        self.jacobians = jacobians
        self.matrix = matrix
        self.step = 0.0

    def reset(self) -> None:
        self.step = 0.0
        for jacobian in self.jacobians:
            jacobian.apply = jacobian._unconfigured

    def __call__(self, block: Block, out: Block) -> None:
        for row_index, row in enumerate(self.matrix):
            out_item = self.scale(out[row_index], 0.0, out[row_index])
            for column_index, coefficient in enumerate(row):
                if row_index == column_index:
                    out_item = self.combine2(out_item, 1.0, out_item, 1.0, block[column_index])
                if coefficient == 0.0:
                    continue
                self.jacobians[column_index](block[column_index], self.jacobian_buffer)
                out_item = self.combine2(out_item, 1.0, out_item, -self.step * coefficient, self.jacobian_buffer)
            out.items[row_index] = out_item


class StageResidual:
    """
    Residual worker for shifted implicit stage equations.

    The nonlinear problem is

        delta - rhs - alpha f(state + delta) = 0

    with optional right-hand side shift `rhs`. This is the common one-stage
    shape behind backward Euler, sequential DIRK stages, and similar implicit
    corrections. When a linearizer is supplied, the Newton operator is
    `I - alpha J`.
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
        "_linearize",
    )

    def __init__(
        self,
        method_name: str,
        derivative: Derivative,
        workspace: SchemeWorkspace,
        linearizer: Linearizer | None = None,
    ) -> None:
        self.method_name = method_name
        self.scale = workspace.scale
        self.combine2 = workspace.combine2
        self.combine3 = workspace.combine3
        self.copy_state = workspace.copy_state
        self.base_state = workspace.allocate_state_buffer()
        self.interval: IntervalLike | None = None
        self.trial_state = workspace.allocate_state_buffer()
        self.rhs = workspace.allocate_translation()
        self.derivative = BoundDerivative(derivative)
        self.derivative_buffer = workspace.allocate_translation()
        self.alpha = 0.0
        self.linearizer = BoundLinearizer(linearizer) if linearizer is not None else None
        self.jacobian_operator = StageJacobianOperator(method_name)
        self.residual_operator = StageResidualOperator(workspace, self.jacobian_operator)
        self._linearize = self._linearize_configured if linearizer is not None else self._linearize_missing

    def configure(self, interval: IntervalLike, state: State, alpha: float, rhs: Block | None = None) -> None:
        self.interval = interval
        self.copy_state(self.base_state, state)
        self.alpha = alpha
        if rhs is None:
            self.rhs = self.scale(self.rhs, 0.0, self.rhs)
            return
        self.rhs = self.combine2(self.rhs, 0.0, self.rhs, 1.0, rhs[0])

    def __call__(self, block: Block, out: Block) -> None:
        interval = self.interval
        assert interval is not None
        delta = block[0]
        delta(self.base_state, self.trial_state)
        self.derivative(interval, self.trial_state, self.derivative_buffer)
        out.items[0] = self.combine3(out[0], 1.0, delta, -1.0, self.rhs, -self.alpha, self.derivative_buffer)

    def linearize(self, block: Block, out) -> None:
        self._linearize(block, out)

    def _linearize_missing(self, block: Block, out) -> None:
        del block, out
        raise RuntimeError(f"{self.method_name} Newton resolution requires a linearizer.")

    def _linearize_configured(self, block: Block, out) -> None:
        linearizer = self.linearizer
        assert linearizer is not None
        interval = self.interval
        assert interval is not None
        block[0](self.base_state, self.trial_state)
        linearizer(interval, self.trial_state, self.jacobian_operator)
        self.residual_operator.alpha = self.alpha
        out.operators[0] = self.residual_operator


class CoupledStageResidual:
    """
    Residual worker for fully coupled implicit Runge-Kutta stage systems.

    For a tableau with coefficients `a_ij` and stage block `delta = (delta_i)`,
    the nonlinear problem is

        delta_i - rhs_i - dt * sum_j a_ij f(t_n + c_j dt, state + delta_j) = 0.

    This is the coupled stage shape behind fully implicit Gauss, Lobatto, and
    Radau-type Runge-Kutta methods.
    """

    __slots__ = (
        "method_name",
        "stage_count",
        "stage_shifts",
        "matrix",
        "scale",
        "combine2",
        "copy_state",
        "base_state",
        "stage_states",
        "stage_intervals",
        "rhs_block",
        "derivative",
        "derivative_buffers",
        "linearizer",
        "jacobian_operators",
        "residual_operator",
        "block_operator",
        "_linearize",
        "step",
    )

    def __init__(
        self,
        method_name: str,
        derivative: Derivative,
        workspace: SchemeWorkspace,
        stage_shifts: tuple[float, ...],
        matrix: tuple[tuple[float, ...], ...],
        linearizer: Linearizer | None = None,
    ) -> None:
        self.method_name = method_name
        self.stage_count = len(stage_shifts)
        self.stage_shifts = stage_shifts
        self.matrix = matrix
        self.scale = workspace.scale
        self.combine2 = workspace.combine2
        self.copy_state = workspace.copy_state
        self.base_state = workspace.allocate_state_buffer()
        self.stage_states = [workspace.allocate_state_buffer() for _ in range(self.stage_count)]
        self.stage_intervals: list[IntervalLike | None] = [None for _ in range(self.stage_count)]
        self.rhs_block = Block([workspace.allocate_translation() for _ in range(self.stage_count)])
        self.derivative = BoundDerivative(derivative)
        self.derivative_buffers = [workspace.allocate_translation() for _ in range(self.stage_count)]
        self.linearizer = BoundLinearizer(linearizer) if linearizer is not None else None
        self.jacobian_operators = [StageJacobianOperator(f"{method_name}[stage {index}]") for index in range(self.stage_count)]
        self.residual_operator = CoupledStageResidualOperator(workspace, self.jacobian_operators, matrix)
        self.block_operator = self.residual_operator
        self._linearize = self._linearize_configured if linearizer is not None else self._linearize_missing
        self.step = 0.0

    def configure(self, interval: IntervalLike, state: State, step: float, rhs: Block | None = None) -> None:
        self.copy_state(self.base_state, state)
        self.step = step
        self.residual_operator.step = step
        for index, shift in enumerate(self.stage_shifts):
            stage_interval = self.stage_intervals[index]
            if stage_interval is None:
                stage_interval = interval.copy()
                self.stage_intervals[index] = stage_interval
            stage_interval.present = interval.present + shift * step
            stage_interval.step = step
            stage_interval.stop = interval.stop
        if rhs is None:
            for index, item in enumerate(self.rhs_block):
                self.rhs_block.items[index] = self.scale(item, 0.0, item)
            return
        if len(rhs) != self.stage_count:
            raise ValueError(f"rhs must have {self.stage_count} items for {self.method_name}.")
        for index, item in enumerate(self.rhs_block):
            self.rhs_block.items[index] = self.combine2(item, 0.0, item, 1.0, rhs[index])

    def __call__(self, block: Block, out: Block) -> None:
        if len(block) != self.stage_count or len(out) != self.stage_count:
            raise ValueError(f"{self.method_name} expects {self.stage_count}-item stage blocks.")

        for index, delta in enumerate(block):
            delta(self.base_state, self.stage_states[index])
            interval = self.stage_intervals[index]
            assert interval is not None
            self.derivative(interval, self.stage_states[index], self.derivative_buffers[index])

        for row_index, row in enumerate(self.matrix):
            out_item = self.combine2(out[row_index], 1.0, block[row_index], -1.0, self.rhs_block[row_index])
            for column_index, coefficient in enumerate(row):
                if coefficient == 0.0:
                    continue
                out_item = self.combine2(
                    out_item,
                    1.0,
                    out_item,
                    -self.step * coefficient,
                    self.derivative_buffers[column_index],
                )
            out.items[row_index] = out_item

    def linearize(self, block: Block, out) -> None:
        self._linearize(block, out)

    def _linearize_missing(self, block: Block, out) -> None:
        del block, out
        raise RuntimeError(f"{self.method_name} Newton resolution requires a linearizer.")

    def _linearize_configured(self, block: Block, out) -> None:
        linearizer = self.linearizer
        assert linearizer is not None
        if len(block) != self.stage_count:
            raise ValueError(f"{self.method_name} expects {self.stage_count}-item stage blocks.")
        out.reset()
        out.step = self.step
        for index, delta in enumerate(block):
            delta(self.base_state, self.stage_states[index])
            interval = self.stage_intervals[index]
            assert interval is not None
            linearizer(interval, self.stage_states[index], self.jacobian_operators[index])


__all__ = [
    "CoupledStageResidual",
    "CoupledStageResidualOperator",
    "StageJacobianOperator",
    "StageResidual",
    "StageResidualOperator",
]












