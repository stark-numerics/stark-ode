from __future__ import annotations

from stark.audit import Auditor
from stark.butcher_tableau import ButcherTableau
from stark.control import Tolerance
from stark.contracts import Block, Derivative, IntervalLike, Linearizer, ResolverLike, State, Workbench
from stark.resolver_library.picard import ResolverPicard
from stark.scheme_support.descriptor import SchemeDescriptor
from stark.scheme_support.workspace import SchemeWorkspace


BE_TABLEAU = ButcherTableau(
    c=(1.0,),
    a=((1.0,),),
    b=(1.0,),
    order=1,
)


class _BackwardEulerResidual:
    __slots__ = (
        "combine2",
        "copy_state",
        "base_state",
        "trial_state",
        "derivative",
        "derivative_buffer",
        "dt",
        "linearizer",
        "jacobian_operator",
        "residual_operator",
    )

    def __init__(self, derivative: Derivative, workspace: SchemeWorkspace, linearizer: Linearizer | None = None) -> None:
        self.combine2 = workspace.combine2
        self.copy_state = workspace.copy_state
        self.base_state = workspace.allocate_state_buffer()
        self.trial_state = workspace.allocate_state_buffer()
        self.derivative = derivative
        self.derivative_buffer = workspace.allocate_translation()
        self.dt = 0.0
        self.linearizer = linearizer
        self.jacobian_operator = _BackwardEulerJacobianOperator(workspace)
        self.residual_operator = _BackwardEulerResidualOperator(workspace, self.jacobian_operator)

    def configure(self, state: State, dt: float) -> None:
        self.copy_state(self.base_state, state)
        self.dt = dt

    def __call__(self, out: Block, block: Block) -> None:
        if len(out) != 1 or len(block) != 1:
            raise ValueError("Backward Euler residual expects one-translation blocks.")

        delta = block[0]
        delta(self.base_state, self.trial_state)
        self.derivative(self.trial_state, self.derivative_buffer)
        out.items[0] = self.combine2(out[0], 1.0, delta, -self.dt, self.derivative_buffer)

    def linearize(self, out, block: Block) -> None:
        if len(block) != 1:
            raise ValueError("Backward Euler residual expects one-translation blocks.")
        if self.linearizer is None:
            raise RuntimeError("Backward Euler Newton resolution requires a linearizer.")
        if len(out.operators) != 1:
            raise ValueError("Backward Euler linearization expects a one-operator block.")

        block[0](self.base_state, self.trial_state)
        self.linearizer(self.jacobian_operator, self.trial_state)
        self.residual_operator.dt = self.dt
        self.residual_operator.jacobian = self.jacobian_operator
        out.operators[0] = self.residual_operator


class _BackwardEulerJacobianOperator:
    __slots__ = ("apply",)

    def __init__(self, workspace: SchemeWorkspace) -> None:
        del workspace
        self.apply = _unconfigured_operator

    def __call__(self, out, translation) -> None:
        self.apply(out, translation)


class _BackwardEulerResidualOperator:
    __slots__ = ("combine2", "jacobian_buffer", "jacobian", "dt")

    def __init__(self, workspace: SchemeWorkspace, jacobian) -> None:
        self.combine2 = workspace.combine2
        self.jacobian_buffer = workspace.allocate_translation()
        self.jacobian = jacobian
        self.dt = 0.0

    def __call__(self, out, translation) -> None:
        self.jacobian(self.jacobian_buffer, translation)
        return self.combine2(out, 1.0, translation, -self.dt, self.jacobian_buffer)


def _unconfigured_operator(out, translation) -> None:
    del out, translation
    raise RuntimeError("Backward Euler Jacobian operator was used before the linearizer configured it.")


class SchemeBackwardEuler:
    """Implicit backward Euler scheme resolved by a nonlinear residual solver."""

    __slots__ = ("derivative", "resolver", "residual", "trial_block", "workspace")

    descriptor = SchemeDescriptor("BE", "Backward Euler")
    tableau = BE_TABLEAU

    def __init__(
        self,
        derivative: Derivative,
        workbench: Workbench,
        linearizer: Linearizer | None = None,
        resolver: ResolverLike | None = None,
    ) -> None:
        translation_probe = workbench.allocate_translation()
        Auditor.require_scheme_inputs(derivative, workbench, translation_probe)
        self.derivative = derivative
        self.workspace = SchemeWorkspace(workbench, translation_probe)
        self.resolver = resolver if resolver is not None else ResolverPicard(workbench)
        self.trial_block = Block([translation_probe])
        self.residual = _BackwardEulerResidual(derivative, self.workspace, linearizer=linearizer)

    @classmethod
    def display_tableau(cls) -> str:
        return cls.descriptor.display_tableau(cls.tableau)

    @property
    def short_name(self) -> str:
        return self.descriptor.short_name

    @property
    def full_name(self) -> str:
        return self.descriptor.full_name

    def __repr__(self) -> str:
        return self.descriptor.repr_for(type(self).__name__, self.tableau)

    def __str__(self) -> str:
        return self.display_tableau()

    def __format__(self, format_spec: str) -> str:
        return format(str(self), format_spec)

    def set_apply_delta_safety(self, enabled: bool) -> None:
        self.workspace.set_apply_delta_safety(enabled)

    def snapshot_state(self, state: State) -> State:
        return self.workspace.snapshot_state(state)

    def __call__(self, interval: IntervalLike, state: State, tolerance: Tolerance) -> float:
        del tolerance
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        workspace = self.workspace
        dt = interval.step if interval.step <= remaining else remaining
        self.residual.configure(state, dt)
        self.trial_block.items[0] = workspace.scale(self.trial_block[0], 0.0, self.trial_block[0])
        self.resolver(self.trial_block, self.residual)
        workspace.apply_delta(self.trial_block[0], state)

        remaining_after = interval.stop - (interval.present + dt)
        interval.step = 0.0 if remaining_after <= 0.0 else min(interval.step, remaining_after)
        return dt


__all__ = ["BE_TABLEAU", "SchemeBackwardEuler"]
