from __future__ import annotations

from stark.audit import Auditor
from stark.butcher_tableau import ButcherTableau
from stark.tolerance import Tolerance
from stark.contracts import Block, Derivative, IntervalLike, Linearizer, ResolverLike, State, Workbench
from stark.resolver_library.picard import ResolverPicard
from stark.scheme_support.descriptor import SchemeDescriptor
from stark.scheme_support.implicit_residual import ShiftedImplicitResidual
from stark.scheme_support.workspace import SchemeWorkspace


BE_TABLEAU = ButcherTableau(
    c=(1.0,),
    a=((1.0,),),
    b=(1.0,),
    order=1,
)


class SchemeBackwardEuler:
    """
    The implicit backward Euler method resolved by a nonlinear solver.

    Backward Euler advances by solving

        x_{n+1} = x_n + dt f(x_{n+1}),

    so each step requires a residual equation to be resolved rather than a
    direct explicit stage evaluation. In STARK that residual is handed to a
    `ResolverLike`, which may use Picard, Newton, Anderson, Broyden, or a
    user-defined strategy.

    Further reading: https://en.wikipedia.org/wiki/Backward_Euler_method
    """

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
        if linearizer is not None:
            Auditor.require_linearizer_inputs(linearizer, workbench, translation_probe)
        self.derivative = derivative
        self.workspace = SchemeWorkspace(workbench, translation_probe)
        self.resolver = resolver if resolver is not None else ResolverPicard(workbench)
        self.trial_block = Block([translation_probe])
        self.residual = ShiftedImplicitResidual("Backward Euler", derivative, self.workspace, linearizer=linearizer)

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

