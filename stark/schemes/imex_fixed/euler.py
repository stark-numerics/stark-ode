from __future__ import annotations

from stark.block import Block
from stark.contracts import DerivativeIMEX, IntervalLike, Resolvent, State, Allocator
from stark.schemes.support.executor import SchemeExecutor
from stark.executor.adaptivity import ExecutorAdaptivity
from stark.schemes.support.descriptor import SchemeDescriptor
from stark.schemes.support import (
    initialise_imex_support,
    refresh_fixed_step_call,
    unbound_scheme_call,
    with_fixed_step_monitoring,
    with_imex_workspace_methods,
    with_scheme_display,
)
from stark.schemes.support.imex_stencil import SchemeStencilImexTableau
from stark.schemes.support.specialist import SchemeSpecialist
from stark.schemes.support.stage_problem import SchemeStageProblem
from stark.schemes.support.tableau import ButcherTableau, ButcherTableauImex


IMEX_EULER_EXPLICIT = ButcherTableau(
    c=(0.0, 1.0),
    a=((), (1.0,)),
    b=(1.0, 0.0),
    order=1,
)
IMEX_EULER_IMPLICIT = ButcherTableau(
    c=(0.0, 1.0),
    a=((), (0.0, 1.0)),
    b=(0.0, 1.0),
    order=1,
)
IMEX_EULER_TABLEAU = ButcherTableauImex(
    explicit=IMEX_EULER_EXPLICIT,
    implicit=IMEX_EULER_IMPLICIT,
    short_name="IMEXEuler",
    full_name="IMEX Euler",
)


@with_scheme_display
@with_fixed_step_monitoring
@with_imex_workspace_methods
class SchemeIMEXEuler:
    """First-order IMEX Euler with explicit and implicit derivative splits.

    Algorithm sketch for one accepted step of size h:

        1. Compute the explicit contribution at the current state:
               rhs = h * fE(t, y)

        2. Solve the implicit shifted stage problem:
               delta = rhs + h * fI(t + h, y + delta)

        3. Apply the solved IMEX increment:
               y <- y + delta

    The scheme owns the explicit/implicit split. The resolvent owns only the
    one-stage implicit solve.
    """

    __slots__ = (
        "_monitor",
        "advance_call",
        "call_pure",
        "delta",
        "delta_block",
        "explicit_derivative",
        "explicit_rate",
        "implicit_derivative",
        "redirect_call",
        "resolvent",
        "rhs",
        "rhs_block",
        "workspace",
    )

    descriptor = SchemeDescriptor("IMEXEuler", "IMEX Euler")
    tableau = IMEX_EULER_TABLEAU

    def __init__(
        self,
        derivative: DerivativeIMEX,
        allocator: Allocator,
        resolvent: Resolvent,
        adaptivity: ExecutorAdaptivity | None = None,
        *,
        specialist: SchemeSpecialist | None = None,
    ) -> None:
        del adaptivity
        self._monitor = None
        self.advance_call = unbound_scheme_call
        self.resolvent = resolvent

        initialise_imex_support(self, derivative, allocator)
        self.explicit_derivative = derivative.explicit
        self.implicit_derivative = derivative.implicit

        workspace = self.workspace
        self.explicit_rate, self.rhs, self.delta = workspace.allocate_translation_buffers(3)
        self.rhs_block = Block([self.rhs])
        self.delta_block = Block([self.delta])

        self.call_pure = self.call_inline
        refresh_fixed_step_call(self)

        if specialist is not None:
            self.prepare_specialized_kernels(specialist)
            self.call_pure = self.call_specialized
            refresh_fixed_step_call(self)

    def __call__(
        self,
        interval: IntervalLike,
        state: State,
        executor: SchemeExecutor,
    ) -> float:
        return self.redirect_call(interval, state, executor)

    def prepare_specialized_kernels(self, specialist: SchemeSpecialist) -> None:
        stencils = SchemeStencilImexTableau(self.tableau)
        # Step 1 builds the explicit right-hand side from the first explicit row.
        self.advance_call = specialist.provide(stencils.stage_rhs(1))

    def call_inline(
        self,
        interval: IntervalLike,
        state: State,
        executor: SchemeExecutor,
    ) -> float:
        del executor
        return self._call(interval, state, specialized=False)

    def call_specialized(
        self,
        interval: IntervalLike,
        state: State,
        executor: SchemeExecutor,
    ) -> float:
        del executor
        return self._call(interval, state, specialized=True)

    def _call(
        self,
        interval: IntervalLike,
        state: State,
        *,
        specialized: bool,
    ) -> float:
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        dt = interval.step if interval.step <= remaining else remaining
        workspace = self.workspace

        # 1. Compute the explicit contribution rhs = h * fE(t, y).
        self.explicit_derivative(interval, state, self.explicit_rate)
        if specialized:
            self.rhs = self.advance_call(dt, self.explicit_rate, self.delta, self.rhs)
        else:
            self.rhs = workspace.scale(dt, self.explicit_rate, self.rhs)

        # 2. Solve delta = rhs + h * fI(t + h, y + delta).
        self.rhs_block[0] = self.rhs
        self.delta_block[0] = self.delta
        problem = SchemeStageProblem(
            derivative=self.implicit_derivative,
            interval=workspace.stage_at(interval, dt, dt),
            origin=state,
            rhs=self.rhs_block,
            alpha=dt,
        )
        self.resolvent(problem, self.delta_block)
        self.delta = self.delta_block[0]

        # 3. Apply the solved IMEX increment.
        workspace.apply_delta(self.delta, state)

        remaining_after = remaining - dt
        interval.step = 0.0 if remaining_after <= 0.0 else min(interval.step, remaining_after)
        return dt


__all__ = ["IMEX_EULER_TABLEAU", "SchemeIMEXEuler"]
