from __future__ import annotations

from stark.methods.schemes.configuration import SchemeConfiguration
from stark.core.block import Block
from stark.core.contracts import DerivativeSplitLike, IntervalLike, Resolvent, State, Allocator
from stark.methods.schemes.method.descriptor import SchemeDescriptor
from stark.methods.schemes.monitoring.monitor import SchemeMonitor
from stark.methods.schemes.monitoring.decorators import with_fixed_step_monitoring
from stark.methods.schemes.execution.call import SchemeCall
from stark.methods.schemes.display.display import display_imex_resolvent_problem
from stark.methods.schemes.imex.runtime import SchemeRuntimeImex
from stark.methods.schemes.execution.unbound import unbound_scheme_call
from stark.methods.schemes.display.decorators import with_scheme_display
from stark.methods.schemes.specialization.imex_stencil import SchemeStencilImexTableau
from stark.methods.schemes.specialization.specialist import SchemeSpecialist
from stark.methods.schemes.request import SchemeResolventRequest
from stark.methods.schemes.method.tableau import Tableau, TableauImex


IMEX_EULER_EXPLICIT = Tableau(
    c=(0.0, 1.0),
    a=((), (1.0,)),
    b=(1.0, 0.0),
    order=1,
)
IMEX_EULER_IMPLICIT = Tableau(
    c=(0.0, 1.0),
    a=((), (0.0, 1.0)),
    b=(0.0, 1.0),
    order=1,
)
IMEX_EULER_TABLEAU = TableauImex(
    explicit=IMEX_EULER_EXPLICIT,
    implicit=IMEX_EULER_IMPLICIT,
    short_name="IMEXEuler",
    full_name="IMEX Euler",
)


# Optional extension: adds human-readable scheme metadata and formatting helpers.
# Provides: with_scheme_display, display_tableau, __repr__, __str__, and __format__.
@with_scheme_display
# Optional extension: records fixed-step monitor events.
# Provides: call_monitored.
@with_fixed_step_monitoring
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

    # Installed by the scheme monitoring decorator above this class.
    call_monitored: SchemeCall

    __slots__ = (
        "monitor",
        "advance_call",
        "call_body",
        "call_step",
        "delta",
        "delta_block",
        "explicit_derivative",
        "explicit_rate",
        "implicit_derivative",
        "redirect_call",
        "resolvent",
        "rhs",
        "rhs_block",
        "runtime",
        "workspace",
    )

    descriptor = SchemeDescriptor("IMEXEuler", "IMEX Euler")

    @classmethod
    def display_tableau(cls) -> str:
        """Installed by `with_scheme_display` from `stark.methods.schemes.display`."""

        raise NotImplementedError("with_scheme_display installs display_tableau.")

    @classmethod
    def display_resolvent_problem(cls) -> str:
        return display_imex_resolvent_problem(
            cls.tableau,
            cls.descriptor.short_name,
            cls.descriptor.full_name,
        )

    def snapshot_state(self, state: State) -> State:
        return self.runtime.snapshot_state(state)

    tableau = IMEX_EULER_TABLEAU

    def __init__(
        self,
        derivative: DerivativeSplitLike,
        allocator: Allocator,
        resolvent: Resolvent,
        *,
        configuration: SchemeConfiguration | None = None,
        specialist: SchemeSpecialist | None = None,
        monitor: SchemeMonitor | None = None,
    ) -> None:
        del configuration
        self.advance_call = unbound_scheme_call
        self.monitor = monitor
        self.call_body = self.call_inline
        self.call_step = self.call_monitored if monitor is not None else self.call_body
        self.redirect_call = self.call_step
        self.resolvent = resolvent

        self.runtime = SchemeRuntimeImex(derivative, allocator)
        self.workspace = self.runtime.workspace
        self.explicit_derivative = derivative.explicit
        self.implicit_derivative = derivative.implicit

        workspace = self.workspace
        self.explicit_rate, self.rhs, self.delta = workspace.allocate_translation_buffers(3)
        self.rhs_block = Block([self.rhs])
        self.delta_block = Block([self.delta])

        if specialist is not None:
            self.prepare_specialized_kernels(specialist)
            self.call_body = self.call_specialized
            if monitor is None:
                self.call_step = self.call_body
                self.redirect_call = self.call_step

    def __call__(
        self,
        interval: IntervalLike,
        state: State,
    ) -> float:
        return self.redirect_call(interval, state)

    def prepare_specialized_kernels(self, specialist: SchemeSpecialist) -> None:
        stencils = SchemeStencilImexTableau(self.tableau)
        # Step 1 builds the explicit right-hand side from the first explicit row.
        self.advance_call = specialist.provide_delta(stencils.stage_rhs(1))

    def call_inline(
        self,
        interval: IntervalLike,
        state: State,
    ) -> float:
        return self._call(interval, state, specialized=False)

    def call_specialized(
        self,
        interval: IntervalLike,
        state: State,
    ) -> float:
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
        problem = SchemeResolventRequest(
            derivative=self.implicit_derivative,
            interval=workspace.interval_at(interval, dt, dt),
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
