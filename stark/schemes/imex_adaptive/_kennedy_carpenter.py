from __future__ import annotations

from stark.block import Block
from stark.contracts import ImExDerivative, IntervalLike, Resolvent, State, Workbench
from stark.execution.executor import Executor
from stark.execution.regulator import Regulator
from stark.resolvents.support.failure import ResolventError
from stark.schemes.support import (
    SchemeStepControl,
    initialise_adaptive_runtime,
    initialise_imex_support,
    refresh_adaptive_call,
    unbound_scheme_call,
)
from stark.schemes.support.imex_stencil import SchemeStencilImexTableau
from stark.schemes.support.specialist import SchemeSpecialist
from stark.schemes.support.stage_problem import SchemeStageProblem


class SchemeKennedyCarpenterAdaptive:
    """Shared implementation for Kennedy-Carpenter adaptive IMEX schemes.

    Concrete subclasses provide only the paired IMEX tableau, descriptor, and
    docstring. The algorithm remains visible here because all members of this
    family have the same additive Runge-Kutta structure.
    """

    step_control: SchemeStepControl

    __slots__ = (
        "step_control",
        "advance_delta_kernel",
        "call_pure",
        "delta",
        "delta_high",
        "error_delta",
        "error_delta_kernel",
        "explicit_derivative",
        "implicit_derivative",
        "k_explicit",
        "k_implicit",
        "redirect_call",
        "resolvent",
        "rhs",
        "stage_rhs_kernels",
        "stage_states",
        "workspace",
    )

    def __init__(
        self,
        derivative: ImExDerivative,
        workbench: Workbench,
        resolvent: Resolvent,
        regulator: Regulator | None = None,
        *,
        specialist: SchemeSpecialist | None = None,
    ) -> None:
        initialise_imex_support(self, derivative, workbench)
        initialise_adaptive_runtime(self, regulator)

        self.explicit_derivative = derivative.explicit
        self.implicit_derivative = derivative.implicit
        self.resolvent = resolvent

        workspace = self.workspace
        stage_count = len(self.tableau.implicit.a)

        self.delta = Block(list(workspace.allocate_translation_buffers(stage_count)))
        self.rhs = Block(list(workspace.allocate_translation_buffers(stage_count)))
        self.stage_states = tuple(
            workspace.allocate_state_buffer() for _ in range(stage_count)
        )
        self.k_explicit = tuple(workspace.allocate_translation_buffers(stage_count))
        self.k_implicit = tuple(workspace.allocate_translation_buffers(stage_count))
        self.delta_high, self.error_delta = workspace.allocate_translation_buffers(2)

        self.stage_rhs_kernels = ()
        self.advance_delta_kernel = unbound_scheme_call
        self.error_delta_kernel = unbound_scheme_call

        self.call_pure = self.call_inline
        refresh_adaptive_call(self)

        if specialist is not None:
            self.prepare_specialized_kernels(specialist)
            self.call_pure = self.call_specialized
            refresh_adaptive_call(self)

    def __call__(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        return self.redirect_call(interval, state, executor)

    def prepare_specialized_kernels(self, specialist: SchemeSpecialist) -> None:
        """Prepare fixed-coefficient kernels for the specialized IMEX path."""

        stencils = SchemeStencilImexTableau(self.tableau)
        stage_count = len(self.tableau.implicit.a)

        # Step 1 builds each known stage RHS from previous explicit and
        # implicit derivative buffers. The diagonal implicit coefficient is
        # deliberately excluded from these stencils.
        self.stage_rhs_kernels = tuple(
            specialist.provide(stencils.stage_rhs(index))
            for index in range(stage_count)
        )

        # Steps 4 and 5 build the accepted increment and embedded error
        # estimate from the split derivative families.
        self.advance_delta_kernel = specialist.provide(stencils.advance_delta())
        self.error_delta_kernel = specialist.provide(stencils.error_delta())

    def call_inline(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        del executor
        return self._call(interval, state, specialized=False)

    def call_specialized(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
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
        proposal = self.step_control.propose_step(interval)
        if proposal.remaining <= 0.0:
            self.step_control.record_stopped(interval)
            return 0.0

        ratio = self.step_control.ratio
        remaining = proposal.remaining
        dt = proposal.dt
        proposed_dt = proposal.proposed_dt
        t_start = proposal.t_start
        rejection_count = 0
        scheme_name = self.tableau.short_name or type(self).__name__

        while True:
            try:
                self._trial_step(interval, state, dt, specialized=specialized)
            except ResolventError:
                rejection_count += 1
                dt = self.step_control.rejected_step(dt, 1.0, remaining, scheme_name)
                continue

            delta_high_norm = self.delta_high.norm()
            error_norm = self.error_delta.norm()
            error_ratio = ratio(error_norm, delta_high_norm)
            if error_ratio <= 1.0:
                break

            rejection_count += 1
            dt = self.step_control.rejected_step(
                dt,
                error_ratio,
                remaining,
                scheme_name,
            )

        accepted_dt = dt
        remaining_after = remaining - accepted_dt
        next_dt = self.step_control.accepted_next_step(
            accepted_dt,
            error_ratio,
            remaining_after,
        )
        interval.step = next_dt

        self.workspace.apply_delta(self.delta_high, state)

        report = self.step_control.record_accepted(
            accepted_dt=accepted_dt,
            t_start=t_start,
            proposed_dt=proposed_dt,
            next_dt=next_dt,
            error_ratio=error_ratio,
            rejection_count=rejection_count,
        )
        return report.accepted_dt

    def _trial_step(
        self,
        interval: IntervalLike,
        state: State,
        dt: float,
        *,
        specialized: bool,
    ) -> None:
        workspace = self.workspace
        tableau = self.tableau
        explicit_derivative = self.explicit_derivative
        implicit_derivative = self.implicit_derivative
        k_explicit = self.k_explicit
        k_implicit = self.k_implicit
        delta = self.delta
        rhs = self.rhs

        stage_intervals = tuple(
            workspace.stage_at(interval, dt, coefficient * dt)
            for coefficient in tableau.implicit.c
        )

        for stage_index in range(len(tableau.implicit.a)):
            # 1. Build the known IMEX stage RHS:
            #        rhs_i = h * sum_{j<i} aE_ij kE_j
            #              + h * sum_{j<i} aI_ij kI_j
            if specialized:
                rhs[stage_index] = self.stage_rhs_kernels[stage_index](
                    dt,
                    *self._stage_sources(stage_index),
                    rhs[stage_index],
                )
            else:
                rhs[stage_index] = self._stage_rhs_inline(
                    stage_index,
                    dt,
                    rhs[stage_index],
                )

            implicit_row = tableau.implicit.a[stage_index]
            diagonal = implicit_row[stage_index] if stage_index < len(implicit_row) else 0.0

            if diagonal == 0.0:
                # Explicit first stages have no diagonal implicit solve.
                delta[stage_index] = rhs[stage_index]
            else:
                # 2. Solve the diagonal implicit stage equation:
                #        delta_i = rhs_i + h aI_ii fI(t_i, y + delta_i)
                rhs_block = Block([rhs[stage_index]])
                delta_block = Block([delta[stage_index]])
                problem = SchemeStageProblem(
                    derivative=implicit_derivative,
                    interval=stage_intervals[stage_index],
                    origin=state,
                    rhs=rhs_block,
                    alpha=dt * diagonal,
                )
                self.resolvent(problem, delta_block)
                delta[stage_index] = delta_block[0]

            # 3. Recompute both split derivatives at the solved stage.
            delta[stage_index](state, self.stage_states[stage_index])
            explicit_derivative(
                stage_intervals[stage_index],
                self.stage_states[stage_index],
                k_explicit[stage_index],
            )
            implicit_derivative(
                stage_intervals[stage_index],
                self.stage_states[stage_index],
                k_implicit[stage_index],
            )

        if specialized:
            # 4. Build the high-order accepted increment.
            self.delta_high = self.advance_delta_kernel(
                dt,
                *self._advance_sources(),
                self.delta_high,
            )
            # 5. Build the embedded error increment.
            self.error_delta = self.error_delta_kernel(
                dt,
                *self._advance_sources(),
                self.error_delta,
            )
        else:
            # 4. Build the high-order accepted increment.
            self.delta_high = self._advance_inline(dt, self.delta_high, error=False)
            # 5. Build the embedded error increment.
            self.error_delta = self._advance_inline(dt, self.error_delta, error=True)

    def _stage_sources(self, stage_index: int):
        return (
            *self.k_explicit[:stage_index],
            *self.k_implicit[:stage_index],
        )

    def _advance_sources(self):
        return (*self.k_explicit, *self.k_implicit)

    def _stage_rhs_inline(self, stage_index: int, dt: float, out):
        explicit_coefficients = self.tableau.explicit.a[stage_index][:stage_index]
        implicit_coefficients = self.tableau.implicit.a[stage_index][:stage_index]
        terms = (
            *(zip(explicit_coefficients, self.k_explicit[:stage_index], strict=True)),
            *(zip(implicit_coefficients, self.k_implicit[:stage_index], strict=True)),
        )
        return self._linear_combination_inline(dt, out, terms)

    def _advance_inline(self, dt: float, out, *, error: bool):
        if error:
            explicit_coefficients = _difference(
                self.tableau.explicit.b_high,
                self.tableau.explicit.b_low,
            )
            implicit_coefficients = _difference(
                self.tableau.implicit.b_high,
                self.tableau.implicit.b_low,
            )
        else:
            explicit_coefficients = self.tableau.explicit.b
            implicit_coefficients = self.tableau.implicit.b

        terms = (
            *(zip(explicit_coefficients, self.k_explicit, strict=True)),
            *(zip(implicit_coefficients, self.k_implicit, strict=True)),
        )
        return self._linear_combination_inline(dt, out, terms)

    def _linear_combination_inline(self, dt: float, out, terms):
        workspace = self.workspace
        first = True

        for coefficient, source in terms:
            if coefficient == 0.0:
                continue
            if first:
                out = workspace.scale(dt * coefficient, source, out)
                first = False
            else:
                out = workspace.combine2(1.0, out, dt * coefficient, source, out)

        if first:
            out = workspace.scale(0.0, out, out)

        return out


def _difference(high, low) -> tuple[float, ...]:
    return tuple(
        high_weight - low_weight
        for high_weight, low_weight in zip(tuple(high), tuple(low), strict=True)
    )


__all__ = ["SchemeKennedyCarpenterAdaptive"]
