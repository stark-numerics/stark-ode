from __future__ import annotations

from stark.core.contracts import DerivativeLike, IntervalLike, State, Allocator

from stark.methods.schemes.configuration import SchemeConfiguration, SchemeConfigurationDefault
from stark.methods.schemes.method.descriptor import SchemeDescriptor
from stark.methods.schemes.monitoring.monitor import SchemeMonitor
from stark.methods.schemes.monitoring.decorators import with_adaptive_step_monitoring
from stark.methods.schemes.execution.call import SchemeCall
from stark.methods.schemes.execution.step_control import SchemeStepControl
from stark.methods.schemes.explicit.runtime import SchemeRuntimeExplicit
from stark.methods.schemes.execution.unbound import unbound_scheme_call
from stark.methods.schemes.display.decorators import with_scheme_display
from stark.methods.schemes.specialization.specialist import SchemeSpecialist
from stark.methods.schemes.specialization.stencil import SchemeStencilTableau
from stark.methods.schemes.method.tableau import Tableau


BS23_TABLEAU = Tableau(
    c=(0.0, 0.5, 0.75, 1.0),
    a=(
        (),
        (0.5,),
        (0.0, 0.75),
        (2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0),
    ),
    b=(2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0),
    order=3,
    b_embedded=(7.0 / 24.0, 0.25, 1.0 / 3.0, 0.125),
    embedded_order=2,
)

BS23_A = BS23_TABLEAU.a
BS23_C = BS23_TABLEAU.c
BS23_B_HIGH = BS23_TABLEAU.b
BS23_B_LOW = BS23_TABLEAU.b_embedded
assert BS23_B_LOW is not None
BS23_B_HIGH_NZ = (
    BS23_B_HIGH[0],
    BS23_B_HIGH[1],
    BS23_B_HIGH[2],
)
BS23_ERROR_WEIGHTS = tuple(
    high - low
    for high, low in zip(BS23_B_HIGH, BS23_B_LOW, strict=True)
)


# Optional extension: adds human-readable scheme metadata and formatting helpers.
# Provides: with_scheme_display, display_tableau, __repr__, __str__, and __format__.
@with_scheme_display
# Optional extension: records accepted/rejected adaptive-step monitor events.
# Provides: call_monitored.
@with_adaptive_step_monitoring
class SchemeBogackiShampine:
    """The adaptive Bogacki-Shampine embedded 3(2) Runge-Kutta pair.

    This method advances with a third-order explicit Runge-Kutta formula while estimating local error with an embedded second-order formula. It is a light adaptive method for non-stiff problems where a cheap error estimate is more valuable than very high formal order.

    Algorithm sketch for one accepted step of size h:

        1. k1 = f(t, y)
        2. k2 = f(t + c[1] h, y + h * A[1] dot k)
        3. k3 = f(t + c[2] h, y + h * A[2] dot k)
        4. k4 = f(t + c[3] h, y + h * A[3] dot k)
        5. high_delta = h * b_high dot k
        6. error_delta = h * ((b_high - b_low) dot k)

    The adaptive controller repeats the stage and error-estimate steps with a
    smaller h when the error ratio is too large. k1 is reused across rejected
    attempts because it depends only on the current accepted state.

    The inline path expresses the method directly with workspace arithmetic.
    The specialized path uses fixed-coefficient kernels prepared from the same
    tableau rows and weights.

    Further reading: https://en.wikipedia.org/wiki/Bogacki%E2%80%93Shampine_method
    """

    # Assigned by initialise_adaptive_runtime from stark.methods.schemes.execution.step_control.
    step_control: SchemeStepControl

    # Installed by the scheme monitoring decorator above this class.
    call_monitored: SchemeCall

    __slots__ = (
        "monitor",
        "step_control",
        "advance_delta",
        "bound_apply_delta",
        "bound_interval_at",
        "call_body",
        "call_step",
        "derivative",
        "error",
        "error_delta",
        "runtime",
        "k1",
        "k2",
        "k3",
        "k4",
        "redirect_call",
        "stage",
        "stage2_update",
        "stage3_update",
        "stage4_update",
        "trial",
        "workspace",
    )

    descriptor = SchemeDescriptor('BS23', 'Bogacki-Shampine')

    def snapshot_state(self, state: State) -> State:
        return self.runtime.snapshot_state(state)

    tableau = BS23_TABLEAU

    def __init__(
        self,
        derivative: DerivativeLike,
        allocator: Allocator,
        configuration: SchemeConfiguration | None = None,
        specialist: SchemeSpecialist | None = None,
        monitor: SchemeMonitor | None = None,
    ) -> None:
        self.runtime = SchemeRuntimeExplicit(derivative, allocator)
        self.derivative = self.runtime.derivative
        self.workspace = self.runtime.workspace
        self.k1 = self.runtime.k1
        self.step_control = SchemeStepControl(configuration if configuration is not None else SchemeConfigurationDefault())

        self.initialise_buffers()

        self.call_body = self.call_inline
        self.monitor = monitor
        self.call_step = self.call_monitored if monitor is not None else self.call_body
        self.redirect_call = self.call_step

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

    def initialise_buffers(self) -> None:
        workspace = self.workspace

        self.stage = workspace.allocate_state_buffer()
        self.trial, self.error, self.k2, self.k3, self.k4 = workspace.allocate_translation_buffers(5)

        self.advance_delta = unbound_scheme_call
        self.error_delta = unbound_scheme_call
        self.stage2_update = unbound_scheme_call
        self.stage3_update = unbound_scheme_call
        self.stage4_update = unbound_scheme_call
        self.bound_apply_delta = workspace.apply_delta
        self.bound_interval_at = workspace.interval_at

    def prepare_specialized_kernels(
        self,
        specialist: SchemeSpecialist,
    ) -> None:
        """Prepare fixed-coefficient kernels for the specialized path."""

        stencils = SchemeStencilTableau(self.tableau)

        # Stage rows build staged states from the tableau's A matrix.
        self.stage2_update = specialist.provide_apply(stencils.stage(1))
        self.stage3_update = specialist.provide_apply(stencils.stage(2))
        self.stage4_update = specialist.provide_apply(stencils.stage(3))

        # The accepted advance and embedded error are translation deltas.
        self.advance_delta = specialist.provide_delta(stencils.advance_delta())
        self.error_delta = specialist.provide_delta(stencils.error_delta())

    def call_inline(
        self,
        interval: IntervalLike,
        state: State,
    ) -> float:
        step_control = self.step_control
        proposal = step_control.propose_step(interval)
        record_stopped = step_control.record_stopped

        if proposal.remaining <= 0.0:
            record_stopped(interval)
            return 0.0

        workspace = self.workspace
        derivative = self.derivative
        scale = workspace.scale
        combine2 = workspace.combine2
        combine3 = workspace.combine3
        combine4 = workspace.combine4
        apply_delta = self.bound_apply_delta
        interval_at = self.bound_interval_at

        ratio = step_control.ratio
        rejected_step = step_control.rejected_step
        accepted_next_step = step_control.accepted_next_step
        record_accepted = step_control.record_accepted

        stage = self.stage
        trial_buffer = self.trial
        error_buffer = self.error
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        k4 = self.k4

        remaining = proposal.remaining
        dt = proposal.dt
        proposed_dt = proposal.proposed_dt
        t_start = proposal.t_start
        rejection_count = 0

        # 1. k1 = f(t, y)
        #
        # k1 depends only on the current accepted state, so rejected attempts
        # can reuse it while trying smaller step sizes.
        derivative(interval, state, k1)

        while True:
            # 2. k2 = f(t + BS23_C[1] h, y + h * A[1] dot k)
            stage_delta = scale(dt * BS23_A[1][0], k1, trial_buffer)
            stage_delta(state, stage)
            derivative(interval_at(interval, dt, dt * BS23_C[1]), stage, k2)
            # 3. k3 = f(t + BS23_C[2] h, y + h * A[2] dot k)
            stage_delta = combine2(
                dt * BS23_A[2][0],
                k1,
                dt * BS23_A[2][1],
                k2,
                trial_buffer,
            )
            stage_delta(state, stage)
            derivative(interval_at(interval, dt, dt * BS23_C[2]), stage, k3)
            # 4. k4 = f(t + BS23_C[3] h, y + h * A[3] dot k)
            stage_delta = combine3(
                dt * BS23_A[3][0],
                k1,
                dt * BS23_A[3][1],
                k2,
                dt * BS23_A[3][2],
                k3,
                trial_buffer,
            )
            stage_delta(state, stage)
            derivative(interval_at(interval, dt, dt * BS23_C[3]), stage, k4)
            # 5. high_delta = h * b_high dot k
            high_delta = combine3(
                dt * BS23_B_HIGH_NZ[0],
                k1,
                dt * BS23_B_HIGH_NZ[1],
                k2,
                dt * BS23_B_HIGH_NZ[2],
                k3,
                trial_buffer,
            )

            # 6. error_delta = h * ((b_high - b_low) dot k)
            error_delta = combine4(
                dt * BS23_ERROR_WEIGHTS[0],
                k1,
                dt * BS23_ERROR_WEIGHTS[1],
                k2,
                dt * BS23_ERROR_WEIGHTS[2],
                k3,
                dt * BS23_ERROR_WEIGHTS[3],
                k4,
                error_buffer,
            )

            error_ratio = ratio(error_delta.norm(), high_delta.norm())

            if error_ratio <= 1.0:
                break

            rejection_count += 1
            dt = rejected_step(
                dt,
                error_ratio,
                remaining,
                self.tableau.short_name,
            )

        accepted_dt = dt
        remaining_after = remaining - accepted_dt
        next_dt = accepted_next_step(
            accepted_dt,
            error_ratio,
            remaining_after,
        )

        interval.step = next_dt
        apply_delta(high_delta, state)

        report = record_accepted(
            accepted_dt=accepted_dt,
            t_start=t_start,
            proposed_dt=proposed_dt,
            next_dt=next_dt,
            error_ratio=error_ratio,
            rejection_count=rejection_count,
        )

        return report.accepted_dt

    def call_specialized(
        self,
        interval: IntervalLike,
        state: State,
    ) -> float:
        step_control = self.step_control
        proposal = step_control.propose_step(interval)
        record_stopped = step_control.record_stopped

        if proposal.remaining <= 0.0:
            record_stopped(interval)
            return 0.0

        derivative = self.derivative
        apply_delta = self.bound_apply_delta
        interval_at = self.bound_interval_at

        ratio = step_control.ratio
        rejected_step = step_control.rejected_step
        accepted_next_step = step_control.accepted_next_step
        record_accepted = step_control.record_accepted

        stage = self.stage
        trial_buffer = self.trial
        error_buffer = self.error
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        k4 = self.k4

        advance_delta = self.advance_delta
        error_delta = self.error_delta
        stage2_update = self.stage2_update
        stage3_update = self.stage3_update
        stage4_update = self.stage4_update

        remaining = proposal.remaining
        dt = proposal.dt
        proposed_dt = proposal.proposed_dt
        t_start = proposal.t_start
        rejection_count = 0

        # 1. k1 = f(t, y)
        #
        # k1 depends only on the current accepted state, so rejected attempts
        # can reuse it while trying smaller step sizes.
        derivative(interval, state, k1)

        while True:
            # 2. k2 = f(t + BS23_C[1] h, y + h * A[1] dot k)
            stage2_update(dt, state, k1, stage)
            derivative(interval_at(interval, dt, dt * BS23_C[1]), stage, k2)
            # 3. k3 = f(t + BS23_C[2] h, y + h * A[2] dot k)
            stage3_update(dt, state, k1, k2, stage)
            derivative(interval_at(interval, dt, dt * BS23_C[2]), stage, k3)
            # 4. k4 = f(t + BS23_C[3] h, y + h * A[3] dot k)
            stage4_update(dt, state, k1, k2, k3, stage)
            derivative(interval_at(interval, dt, dt * BS23_C[3]), stage, k4)
            # 5. high_delta = h * b_high dot k
            high_delta = advance_delta(dt, k1, k2, k3, k4, trial_buffer)

            # 6. error_delta = h * ((b_high - b_low) dot k)
            error = error_delta(dt, k1, k2, k3, k4, error_buffer)

            error_ratio = ratio(error.norm(), high_delta.norm())

            if error_ratio <= 1.0:
                break

            rejection_count += 1
            dt = rejected_step(
                dt,
                error_ratio,
                remaining,
                self.tableau.short_name,
            )

        accepted_dt = dt
        remaining_after = remaining - accepted_dt
        next_dt = accepted_next_step(
            accepted_dt,
            error_ratio,
            remaining_after,
        )

        interval.step = next_dt
        apply_delta(high_delta, state)

        report = record_accepted(
            accepted_dt=accepted_dt,
            t_start=t_start,
            proposed_dt=proposed_dt,
            next_dt=next_dt,
            error_ratio=error_ratio,
            rejection_count=rejection_count,
        )

        return report.accepted_dt


__all__ = ["BS23_TABLEAU", "SchemeBogackiShampine"]
