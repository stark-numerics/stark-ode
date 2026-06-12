from __future__ import annotations

from stark.methods.schemes.configuration import SchemeConfiguration, SchemeConfigurationDefault
from stark.core.contracts import DerivativeLike, IntervalLike, State, Allocator
from stark.methods.schemes.method.descriptor import SchemeDescriptor
from stark.methods.schemes.monitoring.monitor import SchemeMonitor
from stark.methods.schemes.monitoring.decorators import with_adaptive_step_monitoring
from stark.methods.schemes.execution.step_control import SchemeStepControl
from stark.methods.schemes.explicit._support import (
    initialise_explicit_support,
    explicit_snapshot_state,
)
from stark.methods.schemes.execution.unbound import unbound_scheme_call
from stark.methods.schemes.display.decorators import with_scheme_display
from stark.methods.schemes.specialization.specialist import SchemeSpecialist
from stark.methods.schemes.specialization.stencil import SchemeStencilTableau
from stark.methods.schemes.method.tableau import ButcherTableau


RKDP_TABLEAU = ButcherTableau(
    c=(0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0),
    a=(
        (),
        (1.0 / 5.0,),
        (3.0 / 40.0, 9.0 / 40.0),
        (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0),
        (
            19372.0 / 6561.0,
            -25360.0 / 2187.0,
            64448.0 / 6561.0,
            -212.0 / 729.0,
        ),
        (
            9017.0 / 3168.0,
            -355.0 / 33.0,
            46732.0 / 5247.0,
            49.0 / 176.0,
            -5103.0 / 18656.0,
        ),
        (),
    ),
    b=(35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0),
    order=5,
    b_embedded=(
        5179.0 / 57600.0,
        0.0,
        7571.0 / 16695.0,
        393.0 / 640.0,
        -92097.0 / 339200.0,
        187.0 / 2100.0,
        1.0 / 40.0,
    ),
    embedded_order=4,
)

RKDP_A = RKDP_TABLEAU.a
RKDP_C = RKDP_TABLEAU.c
RKDP_B_HIGH = RKDP_TABLEAU.b
RKDP_B_LOW = RKDP_TABLEAU.b_embedded
assert RKDP_B_LOW is not None
RKDP_B_HIGH_NZ = (
    RKDP_B_HIGH[0],
    RKDP_B_HIGH[2],
    RKDP_B_HIGH[3],
    RKDP_B_HIGH[4],
    RKDP_B_HIGH[5],
)
RKDP_B_ERR_NZ = (
    RKDP_B_HIGH[0] - RKDP_B_LOW[0],
    RKDP_B_HIGH[2] - RKDP_B_LOW[2],
    RKDP_B_HIGH[3] - RKDP_B_LOW[3],
    RKDP_B_HIGH[4] - RKDP_B_LOW[4],
    RKDP_B_HIGH[5] - RKDP_B_LOW[5],
    RKDP_B_HIGH[6] - RKDP_B_LOW[6],
)
RKDP_ERROR_WEIGHTS = tuple(
    high - low
    for high, low in zip(RKDP_B_HIGH, RKDP_B_LOW, strict=True)
)


# Optional extension: adds human-readable scheme metadata and formatting helpers.
# Provides: with_scheme_display, display_tableau, short_name, full_name, __repr__, __str__, and __format__.
@with_scheme_display
# Optional extension: records accepted/rejected adaptive-step monitor events.
# Provides: call_monitored.
@with_adaptive_step_monitoring
class SchemeDormandPrince:
    """The adaptive Dormand-Prince embedded 5(4) Runge-Kutta pair.

    This is the RK45 family most users meet first: a fifth-order explicit method with a fourth-order embedded error estimate. It is a strong default choice for smooth non-stiff problems and is the basis of many classic adaptive ODE drivers.

    Algorithm sketch for one accepted step of size h:

        1. k1 = f(t, y)
        2. k2 = f(t + c[1] h, y + h * A[1] dot k)
        3. k3 = f(t + c[2] h, y + h * A[2] dot k)
        4. k4 = f(t + c[3] h, y + h * A[3] dot k)
        5. k5 = f(t + c[4] h, y + h * A[4] dot k)
        6. k6 = f(t + c[5] h, y + h * A[5] dot k)
        7. high_delta = h * b_high dot k; k7 = f(t + h, y + high_delta)
        8. error_delta = h * ((b_high - b_low) dot k)

    The adaptive controller repeats the stage and error-estimate steps with a
    smaller h when the error ratio is too large. k1 is reused across rejected
    attempts because it depends only on the current accepted state.

    The inline path expresses the method directly with workspace arithmetic.
    The specialized path uses fixed-coefficient kernels prepared from the same
    tableau rows and weights.

    Further reading: https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method
    """

    # Assigned by initialise_adaptive_runtime from stark.methods.schemes.execution.step_control.
    step_control: SchemeStepControl

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
        "explicit",
        "k1",
        "k2",
        "k3",
        "k4",
        "k5",
        "k6",
        "k7",
        "redirect_call",
        "stage",
        "stage2_update",
        "stage3_update",
        "stage4_update",
        "stage5_update",
        "stage6_update",
        "stage7_update",
        "trial",
        "workspace",
    )

    descriptor = SchemeDescriptor('RKDP', 'Dormand-Prince')
    snapshot_state = explicit_snapshot_state
    tableau = RKDP_TABLEAU

    def __init__(
        self,
        derivative: DerivativeLike,
        allocator: Allocator,
        configuration: SchemeConfiguration | None = None,
        specialist: SchemeSpecialist | None = None,
        monitor: SchemeMonitor | None = None,
    ) -> None:
        initialise_explicit_support(self, derivative, allocator)
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
        self.trial, self.error, self.k2, self.k3, self.k4, self.k5, self.k6, self.k7 = workspace.allocate_translation_buffers(8)

        self.advance_delta = unbound_scheme_call
        self.error_delta = unbound_scheme_call
        self.stage2_update = unbound_scheme_call
        self.stage3_update = unbound_scheme_call
        self.stage4_update = unbound_scheme_call
        self.stage5_update = unbound_scheme_call
        self.stage6_update = unbound_scheme_call
        self.stage7_update = unbound_scheme_call
        self.bound_apply_delta = workspace.apply_delta
        self.bound_interval_at = workspace.interval_at

    def prepare_specialized_kernels(
        self,
        specialist: SchemeSpecialist,
    ) -> None:
        """Prepare fixed-coefficient kernels for the specialized path."""

        stencils = SchemeStencilTableau(self.tableau)

        # Stage rows build staged states from the tableau's A matrix.
        self.stage2_update = specialist.provide(stencils.stage(1))
        self.stage3_update = specialist.provide(stencils.stage(2))
        self.stage4_update = specialist.provide(stencils.stage(3))
        self.stage5_update = specialist.provide(stencils.stage(4))
        self.stage6_update = specialist.provide(stencils.stage(5))

        # The accepted advance and embedded error are translation deltas.
        self.advance_delta = specialist.provide(stencils.advance_delta())
        self.error_delta = specialist.provide(stencils.error_delta())

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
        combine5 = workspace.combine5
        combine6 = workspace.combine6
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
        k5 = self.k5
        k6 = self.k6
        k7 = self.k7

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
            # 2. k2 = f(t + RKDP_C[1] h, y + h * A[1] dot k)
            stage_delta = scale(dt * RKDP_A[1][0], k1, trial_buffer)
            stage_delta(state, stage)
            derivative(interval_at(interval, dt, dt * RKDP_C[1]), stage, k2)
            # 3. k3 = f(t + RKDP_C[2] h, y + h * A[2] dot k)
            stage_delta = combine2(
                dt * RKDP_A[2][0],
                k1,
                dt * RKDP_A[2][1],
                k2,
                trial_buffer,
            )
            stage_delta(state, stage)
            derivative(interval_at(interval, dt, dt * RKDP_C[2]), stage, k3)
            # 4. k4 = f(t + RKDP_C[3] h, y + h * A[3] dot k)
            stage_delta = combine3(
                dt * RKDP_A[3][0],
                k1,
                dt * RKDP_A[3][1],
                k2,
                dt * RKDP_A[3][2],
                k3,
                trial_buffer,
            )
            stage_delta(state, stage)
            derivative(interval_at(interval, dt, dt * RKDP_C[3]), stage, k4)
            # 5. k5 = f(t + RKDP_C[4] h, y + h * A[4] dot k)
            stage_delta = combine4(
                dt * RKDP_A[4][0],
                k1,
                dt * RKDP_A[4][1],
                k2,
                dt * RKDP_A[4][2],
                k3,
                dt * RKDP_A[4][3],
                k4,
                trial_buffer,
            )
            stage_delta(state, stage)
            derivative(interval_at(interval, dt, dt * RKDP_C[4]), stage, k5)
            # 6. k6 = f(t + RKDP_C[5] h, y + h * A[5] dot k)
            stage_delta = combine5(
                dt * RKDP_A[5][0],
                k1,
                dt * RKDP_A[5][1],
                k2,
                dt * RKDP_A[5][2],
                k3,
                dt * RKDP_A[5][3],
                k4,
                dt * RKDP_A[5][4],
                k5,
                trial_buffer,
            )
            stage_delta(state, stage)
            derivative(interval_at(interval, dt, dt * RKDP_C[5]), stage, k6)
            # 7. high_delta = h * b_high dot k, then k7 = f(t + h, y + high_delta)
            high_delta = combine5(
                dt * RKDP_B_HIGH_NZ[0],
                k1,
                dt * RKDP_B_HIGH_NZ[1],
                k3,
                dt * RKDP_B_HIGH_NZ[2],
                k4,
                dt * RKDP_B_HIGH_NZ[3],
                k5,
                dt * RKDP_B_HIGH_NZ[4],
                k6,
                trial_buffer,
            )
            high_delta(state, stage)
            derivative(interval_at(interval, dt, dt), stage, k7)

            # 8. error_delta = h * ((b_high - b_low) dot k)
            error_delta = combine6(
                dt * RKDP_B_ERR_NZ[0],
                k1,
                dt * RKDP_B_ERR_NZ[1],
                k3,
                dt * RKDP_B_ERR_NZ[2],
                k4,
                dt * RKDP_B_ERR_NZ[3],
                k5,
                dt * RKDP_B_ERR_NZ[4],
                k6,
                dt * RKDP_B_ERR_NZ[5],
                k7,
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
                self.short_name,
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
        k5 = self.k5
        k6 = self.k6
        k7 = self.k7

        advance_delta = self.advance_delta
        error_delta = self.error_delta
        stage2_update = self.stage2_update
        stage3_update = self.stage3_update
        stage4_update = self.stage4_update
        stage5_update = self.stage5_update
        stage6_update = self.stage6_update

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
            # 2. k2 = f(t + RKDP_C[1] h, y + h * A[1] dot k)
            stage2_update(dt, state, k1, stage)
            derivative(interval_at(interval, dt, dt * RKDP_C[1]), stage, k2)
            # 3. k3 = f(t + RKDP_C[2] h, y + h * A[2] dot k)
            stage3_update(dt, state, k1, k2, stage)
            derivative(interval_at(interval, dt, dt * RKDP_C[2]), stage, k3)
            # 4. k4 = f(t + RKDP_C[3] h, y + h * A[3] dot k)
            stage4_update(dt, state, k1, k2, k3, stage)
            derivative(interval_at(interval, dt, dt * RKDP_C[3]), stage, k4)
            # 5. k5 = f(t + RKDP_C[4] h, y + h * A[4] dot k)
            stage5_update(dt, state, k1, k2, k3, k4, stage)
            derivative(interval_at(interval, dt, dt * RKDP_C[4]), stage, k5)
            # 6. k6 = f(t + RKDP_C[5] h, y + h * A[5] dot k)
            stage6_update(dt, state, k1, k2, k3, k4, k5, stage)
            derivative(interval_at(interval, dt, dt * RKDP_C[5]), stage, k6)
            # 7. high_delta = h * b_high dot k, then k7 = f(t + h, y + high_delta)
            high_delta = advance_delta(dt, k1, k2, k3, k4, k5, k6, k7, trial_buffer)
            high_delta(state, stage)
            derivative(interval_at(interval, dt, dt), stage, k7)

            # 8. error_delta = h * ((b_high - b_low) dot k)
            error = error_delta(dt, k1, k2, k3, k4, k5, k6, k7, error_buffer)

            error_ratio = ratio(error.norm(), high_delta.norm())

            if error_ratio <= 1.0:
                break

            rejection_count += 1
            dt = rejected_step(
                dt,
                error_ratio,
                remaining,
                self.short_name,
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


__all__ = ["RKDP_TABLEAU", "SchemeDormandPrince"]
