from __future__ import annotations

from stark.contracts import ImExDerivative, IntervalLike, Resolvent, State, Workbench
from stark.execution.executor import Executor
from stark.execution.regulator import Regulator
from stark.machinery.stage_solve.workers import ImExStepper
from stark.monitor import MonitorStep
from stark.resolvents.failure import ResolventError
from stark.resolvents.support.guard import ResolventTableauGuard
from stark.schemes.base import SchemeBaseImExAdaptive
from stark.schemes.descriptor import SchemeDescriptor
from stark.schemes.tableau import ButcherTableau, ImExButcherTableau


ARK436L2SA_EXPLICIT = ButcherTableau(
    c=(0.0, 0.5, 83.0 / 250.0, 31.0 / 50.0, 17.0 / 20.0, 1.0),
    a=(
        (),
        (0.5,),
        (13861.0 / 62500.0, 6889.0 / 62500.0),
        (
            -116923316275.0 / 2393684061468.0,
            -2731218467317.0 / 15368042101831.0,
            9408046702089.0 / 11113171139209.0,
        ),
        (
            -451086348788.0 / 2902428689909.0,
            -2682348792572.0 / 7519795681897.0,
            12662868775082.0 / 11960479115383.0,
            3355817975965.0 / 11060851509271.0,
        ),
        (
            647845179188.0 / 3216320057751.0,
            73281519250.0 / 8382639484533.0,
            552539513391.0 / 3454668386233.0,
            3354512671639.0 / 8306763924573.0,
            4040.0 / 17871.0,
        ),
    ),
    b=(
        82889.0 / 524892.0,
        0.0,
        15625.0 / 83664.0,
        69875.0 / 102672.0,
        -2260.0 / 8211.0,
        0.25,
    ),
    order=4,
    b_embedded=(
        4586570599.0 / 29645900160.0,
        0.0,
        178811875.0 / 945068544.0,
        814220225.0 / 1159782912.0,
        -3700637.0 / 11593932.0,
        61727.0 / 225920.0,
    ),
    embedded_order=3,
)


ARK436L2SA_IMPLICIT = ButcherTableau(
    c=(0.0, 0.5, 83.0 / 250.0, 31.0 / 50.0, 17.0 / 20.0, 1.0),
    a=(
        (),
        (0.25, 0.25),
        (8611.0 / 62500.0, -1743.0 / 31250.0, 0.25),
        (
            5012029.0 / 34652500.0,
            -654441.0 / 2922500.0,
            174375.0 / 388108.0,
            0.25,
        ),
        (
            15267082809.0 / 155376265600.0,
            -71443401.0 / 120774400.0,
            730878875.0 / 902184768.0,
            2285395.0 / 8070912.0,
            0.25,
        ),
        (
            82889.0 / 524892.0,
            0.0,
            15625.0 / 83664.0,
            69875.0 / 102672.0,
            -2260.0 / 8211.0,
            0.25,
        ),
    ),
    b=(
        82889.0 / 524892.0,
        0.0,
        15625.0 / 83664.0,
        69875.0 / 102672.0,
        -2260.0 / 8211.0,
        0.25,
    ),
    order=4,
    b_embedded=(
        4586570599.0 / 29645900160.0,
        0.0,
        178811875.0 / 945068544.0,
        814220225.0 / 1159782912.0,
        -3700637.0 / 11593932.0,
        61727.0 / 225920.0,
    ),
    embedded_order=3,
)


ARK436L2SA_TABLEAU = ImExButcherTableau(
    explicit=ARK436L2SA_EXPLICIT,
    implicit=ARK436L2SA_IMPLICIT,
    short_name="ARK436L2SA",
    full_name="ARK4(3)6L[2]SA",
)

KENNEDY_CARPENTER43_6_TABLEAU = ARK436L2SA_TABLEAU


class SchemeKennedyCarpenter43_6(SchemeBaseImExAdaptive):
    """Adaptive Kennedy-Carpenter ARK4(3)6L[2]SA IMEX method.

    This embedded fourth-order additive Runge-Kutta pair is a practical IMEX
    method with a richer stage structure than ARK324L2SA while keeping the
    method size modest enough for routine use.

    The scheme owns the public call routing and adaptive accept/reject loop.
    The `ImExStepper` owns the explicit/implicit stage machinery and keeps
    diagonal implicit solves inside the configured resolvent boundary.

    Further reading: Kennedy and Carpenter, Applied Numerical Mathematics 44,
    2003; SUNDIALS ARKODE Butcher tables.
    """

    __slots__ = (
        "call_pure",
        "resolvent",
        "stepper",
        "tableau_guard",
    )

    descriptor = SchemeDescriptor("KC43-6", "Kennedy-Carpenter 4(3) 6-stage")
    tableau = KENNEDY_CARPENTER43_6_TABLEAU

    def __init__(
        self,
        derivative: ImExDerivative,
        workbench: Workbench,
        resolvent: Resolvent,
        regulator: Regulator | None = None,
    ) -> None:
        super().__init__(derivative, workbench, regulator)

        self.tableau_guard = ResolventTableauGuard("KennedyCarpenter43_6", self.tableau)
        if resolvent is None:
            raise TypeError("KennedyCarpenter43_6 requires an explicit resolvent.")

        self.resolvent = resolvent
        self.tableau_guard(self.resolvent)

        self.stepper = ImExStepper(
            derivative,
            self.workspace,
            self.resolvent,
            self.tableau,
        )

        self.call_pure = self.call_generic
        self.refresh_call()

    def refresh_call(self) -> None:
        if not self.adaptive.runtime_bound:
            self.redirect_call = self.call_bind
            return

        self.redirect_call = (
            self.call_monitored
            if self.adaptive.monitor is not None
            else self.call_pure
        )

    def __call__(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        return self.redirect_call(interval, state, executor)

    def call_bind(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        self.assign_executor(executor)
        return self.redirect_call(interval, state, executor)

    def call_monitored(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        accepted_dt = self.call_pure(interval, state, executor)

        report = self.adaptive.report()
        monitor = self.adaptive.monitor
        if monitor is not None:
            monitor(
                MonitorStep(
                    scheme=self.short_name,
                    t_start=report.t_start,
                    t_end=report.t_end,
                    proposed_dt=report.proposed_dt,
                    accepted_dt=report.accepted_dt,
                    next_dt=report.next_dt,
                    error_ratio=report.error_ratio,
                    rejection_count=report.rejection_count,
                )
            )

        return accepted_dt

    def call_generic(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        del executor

        proposal = self.adaptive.propose_step(interval)
        if proposal.remaining <= 0.0:
            self.adaptive.record_stopped(interval)
            return 0.0

        stepper = self.stepper
        workspace = self.workspace
        apply_delta = workspace.apply_delta
        ratio = self.adaptive.ratio
        assert ratio is not None

        remaining = proposal.remaining
        dt = proposal.dt
        proposed_dt = proposal.proposed_dt
        t_start = proposal.t_start
        rejection_count = 0

        while True:
            try:
                (
                    delta_high,
                    error,
                    delta_high_norm,
                    error_norm,
                ) = stepper.step(
                    interval,
                    state,
                    dt,
                    include_norms=True,
                )
            except ResolventError:
                rejection_count += 1
                dt = self.adaptive.rejected_step(
                    dt,
                    1.0,
                    remaining,
                    self.short_name,
                )
                continue

            assert error is not None
            assert delta_high_norm is not None
            assert error_norm is not None

            error_ratio = ratio(error_norm, delta_high_norm)
            if error_ratio <= 1.0:
                break

            rejection_count += 1
            dt = self.adaptive.rejected_step(
                dt,
                error_ratio,
                remaining,
                self.short_name,
            )

        accepted_dt = dt
        remaining_after = remaining - accepted_dt
        next_dt = self.adaptive.accepted_next_step(
            accepted_dt,
            error_ratio,
            remaining_after,
        )

        interval.step = next_dt
        apply_delta(delta_high, state)

        report = self.adaptive.record_accepted(
            accepted_dt=accepted_dt,
            t_start=t_start,
            proposed_dt=proposed_dt,
            next_dt=next_dt,
            error_ratio=error_ratio,
            rejection_count=rejection_count,
        )
        return report.accepted_dt


__all__ = [
    "ARK436L2SA_EXPLICIT",
    "ARK436L2SA_IMPLICIT",
    "ARK436L2SA_TABLEAU",
    "KENNEDY_CARPENTER43_6_TABLEAU",
    "SchemeKennedyCarpenter43_6",
]