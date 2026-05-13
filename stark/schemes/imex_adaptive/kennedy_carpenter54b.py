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


ARK548L2SAB_EXPLICIT = ButcherTableau(
    c=(
        0.0,
        4.0 / 9.0,
        6456083330201.0 / 8509243623797.0,
        1632083962415.0 / 14158861528103.0,
        6365430648612.0 / 17842476412687.0,
        18.0 / 25.0,
        191.0 / 200.0,
        1.0,
    ),
    a=(
        (),
        (4.0 / 9.0,),
        (
            2366667076620.0 / 8822750406821.0,
            2366667076620.0 / 8822750406821.0,
        ),
        (
            -257962897183.0 / 4451812247028.0,
            -257962897183.0 / 4451812247028.0,
            128530224461.0 / 14379561246022.0,
        ),
        (
            -486229321650.0 / 11227943450093.0,
            -486229321650.0 / 11227943450093.0,
            -225633144460.0 / 6633558740617.0,
            1741320951451.0 / 6824444397158.0,
        ),
        (
            621307788657.0 / 4714163060173.0,
            621307788657.0 / 4714163060173.0,
            -125196015625.0 / 3866852212004.0,
            940440206406.0 / 7593089888465.0,
            961109811699.0 / 6734810228204.0,
        ),
        (
            2036305566805.0 / 6583108094622.0,
            2036305566805.0 / 6583108094622.0,
            -3039402635899.0 / 4450598839912.0,
            -1829510709469.0 / 31102090912115.0,
            -286320471013.0 / 6931253422520.0,
            8651533662697.0 / 9642993110008.0,
        ),
        (
            0.0,
            0.0,
            3517720773327.0 / 20256071687669.0,
            4569610470461.0 / 17934693873752.0,
            2819471173109.0 / 11655438449929.0,
            3296210113763.0 / 10722700128969.0,
            -1142099968913.0 / 5710983926999.0,
        ),
    ),
    b=(
        0.0,
        0.0,
        3517720773327.0 / 20256071687669.0,
        4569610470461.0 / 17934693873752.0,
        2819471173109.0 / 11655438449929.0,
        3296210113763.0 / 10722700128969.0,
        -1142099968913.0 / 5710983926999.0,
        2.0 / 9.0,
    ),
    order=5,
    b_embedded=(
        0.0,
        0.0,
        520639020421.0 / 8300446712847.0,
        4550235134915.0 / 17827758688493.0,
        1482366381361.0 / 6201654941325.0,
        5551607622171.0 / 13911031047899.0,
        -5266607656330.0 / 36788968843917.0,
        1074053359553.0 / 5740751784926.0,
    ),
    embedded_order=4,
)


ARK548L2SAB_IMPLICIT = ButcherTableau(
    c=ARK548L2SAB_EXPLICIT.c,
    a=(
        (),
        (2.0 / 9.0, 2.0 / 9.0),
        (
            2366667076620.0 / 8822750406821.0,
            2366667076620.0 / 8822750406821.0,
            2.0 / 9.0,
        ),
        (
            -257962897183.0 / 4451812247028.0,
            -257962897183.0 / 4451812247028.0,
            128530224461.0 / 14379561246022.0,
            2.0 / 9.0,
        ),
        (
            -486229321650.0 / 11227943450093.0,
            -486229321650.0 / 11227943450093.0,
            -225633144460.0 / 6633558740617.0,
            1741320951451.0 / 6824444397158.0,
            2.0 / 9.0,
        ),
        (
            621307788657.0 / 4714163060173.0,
            621307788657.0 / 4714163060173.0,
            -125196015625.0 / 3866852212004.0,
            940440206406.0 / 7593089888465.0,
            961109811699.0 / 6734810228204.0,
            2.0 / 9.0,
        ),
        (
            2036305566805.0 / 6583108094622.0,
            2036305566805.0 / 6583108094622.0,
            -3039402635899.0 / 4450598839912.0,
            -1829510709469.0 / 31102090912115.0,
            -286320471013.0 / 6931253422520.0,
            8651533662697.0 / 9642993110008.0,
            2.0 / 9.0,
        ),
        (
            0.0,
            0.0,
            3517720773327.0 / 20256071687669.0,
            4569610470461.0 / 17934693873752.0,
            2819471173109.0 / 11655438449929.0,
            3296210113763.0 / 10722700128969.0,
            -1142099968913.0 / 5710983926999.0,
            2.0 / 9.0,
        ),
    ),
    b=ARK548L2SAB_EXPLICIT.b,
    order=5,
    b_embedded=ARK548L2SAB_EXPLICIT.b_embedded,
    embedded_order=4,
)


ARK548L2SAB_TABLEAU = ImExButcherTableau(
    explicit=ARK548L2SAB_EXPLICIT,
    implicit=ARK548L2SAB_IMPLICIT,
    short_name="ARK548L2SAb",
    full_name="ARK5(4)8L[2]SA(b)",
)

KENNEDY_CARPENTER54B_TABLEAU = ARK548L2SAB_TABLEAU


class SchemeKennedyCarpenter54b(SchemeBaseImExAdaptive):
    """Adaptive Kennedy-Carpenter ARK5(4)8L[2]SA(b) IMEX method.

    This is the later eight-stage fifth-order Kennedy-Carpenter variant used
    by SUNDIALS as the default additive fifth-order method. It gives us a
    second high-order IMEX ARK path with noticeably different stage data but
    the same broad API expectations.

    The scheme owns the public call routing and adaptive accept/reject loop.
    The `ImExStepper` owns the explicit/implicit stage machinery and keeps
    diagonal implicit solves inside the configured resolvent boundary.

    Further reading: Kennedy and Carpenter, Applied Numerical Mathematics 136,
    2019; SUNDIALS ARKODE Butcher tables.
    """

    __slots__ = (
        "call_pure",
        "resolvent",
        "stepper",
        "tableau_guard",
    )

    descriptor = SchemeDescriptor("KC54b", "Kennedy-Carpenter 5(4) b")
    tableau = KENNEDY_CARPENTER54B_TABLEAU

    def __init__(
        self,
        derivative: ImExDerivative,
        workbench: Workbench,
        resolvent: Resolvent,
        regulator: Regulator | None = None,
    ) -> None:
        super().__init__(derivative, workbench, regulator)

        self.tableau_guard = ResolventTableauGuard("KennedyCarpenter54b", self.tableau)
        if resolvent is None:
            raise TypeError("KennedyCarpenter54b requires an explicit resolvent.")

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

    @staticmethod
    def default_regulator() -> Regulator:
        return Regulator(error_exponent=0.2)

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
    "ARK548L2SAB_EXPLICIT",
    "ARK548L2SAB_IMPLICIT",
    "ARK548L2SAB_TABLEAU",
    "KENNEDY_CARPENTER54B_TABLEAU",
    "SchemeKennedyCarpenter54b",
]