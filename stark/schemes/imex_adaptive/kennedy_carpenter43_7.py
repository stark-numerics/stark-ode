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


ARK437L2SA_EXPLICIT = ButcherTableau(
    c=(
        0.0,
        247.0 / 1000.0,
        4276536705230.0 / 10142255878289.0,
        67.0 / 200.0,
        3.0 / 40.0,
        7.0 / 10.0,
        1.0,
    ),
    a=(
        (),
        (247.0 / 1000.0,),
        (247.0 / 4000.0, 2694949928731.0 / 7487940209513.0),
        (
            464650059369.0 / 8764239774964.0,
            878889893998.0 / 2444806327765.0,
            -952945855348.0 / 12294611323341.0,
        ),
        (
            476636172619.0 / 8159180917465.0,
            -1271469283451.0 / 7793814740893.0,
            -859560642026.0 / 4356155882851.0,
            1723805262919.0 / 4571918432560.0,
        ),
        (
            6338158500785.0 / 11769362343261.0,
            -4970555480458.0 / 10924838743837.0,
            3326578051521.0 / 2647936831840.0,
            -880713585975.0 / 1841400956686.0,
            -1428733748635.0 / 8843423958496.0,
        ),
        (
            760814592956.0 / 3276306540349.0,
            760814592956.0 / 3276306540349.0,
            -47223648122716.0 / 6934462133451.0,
            71187472546993.0 / 9669769126921.0,
            -13330509492149.0 / 9695768672337.0,
            11565764226357.0 / 8513123442827.0,
        ),
    ),
    b=(
        0.0,
        0.0,
        9164257142617.0 / 17756377923965.0,
        -10812980402763.0 / 74029279521829.0,
        1335994250573.0 / 5691609445217.0,
        2273837961795.0 / 8368240463276.0,
        247.0 / 2000.0,
    ),
    order=4,
    b_embedded=(
        0.0,
        0.0,
        4469248916618.0 / 8635866897933.0,
        -621260224600.0 / 4094290005349.0,
        696572312987.0 / 2942599194819.0,
        1532940081127.0 / 5565293938103.0,
        2441.0 / 20000.0,
    ),
    embedded_order=3,
)


ARK437L2SA_IMPLICIT = ButcherTableau(
    c=(
        0.0,
        247.0 / 1000.0,
        4276536705230.0 / 10142255878289.0,
        67.0 / 200.0,
        3.0 / 40.0,
        7.0 / 10.0,
        1.0,
    ),
    a=(
        (),
        (1235.0 / 10000.0, 1235.0 / 10000.0),
        (
            624185399699.0 / 4186980696204.0,
            624185399699.0 / 4186980696204.0,
            1235.0 / 10000.0,
        ),
        (
            1258591069120.0 / 10082082980243.0,
            1258591069120.0 / 10082082980243.0,
            -322722984531.0 / 8455138723562.0,
            1235.0 / 10000.0,
        ),
        (
            -436103496990.0 / 5971407786587.0,
            -436103496990.0 / 5971407786587.0,
            -2689175662187.0 / 11046760208243.0,
            4431412449334.0 / 12995360898505.0,
            1235.0 / 10000.0,
        ),
        (
            -2207373168298.0 / 14430576638973.0,
            -2207373168298.0 / 14430576638973.0,
            242511121179.0 / 3358618340039.0,
            3145666661981.0 / 7780404714551.0,
            5882073923981.0 / 14490790706663.0,
            1235.0 / 10000.0,
        ),
        (
            0.0,
            0.0,
            9164257142617.0 / 17756377923965.0,
            -10812980402763.0 / 74029279521829.0,
            1335994250573.0 / 5691609445217.0,
            2273837961795.0 / 8368240463276.0,
            1235.0 / 10000.0,
        ),
    ),
    b=(
        0.0,
        0.0,
        9164257142617.0 / 17756377923965.0,
        -10812980402763.0 / 74029279521829.0,
        1335994250573.0 / 5691609445217.0,
        2273837961795.0 / 8368240463276.0,
        1235.0 / 10000.0,
    ),
    order=4,
    b_embedded=(
        0.0,
        0.0,
        4469248916618.0 / 8635866897933.0,
        -621260224600.0 / 4094290005349.0,
        696572312987.0 / 2942599194819.0,
        1532940081127.0 / 5565293938103.0,
        2441.0 / 20000.0,
    ),
    embedded_order=3,
)


ARK437L2SA_TABLEAU = ImExButcherTableau(
    explicit=ARK437L2SA_EXPLICIT,
    implicit=ARK437L2SA_IMPLICIT,
    short_name="ARK437L2SA",
    full_name="ARK4(3)7L[2]SA",
)

KENNEDY_CARPENTER43_7_TABLEAU = ARK437L2SA_TABLEAU


class SchemeKennedyCarpenter43_7(SchemeBaseImExAdaptive):
    """Adaptive Kennedy-Carpenter ARK4(3)7L[2]SA IMEX method.

    This is the seven-stage fourth-order pair used by SUNDIALS as the default
    additive ARK method. It gives us a second practical fourth-order IMEX
    scheme with a different stage pattern, which helps shake out support-layer
    assumptions.

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

    descriptor = SchemeDescriptor("KC43-7", "Kennedy-Carpenter 4(3) 7-stage")
    tableau = KENNEDY_CARPENTER43_7_TABLEAU

    def __init__(
        self,
        derivative: ImExDerivative,
        workbench: Workbench,
        resolvent: Resolvent,
        regulator: Regulator | None = None,
    ) -> None:
        super().__init__(derivative, workbench, regulator)

        self.tableau_guard = ResolventTableauGuard("KennedyCarpenter43_7", self.tableau)
        if resolvent is None:
            raise TypeError("KennedyCarpenter43_7 requires an explicit resolvent.")

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

    def advance_body(self, interval: IntervalLike, state: State) -> None:
        """Compatibility bridge for the transitional adaptive base."""
        self.call_pure(interval, state, Executor())

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
    "ARK437L2SA_EXPLICIT",
    "ARK437L2SA_IMPLICIT",
    "ARK437L2SA_TABLEAU",
    "KENNEDY_CARPENTER43_7_TABLEAU",
    "SchemeKennedyCarpenter43_7",
]