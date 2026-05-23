from __future__ import annotations

from stark.algebraist.classic import Algebraist
from stark.contracts import ImExDerivative, IntervalLike, Resolvent, State, Workbench
from stark.execution.executor import Executor
from stark.execution.regulator import Regulator
from stark.machinery.stage_solve.workers import ImExStepper
from stark.resolvents.support.failure import ResolventError
from stark.resolvents.support.guard import ResolventTableauGuard
from stark.schemes.support.descriptor import SchemeDescriptor
from stark.schemes.support import (
    SchemeStepControl,
    initialise_adaptive_runtime,
    initialise_imex_support,
    refresh_adaptive_call,
    with_adaptive_runtime_methods,
    with_imex_workspace_methods,
    with_scheme_display,
)
from stark.schemes.support.tableau import ButcherTableau, ButcherTableauImex


ARK548L2SA_EXPLICIT = ButcherTableau(
    c=(
        0.0,
        41.0 / 100.0,
        2935347310677.0 / 11292855782101.0,
        1426016391358.0 / 7196633302097.0,
        92.0 / 100.0,
        24.0 / 100.0,
        3.0 / 5.0,
        1.0,
    ),
    a=(
        (),
        (41.0 / 100.0,),
        (367902744464.0 / 2072280473677.0, 677623207551.0 / 8224143866563.0),
        (
            1268023523408.0 / 10340822734521.0,
            0.0,
            1029933939417.0 / 13636558850479.0,
        ),
        (
            14463281900351.0 / 6315353703477.0,
            0.0,
            66114435211212.0 / 5879490589093.0,
            -54053170152839.0 / 4284798021562.0,
        ),
        (
            14090043504691.0 / 34967701212078.0,
            0.0,
            15191511035443.0 / 11219624916014.0,
            -18461159152457.0 / 12425892160975.0,
            -281667163811.0 / 9011619295870.0,
        ),
        (
            19230459214898.0 / 13134317526959.0,
            0.0,
            21275331358303.0 / 2942455364971.0,
            -38145345988419.0 / 4862620318723.0,
            -1.0 / 8.0,
            -1.0 / 8.0,
        ),
        (
            -19977161125411.0 / 11928030595625.0,
            0.0,
            -40795976796054.0 / 6384907823539.0,
            177454434618887.0 / 12078138498510.0,
            782672205425.0 / 8267701900261.0,
            -69563011059811.0 / 9646580694205.0,
            7356628210526.0 / 4942186776405.0,
        ),
    ),
    b=(
        -872700587467.0 / 9133579230613.0,
        0.0,
        0.0,
        22348218063261.0 / 9555858737531.0,
        -1143369518992.0 / 8141816002931.0,
        -39379526789629.0 / 19018526304540.0,
        32727382324388.0 / 42900044865799.0,
        41.0 / 200.0,
    ),
    order=5,
    b_embedded=(
        -975461918565.0 / 9796059967033.0,
        0.0,
        0.0,
        78070527104295.0 / 32432590147079.0,
        -548382580838.0 / 3424219808633.0,
        -33438840321285.0 / 15594753105479.0,
        3629800801594.0 / 4656183773603.0,
        4035322873751.0 / 18575991585200.0,
    ),
    embedded_order=4,
)


ARK548L2SA_IMPLICIT = ButcherTableau(
    c=ARK548L2SA_EXPLICIT.c,
    a=(
        (),
        (41.0 / 200.0, 41.0 / 200.0),
        (
            41.0 / 400.0,
            -567603406766.0 / 11931857230679.0,
            41.0 / 200.0,
        ),
        (
            683785636431.0 / 9252920307686.0,
            0.0,
            -110385047103.0 / 1367015193373.0,
            41.0 / 200.0,
        ),
        (
            3016520224154.0 / 10081342136671.0,
            0.0,
            30586259806659.0 / 12414158314087.0,
            -22760509404356.0 / 11113319521817.0,
            41.0 / 200.0,
        ),
        (
            218866479029.0 / 1489978393911.0,
            0.0,
            638256894668.0 / 5436446318841.0,
            -1179710474555.0 / 5321154724896.0,
            -60928119172.0 / 8023461067671.0,
            41.0 / 200.0,
        ),
        (
            1020004230633.0 / 5715676835656.0,
            0.0,
            25762820946817.0 / 25263940353407.0,
            -2161375909145.0 / 9755907335909.0,
            -211217309593.0 / 5846859502534.0,
            -4269925059573.0 / 7827059040749.0,
            41.0 / 200.0,
        ),
        (
            -872700587467.0 / 9133579230613.0,
            0.0,
            0.0,
            22348218063261.0 / 9555858737531.0,
            -1143369518992.0 / 8141816002931.0,
            -39379526789629.0 / 19018526304540.0,
            32727382324388.0 / 42900044865799.0,
            41.0 / 200.0,
        ),
    ),
    b=ARK548L2SA_EXPLICIT.b,
    order=5,
    b_embedded=ARK548L2SA_EXPLICIT.b_embedded,
    embedded_order=4,
)


ARK548L2SA_TABLEAU = ButcherTableauImex(
    explicit=ARK548L2SA_EXPLICIT,
    implicit=ARK548L2SA_IMPLICIT,
    short_name="ARK548L2SA",
    full_name="ARK5(4)8L[2]SA",
)

KENNEDY_CARPENTER54_TABLEAU = ARK548L2SA_TABLEAU


@with_scheme_display
@with_adaptive_runtime_methods
@with_imex_workspace_methods
class SchemeKennedyCarpenter54:
    """Adaptive Kennedy-Carpenter ARK5(4)8L[2]SA IMEX method.

    This is the original eight-stage fifth-order Kennedy-Carpenter additive
    Runge-Kutta pair. It extends the current IMEX coverage into the practical
    fifth-order regime without leaving the additive RK family.

    The scheme owns the public call routing and adaptive accept/reject loop.
    The `ImExStepper` owns the explicit/implicit stage machinery and keeps
    diagonal implicit solves inside the configured resolvent boundary.

    Further reading: Kennedy and Carpenter, Applied Numerical Mathematics 44,
    2003; SUNDIALS ARKODE Butcher tables.
    """

    # Assigned by initialise_adaptive_runtime from stark.schemes.support.
    step_control: SchemeStepControl

    __slots__ = (
        "step_control",
        "call_pure",
        "redirect_call",
        "resolvent",
        "stepper",
        "tableau_guard",
        "workspace",
    )

    descriptor = SchemeDescriptor("KC54", "Kennedy-Carpenter 5(4)")
    tableau = KENNEDY_CARPENTER54_TABLEAU

    def __init__(
        self,
        derivative: ImExDerivative,
        workbench: Workbench,
        resolvent: Resolvent,
        regulator: Regulator | None = None,
        *,
        algebraist: Algebraist | None = None,
    ) -> None:
        initialise_imex_support(self, derivative, workbench)
        initialise_adaptive_runtime(self, regulator)

        self.tableau_guard = ResolventTableauGuard("KennedyCarpenter54", self.tableau)
        if resolvent is None:
            raise TypeError("KennedyCarpenter54 requires an explicit resolvent.")

        self.resolvent = resolvent
        self.tableau_guard(self.resolvent)

        self.stepper = ImExStepper(
            derivative,
            self.workspace,
            self.resolvent,
            self.tableau,
        )
        self.stepper.require_embedded(type(self).__name__)

        self.call_pure = self.call_inline
        refresh_adaptive_call(self)

        if algebraist is not None:
            self.use_specialists(algebraist)

    @staticmethod
    def default_regulator() -> Regulator:
        return Regulator(error_exponent=0.2)

    def __call__(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        return self.redirect_call(interval, state, executor)

    def use_specialists(self, algebraist: Algebraist) -> None:
        self.stepper.bind_algebraist(algebraist)
        self.call_pure = self.call_specialized
        refresh_adaptive_call(self)

    def call_inline(
        self,
        interval: IntervalLike,
        state: State,
        executor: Executor,
    ) -> float:
        del executor

        proposal = self.step_control.propose_step(interval)
        if proposal.remaining <= 0.0:
            self.step_control.record_stopped(interval)
            return 0.0

        stepper = self.stepper
        workspace = self.workspace
        apply_delta = workspace.apply_delta
        ratio = self.step_control.ratio
        remaining = proposal.remaining
        dt = proposal.dt
        proposed_dt = proposal.proposed_dt
        t_start = proposal.t_start
        rejection_count = 0
        scheme_name = self.tableau.short_name or type(self).__name__

        while True:
            try:
                delta_high, delta_high_norm, error_norm = stepper.step_adaptive(
                    interval,
                    state,
                    dt,
                )
            except ResolventError:
                rejection_count += 1
                dt = self.step_control.rejected_step(
                    dt,
                    1.0,
                    remaining,
                    scheme_name,
                )
                continue

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
        apply_delta(delta_high, state)

        report = self.step_control.record_accepted(
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
        executor: Executor,
    ) -> float:
        del executor

        proposal = self.step_control.propose_step(interval)
        if proposal.remaining <= 0.0:
            self.step_control.record_stopped(interval)
            return 0.0

        stepper = self.stepper
        workspace = self.workspace
        apply_delta = workspace.apply_delta
        ratio = self.step_control.ratio
        remaining = proposal.remaining
        dt = proposal.dt
        proposed_dt = proposal.proposed_dt
        t_start = proposal.t_start
        rejection_count = 0
        scheme_name = self.tableau.short_name or type(self).__name__

        while True:
            try:
                delta_high, delta_high_norm, error_norm = stepper.step_adaptive_algebraist(
                    interval,
                    state,
                    dt,
                )
            except ResolventError:
                rejection_count += 1
                dt = self.step_control.rejected_step(
                    dt,
                    1.0,
                    remaining,
                    scheme_name,
                )
                continue

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
        apply_delta(delta_high, state)

        report = self.step_control.record_accepted(
            accepted_dt=accepted_dt,
            t_start=t_start,
            proposed_dt=proposed_dt,
            next_dt=next_dt,
            error_ratio=error_ratio,
            rejection_count=rejection_count,
        )
        return report.accepted_dt


__all__ = [
    "ARK548L2SA_EXPLICIT",
    "ARK548L2SA_IMPLICIT",
    "ARK548L2SA_TABLEAU",
    "KENNEDY_CARPENTER54_TABLEAU",
    "SchemeKennedyCarpenter54",
]
