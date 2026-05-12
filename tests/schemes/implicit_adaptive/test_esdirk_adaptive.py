from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark import Executor, Integrator, Interval, Marcher, Tolerance
from stark.accelerators import Accelerator
from stark.monitor import Monitor
from stark.resolvents import ResolventPicard
from stark.resolvents.policy import ResolventPolicy
from stark.schemes.implicit_adaptive.kvaerno3 import SchemeKvaerno3
from stark.schemes.implicit_adaptive.kvaerno4 import SchemeKvaerno4
from stark.schemes.implicit_adaptive.sdirk21 import SchemeSDIRK21


@dataclass(slots=True)
class ScalarState:
    value: float = 0.0


@dataclass(slots=True)
class ScalarTranslation:
    value: float = 0.0

    def __call__(self, origin: ScalarState, result: ScalarState) -> None:
        result.value = origin.value + self.value

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: ScalarTranslation) -> ScalarTranslation:
        return ScalarTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> ScalarTranslation:
        return ScalarTranslation(scalar * self.value)


class ScalarWorkbench:
    def allocate_state(self) -> ScalarState:
        return ScalarState()

    def copy_state(self, dst: ScalarState, src: ScalarState) -> None:
        dst.value = src.value

    def allocate_translation(self) -> ScalarTranslation:
        return ScalarTranslation()


def constant_rhs(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval, state
    out.value = 1.0


def make_scheme(scheme_cls):
    workbench = ScalarWorkbench()
    resolvent = ResolventPicard(
        constant_rhs,
        workbench,
        tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=ResolventPolicy(max_iterations=8),
        accelerator=Accelerator.none(),
        tableau=scheme_cls.tableau,
    )
    return scheme_cls(
        constant_rhs,
        workbench,
        resolvent=resolvent,
    )


def tight_executor() -> Executor:
    return Executor(tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeSDIRK21,
        SchemeKvaerno3,
        SchemeKvaerno4,
    ],
)
def test_esdirk_adaptive_scheme_owns_its_public_call_method(scheme_cls) -> None:
    assert "__call__" in scheme_cls.__dict__


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeSDIRK21,
        SchemeKvaerno3,
        SchemeKvaerno4,
    ],
)
def test_esdirk_adaptive_default_call_path_is_scheme_owned_generic_call(
    scheme_cls,
) -> None:
    scheme = make_scheme(scheme_cls)

    assert scheme.call_pure.__self__ is scheme
    assert scheme.call_pure.__func__ is scheme_cls.call_generic

    # Adaptive schemes bind executor runtime lazily on first public call.
    assert scheme.redirect_call.__self__ is scheme
    assert scheme.redirect_call.__func__ is scheme.call_bind.__func__


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeSDIRK21,
        SchemeKvaerno3,
        SchemeKvaerno4,
    ],
)
def test_esdirk_adaptive_public_call_uses_redirect_call(scheme_cls) -> None:
    scheme = make_scheme(scheme_cls)
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(0.0)

    def replacement_call(
        replacement_interval: Interval,
        replacement_state: ScalarState,
        replacement_executor: Executor,
    ) -> float:
        del replacement_interval, replacement_executor
        replacement_state.value = 42.0
        return 0.03125

    scheme.redirect_call = replacement_call

    accepted_dt = scheme(interval, state, tight_executor())

    assert accepted_dt == pytest.approx(0.03125)
    assert state.value == pytest.approx(42.0)


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeSDIRK21,
        SchemeKvaerno3,
        SchemeKvaerno4,
    ],
)
def test_esdirk_adaptive_call_returns_accepted_dt_and_updates_next_step(
    scheme_cls,
) -> None:
    scheme = make_scheme(scheme_cls)
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(0.0)

    accepted_dt = scheme(interval, state, tight_executor())

    assert accepted_dt == pytest.approx(0.1)
    assert state.value == pytest.approx(0.1)
    assert interval.step == pytest.approx(0.2)


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeSDIRK21,
        SchemeKvaerno3,
        SchemeKvaerno4,
    ],
)
def test_esdirk_adaptive_call_clips_to_remaining_interval(scheme_cls) -> None:
    scheme = make_scheme(scheme_cls)
    interval = Interval(present=0.1, step=1.0, stop=0.3)
    state = ScalarState(0.0)

    accepted_dt = scheme(interval, state, tight_executor())

    assert accepted_dt == pytest.approx(0.2)
    assert state.value == pytest.approx(0.2)
    assert interval.step == pytest.approx(0.0)


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeSDIRK21,
        SchemeKvaerno3,
        SchemeKvaerno4,
    ],
)
def test_esdirk_adaptive_snapshot_and_safety_are_exposed_through_scheme(
    scheme_cls,
) -> None:
    scheme = make_scheme(scheme_cls)
    state = ScalarState(3.0)

    snapshot = scheme.snapshot_state(state)

    assert snapshot is not state
    assert snapshot.value == pytest.approx(3.0)

    state.value = 9.0

    assert snapshot.value == pytest.approx(3.0)

    scheme.set_apply_delta_safety(False)
    scheme.set_apply_delta_safety(True)


@pytest.mark.parametrize(
    ("scheme_cls", "scheme_name"),
    [
        (SchemeSDIRK21, "SDIRK21"),
        (SchemeKvaerno3, "Kvaerno3"),
        (SchemeKvaerno4, "Kvaerno4"),
    ],
)
def test_esdirk_adaptive_monitoring_records_existing_adaptive_fields(
    scheme_cls,
    scheme_name: str,
) -> None:
    scheme = make_scheme(scheme_cls)
    marcher = Marcher(scheme, tight_executor())
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(0.0)
    monitor = Monitor()

    list(Integrator().live_monitored(marcher, interval, state, monitor))

    assert len(monitor.steps) == 2

    first = monitor.steps[0]
    second = monitor.steps[1]

    assert first.scheme == scheme_name
    assert first.t_start == pytest.approx(0.0)
    assert first.t_end == pytest.approx(0.1)
    assert first.proposed_dt == pytest.approx(0.1)
    assert first.accepted_dt == pytest.approx(0.1)
    assert first.next_dt == pytest.approx(0.2)
    assert first.error_ratio >= 0.0
    assert first.error_ratio < 1.0e-6
    assert first.rejection_count == 0

    assert second.scheme == scheme_name
    assert second.t_start == pytest.approx(0.1)
    assert second.t_end == pytest.approx(0.3)
    assert second.proposed_dt == pytest.approx(0.2)
    assert second.accepted_dt == pytest.approx(0.2)
    assert second.next_dt == pytest.approx(0.0)
    assert second.error_ratio >= 0.0
    assert second.error_ratio < 1.0e-6
    assert second.rejection_count == 0