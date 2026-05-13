from __future__ import annotations

from dataclasses import dataclass
import warnings

import pytest

from stark import Executor, Interval, Tolerance
from stark.accelerators import Accelerator
from stark.resolvents import ResolventCoupledPicard, ResolventPicard
from stark.resolvents.policy import ResolventPolicy
from stark.schemes.explicit_adaptive.bogacki_shampine import SchemeBogackiShampine
from stark.schemes.explicit_fixed.euler import SchemeEuler
from stark.schemes.explicit_fixed.heun import SchemeHeun
from stark.schemes.explicit_fixed.rk4 import SchemeRK4
from stark.schemes.implicit_adaptive.bdf2 import SchemeBDF2
from stark.schemes.implicit_adaptive.kvaerno3 import SchemeKvaerno3
from stark.schemes.implicit_adaptive.kvaerno4 import SchemeKvaerno4
from stark.schemes.implicit_adaptive.sdirk21 import SchemeSDIRK21
from stark.schemes.implicit_fixed.backward_euler import SchemeBackwardEuler
from stark.schemes.implicit_fixed.crank_nicolson import SchemeCrankNicolson
from stark.schemes.implicit_fixed.crouzeix_dirk3 import SchemeCrouzeixDIRK3
from stark.schemes.implicit_fixed.gauss_legendre4 import SchemeGaussLegendre4
from stark.schemes.implicit_fixed.implicit_midpoint import SchemeImplicitMidpoint
from stark.schemes.implicit_fixed.lobatto_iiic4 import SchemeLobattoIIIC4
from stark.schemes.implicit_fixed.radau_iia5 import SchemeRadauIIA5
from stark.schemes.imex_fixed.euler import SchemeIMEXEuler
from stark.schemes.imex_adaptive.kennedy_carpenter32 import SchemeKennedyCarpenter32
from stark.schemes.imex_adaptive.kennedy_carpenter43_6 import SchemeKennedyCarpenter43_6
from stark.schemes.imex_adaptive.kennedy_carpenter43_7 import SchemeKennedyCarpenter43_7
from stark.schemes.imex_adaptive.kennedy_carpenter54 import SchemeKennedyCarpenter54
from stark.schemes.imex_adaptive.kennedy_carpenter54b import SchemeKennedyCarpenter54b

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

@dataclass(slots=True)
class SplitDerivative:
    explicit: object
    implicit: object

def exponential_growth(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval
    out.value = state.value


def constant_rhs(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval, state
    out.value = 1.0


def zero_rhs(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval, state
    out.value = 0.0


def make_one_stage_resolvent(scheme_cls, workbench: ScalarWorkbench) -> ResolventPicard:
    return ResolventPicard(
        constant_rhs,
        workbench,
        tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=ResolventPolicy(max_iterations=8),
        accelerator=Accelerator.none(),
        tableau=scheme_cls.tableau,
    )


def make_bdf2_resolvent(workbench: ScalarWorkbench) -> ResolventPicard:
    return ResolventPicard(
        constant_rhs,
        workbench,
        tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=ResolventPolicy(max_iterations=8),
        accelerator=Accelerator.none(),
        tableau=None,
    )


def make_coupled_resolvent(
    scheme_cls,
    workbench: ScalarWorkbench,
) -> ResolventCoupledPicard:
    return ResolventCoupledPicard(
        constant_rhs,
        workbench,
        tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=ResolventPolicy(max_iterations=8),
        accelerator=Accelerator.none(),
        tableau=scheme_cls.tableau,
    )


def make_implicit_fixed_scheme(scheme_cls):
    workbench = ScalarWorkbench()
    return scheme_cls(
        constant_rhs,
        workbench,
        resolvent=make_one_stage_resolvent(scheme_cls, workbench),
    )


def make_coupled_implicit_fixed_scheme(scheme_cls):
    workbench = ScalarWorkbench()
    return scheme_cls(
        constant_rhs,
        workbench,
        resolvent=make_coupled_resolvent(scheme_cls, workbench),
    )


def make_implicit_adaptive_scheme(scheme_cls):
    workbench = ScalarWorkbench()
    return scheme_cls(
        constant_rhs,
        workbench,
        resolvent=make_one_stage_resolvent(scheme_cls, workbench),
    )


def make_bdf2_scheme() -> SchemeBDF2:
    workbench = ScalarWorkbench()
    return SchemeBDF2(
        constant_rhs,
        workbench,
        resolvent=make_bdf2_resolvent(workbench),
    )

def make_imex_fixed_scheme() -> SchemeIMEXEuler:
    workbench = ScalarWorkbench()
    implicit = constant_rhs
    derivative = SplitDerivative(
        explicit=zero_rhs,
        implicit=implicit,
    )
    resolvent = ResolventPicard(
        implicit,
        workbench,
        tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=ResolventPolicy(max_iterations=8),
        accelerator=Accelerator.none(),
        tableau=SchemeIMEXEuler.tableau,
    )
    return SchemeIMEXEuler(
        derivative,
        workbench,
        resolvent=resolvent,
    )

def make_kennedy_carpenter32_scheme():
    workbench = ScalarWorkbench()
    derivative = SplitDerivative(
        explicit=zero_rhs,
        implicit=zero_rhs,
    )
    resolvent = ResolventPicard(
        zero_rhs,
        workbench,
        tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=ResolventPolicy(max_iterations=8),
        accelerator=Accelerator.none(),
        tableau=SchemeKennedyCarpenter32.tableau,
    )
    return SchemeKennedyCarpenter32(
        derivative,
        workbench,
        resolvent=resolvent,
    )

def make_kennedy_carpenter43_6_scheme():
    workbench = ScalarWorkbench()
    derivative = SplitDerivative(
        explicit=zero_rhs,
        implicit=zero_rhs,
    )
    resolvent = ResolventPicard(
        zero_rhs,
        workbench,
        tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=ResolventPolicy(max_iterations=8),
        accelerator=Accelerator.none(),
        tableau=SchemeKennedyCarpenter43_6.tableau,
    )
    return SchemeKennedyCarpenter43_6(
        derivative,
        workbench,
        resolvent=resolvent,
    )

def make_kennedy_carpenter43_7_scheme():
    workbench = ScalarWorkbench()
    derivative = SplitDerivative(
        explicit=zero_rhs,
        implicit=zero_rhs,
    )
    resolvent = ResolventPicard(
        zero_rhs,
        workbench,
        tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=ResolventPolicy(max_iterations=8),
        accelerator=Accelerator.none(),
        tableau=SchemeKennedyCarpenter43_7.tableau,
    )
    return SchemeKennedyCarpenter43_7(
        derivative,
        workbench,
        resolvent=resolvent,
    )

def make_kennedy_carpenter54_scheme():
    workbench = ScalarWorkbench()
    derivative = SplitDerivative(
        explicit=zero_rhs,
        implicit=zero_rhs,
    )
    resolvent = ResolventPicard(
        zero_rhs,
        workbench,
        tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=ResolventPolicy(max_iterations=8),
        accelerator=Accelerator.none(),
        tableau=SchemeKennedyCarpenter54.tableau,
    )
    return SchemeKennedyCarpenter54(
        derivative,
        workbench,
        resolvent=resolvent,
    )

def public_contract_schemes():
    return [
        SchemeRK4(exponential_growth, ScalarWorkbench()),
        SchemeBogackiShampine(zero_rhs, ScalarWorkbench()),
        make_implicit_fixed_scheme(SchemeBackwardEuler),
        make_implicit_fixed_scheme(SchemeImplicitMidpoint),
        make_implicit_fixed_scheme(SchemeCrankNicolson),
        make_implicit_fixed_scheme(SchemeCrouzeixDIRK3),
        make_coupled_implicit_fixed_scheme(SchemeGaussLegendre4),
        make_coupled_implicit_fixed_scheme(SchemeLobattoIIIC4),
        make_coupled_implicit_fixed_scheme(SchemeRadauIIA5),
        make_implicit_adaptive_scheme(SchemeSDIRK21),
        make_implicit_adaptive_scheme(SchemeKvaerno3),
        make_implicit_adaptive_scheme(SchemeKvaerno4),
        make_bdf2_scheme(),
        make_imex_fixed_scheme(),
        make_kennedy_carpenter32_scheme(),
        make_kennedy_carpenter43_6_scheme(),
        make_kennedy_carpenter43_7_scheme(),
        make_kennedy_carpenter54_scheme(),
    ]


def test_fixed_scheme_call_returns_accepted_dt() -> None:
    scheme = SchemeRK4(exponential_growth, ScalarWorkbench())
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = ScalarState(1.0)

    accepted_dt = scheme(interval, state, Executor())

    assert accepted_dt == pytest.approx(0.125)
    assert state.value == pytest.approx(1.133148193359375)


def test_fixed_scheme_call_clips_to_remaining_interval() -> None:
    scheme = SchemeRK4(exponential_growth, ScalarWorkbench())
    interval = Interval(present=0.2, step=0.125, stop=0.25)
    state = ScalarState(1.0)

    accepted_dt = scheme(interval, state, Executor())

    assert accepted_dt == pytest.approx(0.05)
    assert state.value == pytest.approx(1.05127109375)


def test_adaptive_scheme_call_returns_accepted_dt() -> None:
    scheme = SchemeBogackiShampine(zero_rhs, ScalarWorkbench())
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(2.0)
    executor = Executor(tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))

    accepted_dt = scheme(interval, state, executor)

    assert accepted_dt == pytest.approx(0.1)
    assert state.value == pytest.approx(2.0)


def test_adaptive_scheme_call_clips_next_accepted_dt_to_remaining_interval() -> None:
    scheme = SchemeBogackiShampine(zero_rhs, ScalarWorkbench())
    interval = Interval(present=0.1, step=1.0, stop=0.3)
    state = ScalarState(2.0)
    executor = Executor(tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))

    accepted_dt = scheme(interval, state, executor)

    assert accepted_dt == pytest.approx(0.2)
    assert state.value == pytest.approx(2.0)


@pytest.mark.parametrize("scheme", public_contract_schemes())
def test_snapshot_state_works_through_scheme_object(scheme) -> None:
    state = ScalarState(3.0)

    snapshot = scheme.snapshot_state(state)

    assert snapshot is not state
    assert snapshot.value == pytest.approx(3.0)

    state.value = 9.0

    assert snapshot.value == pytest.approx(3.0)


@pytest.mark.parametrize("scheme", public_contract_schemes())
def test_set_apply_delta_safety_works_through_scheme_object(scheme) -> None:
    # This is a public contract guard. It deliberately avoids asserting private
    # workspace/stepper internals; the important point for the refactor is that
    # schemes continue to expose the safety switch directly.
    scheme.set_apply_delta_safety(False)
    scheme.set_apply_delta_safety(True)

    snapshot = scheme.snapshot_state(ScalarState(1.0))

    assert snapshot.value == pytest.approx(1.0)


def test_self_contained_scheme_exemplars_own_public_call_method() -> None:
    assert "__call__" in SchemeEuler.__dict__
    assert "__call__" in SchemeRK4.__dict__
    assert "__call__" in SchemeBogackiShampine.__dict__
    assert "__call__" in SchemeBackwardEuler.__dict__
    assert "__call__" in SchemeImplicitMidpoint.__dict__
    assert "__call__" in SchemeCrankNicolson.__dict__
    assert "__call__" in SchemeCrouzeixDIRK3.__dict__
    assert "__call__" in SchemeGaussLegendre4.__dict__
    assert "__call__" in SchemeLobattoIIIC4.__dict__
    assert "__call__" in SchemeRadauIIA5.__dict__
    assert "__call__" in SchemeSDIRK21.__dict__
    assert "__call__" in SchemeKvaerno3.__dict__
    assert "__call__" in SchemeKvaerno4.__dict__
    assert "__call__" in SchemeBDF2.__dict__
    assert "__call__" in SchemeIMEXEuler.__dict__
    assert "__call__" in SchemeKennedyCarpenter32.__dict__
    assert "__call__" in SchemeKennedyCarpenter43_6.__dict__
    assert "__call__" in SchemeKennedyCarpenter43_7.__dict__
    assert "__call__" in SchemeKennedyCarpenter54.__dict__


def test_converted_fixed_explicit_scheme_still_works_without_warning() -> None:
    scheme = SchemeHeun(exponential_growth, ScalarWorkbench())
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = ScalarState(1.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        accepted_dt = scheme(interval, state, Executor())

    assert accepted_dt == pytest.approx(0.125)
    assert state.value > 1.0
    assert caught == []