from dataclasses import dataclass

import numpy as np
import pytest

from stark.algebraist import Algebraist, AlgebraistField, AlgebraistTableauPlanner
from stark.schemes.explicit_adaptive.cash_karp import RKCK_TABLEAU
from stark.schemes.explicit_fixed.euler import EULER_TABLEAU
from stark.schemes.explicit_fixed.rk4 import RK4_TABLEAU


@dataclass(frozen=True, slots=True)
class FakeButcherTableau:
    c: tuple[float, ...]
    a: tuple[tuple[float, ...], ...]
    b: tuple[float, ...]
    order: int
    b_embedded: tuple[float, ...] | None = None
    embedded_order: int | None = None
    short_name: str | None = None
    full_name: str | None = None


class FakeTranslation:
    def __init__(self, value):
        self.value = value


class FakeState:
    def __init__(self, value):
        self.value = value


def test_euler_tableau_plan_has_one_solution_term():
    tableau = AlgebraistTableauPlanner()(EULER_TABLEAU)

    assert [stage.term_count for stage in tableau.stages] == [0]
    assert tableau.solution.role == "solution"
    assert tableau.solution.coefficients == (1.0,)
    assert tableau.solution.term_indices == (0,)
    assert tableau.error is None
    assert not tableau.has_error
    assert tableau.order == 1


def test_rk4_tableau_plan_matches_nonzero_structure():
    tableau = AlgebraistTableauPlanner()(RK4_TABLEAU)

    assert [stage.coefficients for stage in tableau.stages] == [
        (),
        (0.5,),
        (0.5,),
        (1.0,),
    ]
    assert [stage.term_indices for stage in tableau.stages] == [
        (),
        (0,),
        (1,),
        (2,),
    ]
    assert tableau.solution.coefficients == (
        1.0 / 6.0,
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 6.0,
    )
    assert tableau.solution.term_indices == (0, 1, 2, 3)
    assert tableau.error is None


def test_cash_karp_tableau_plan_drops_zero_solution_and_error_terms():
    tableau = AlgebraistTableauPlanner()(RKCK_TABLEAU)
    assert tableau.error is not None

    assert tableau.solution.term_indices == (0, 2, 3, 5)
    assert tableau.solution.coefficients == (
        37.0 / 378.0,
        250.0 / 621.0,
        125.0 / 594.0,
        512.0 / 1771.0,
    )
    assert tableau.error.term_indices == (0, 2, 3, 4, 5)
    assert tableau.error.coefficients == (
        RKCK_TABLEAU.b[0] - RKCK_TABLEAU.b_embedded[0],
        RKCK_TABLEAU.b[2] - RKCK_TABLEAU.b_embedded[2],
        RKCK_TABLEAU.b[3] - RKCK_TABLEAU.b_embedded[3],
        RKCK_TABLEAU.b[4] - RKCK_TABLEAU.b_embedded[4],
        RKCK_TABLEAU.b[5] - RKCK_TABLEAU.b_embedded[5],
    )
    assert tableau.has_error
    assert tableau.order == 5
    assert tableau.embedded_order == 4


def test_planner_accepts_structural_butcher_tableau_like_object():
    fake = FakeButcherTableau(
        c=(0.0, 1.0),
        a=((), (1.0,)),
        b=(0.5, 0.5),
        order=2,
        short_name="Fake2",
    )

    tableau = AlgebraistTableauPlanner()(fake)

    assert tableau.stages[1].coefficients == (1.0,)
    assert tableau.solution.coefficients == (0.5, 0.5)
    assert tableau.short_name == "Fake2"


def test_planner_zero_tolerance_drops_near_zero_coefficients():
    fake = FakeButcherTableau(
        c=(0.0, 1.0),
        a=((), (1.0e-14,)),
        b=(1.0e-14, 1.0),
        order=1,
    )

    tableau = AlgebraistTableauPlanner(zero=1.0e-12)(fake)

    assert tableau.stages[1].coefficients == ()
    assert tableau.solution.coefficients == (1.0,)
    assert tableau.solution.term_indices == (1,)


def test_planner_rejects_inconsistent_tableau_shape():
    fake = FakeButcherTableau(
        c=(0.0, 1.0),
        a=((),),
        b=(1.0, 0.0),
        order=1,
    )

    with pytest.raises(ValueError, match="stage rows"):
        AlgebraistTableauPlanner()(fake)


def test_algebraist_binds_tableau_combinations_as_specific_calls():
    algebraist = Algebraist(fields=(AlgebraistField("value", "value"),))
    fake = FakeButcherTableau(
        c=(0.0, 1.0),
        a=((), (0.5,)),
        b=(0.25, 0.75),
        order=2,
    )
    calls = algebraist.bind_tableau(fake)
    out = FakeTranslation(np.zeros(2))
    k0 = FakeTranslation(np.array([2.0, 4.0]))
    k1 = FakeTranslation(np.array([6.0, 8.0]))

    calls.stages[1](out, 3.0, k0)
    np.testing.assert_allclose(out.value, np.array([3.0, 6.0]))

    calls.solution(out, 2.0, k0, k1)
    np.testing.assert_allclose(out.value, np.array([10.0, 14.0]))


def test_algebraist_binds_explicit_tableau_stages_as_state_calls():
    algebraist = Algebraist(fields=(AlgebraistField("value", "value"),))
    fake = FakeButcherTableau(
        c=(0.0, 1.0),
        a=((), (0.5,)),
        b=(1.0, 0.0),
        order=1,
    )
    calls = algebraist.bind_explicit_scheme(fake)
    result = FakeState(np.zeros(2))
    origin = FakeState(np.array([10.0, 20.0]))
    k0 = FakeTranslation(np.array([2.0, 4.0]))

    calls.stage_state_calls[1](result, origin, 3.0, k0)

    np.testing.assert_allclose(result.value, np.array([13.0, 26.0]))

    calls.solution_state_call(result, origin, 2.0, k0)
    np.testing.assert_allclose(result.value, np.array([14.0, 28.0]))
