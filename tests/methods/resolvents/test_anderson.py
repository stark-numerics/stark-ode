from __future__ import annotations

from stark import Configuration, Interval, Tolerance
from stark.core.block import Block
from stark.methods.resolvents import ResolventAnderson
from stark.methods.schemes.request import SchemeResolventRequest
from tests.support import (
    DummyScalarAllocator,
    DummyScalarState,
    DummyScalarTranslation,
    dummy_zero_rhs,
)


def inner_product(left: DummyScalarTranslation, right: DummyScalarTranslation) -> float:
    return left.value * right.value


def test_anderson_resolvent_solves_rhs_shift_without_reversing_residual_arguments() -> None:
    resolvent = ResolventAnderson(
        DummyScalarAllocator(),
        inner_product,
        configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12), resolvent_maximum_steps=4),
    )
    interval = Interval(present=0.0, step=0.1, stop=1.0)
    state = DummyScalarState(1.0)
    rhs = Block([DummyScalarTranslation(2.0)])
    out = Block([DummyScalarTranslation()])
    problem = SchemeResolventRequest(dummy_zero_rhs, interval, state, rhs, 0.0)

    resolvent(problem, out)

    assert out[0].value == 2.0
