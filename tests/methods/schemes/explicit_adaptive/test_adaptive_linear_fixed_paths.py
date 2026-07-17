from __future__ import annotations

import pytest

from stark import Interval
from stark.methods.schemes.explicit.adaptive.bogacki_shampine import SchemeBogackiShampine
from stark.methods.schemes.explicit.adaptive.cash_karp import SchemeCashKarp
from stark.methods.schemes.explicit.adaptive.dormand_prince import SchemeDormandPrince
from stark.methods.schemes.explicit.adaptive.fehlberg45 import SchemeFehlberg45
from stark.methods.schemes.explicit.adaptive.tsitouras5 import SchemeTsitouras5
from tests.support import (
    DummyScalarAllocator,
    DummyScalarState,
    DummyTableauLinearFixed,
    dummy_exponential_growth_rhs,
)


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeBogackiShampine,
        SchemeCashKarp,
        SchemeDormandPrince,
        SchemeFehlberg45,
        SchemeTsitouras5,
    ],
)
def test_explicit_adaptive_linear_fixed_path_matches_inline_path(scheme_cls) -> None:
    interval_inline = Interval(present=0.0, step=0.1, stop=0.3)
    interval_linear_fixed = Interval(present=0.0, step=0.1, stop=0.3)
    state_inline = DummyScalarState(1.0)
    state_linear_fixed = DummyScalarState(1.0)

    inline = scheme_cls(dummy_exponential_growth_rhs, DummyScalarAllocator())
    specialized = scheme_cls(
        dummy_exponential_growth_rhs,
        DummyScalarAllocator(),
        linear_fixed=DummyTableauLinearFixed(),
    )

    accepted_inline = inline(interval_inline, state_inline)
    accepted_linear_fixed = specialized(
        interval_linear_fixed,
        state_linear_fixed,
    )

    assert accepted_linear_fixed == pytest.approx(accepted_inline)
    assert state_linear_fixed.value == pytest.approx(state_inline.value)
    assert interval_linear_fixed.step == pytest.approx(interval_inline.step)
