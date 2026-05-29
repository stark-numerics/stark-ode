from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark.algebraist.arity import AlgebraistArity
from stark.algebraist.generator import (
    AlgebraistGeneratorGeneral,
    AlgebraistGeneratorSpecialist,
)
from stark.algebraist.layout import (
    AlgebraistLayout,
    AlgebraistLayoutField,
    AlgebraistLayoutScalar,
    AlgebraistLayoutUnravel,
)
from stark.schemes.support.stencil import SchemeStencil


@dataclass
class State:
    x: float
    values: list[float]


@dataclass
class Translation:
    dx: float
    values: list[float]


class Allocator:
    def allocate_translation(self) -> Translation:
        return Translation(0.0, [0.0, 0.0])


def layout() -> AlgebraistLayout:
    return AlgebraistLayout(
        fields=(
            AlgebraistLayoutField(
                translation_path="dx",
                state_path="x",
                policy=AlgebraistLayoutScalar(),
            ),
            AlgebraistLayoutField(
                translation_path="values",
                state_path="values",
                policy=AlgebraistLayoutUnravel(shape=(2,)),
            ),
        )
    )


def test_generator_general_combines_scalar_and_unravel_fields():
    provider = AlgebraistGeneratorGeneral(
        translation=Translation(0.0, [0.0, 0.0]),
        allocator=Allocator(),
        layout=layout(),
    )
    kernel = provider.provide(AlgebraistArity(2))

    x0 = Translation(2.0, [1.0, 2.0])
    x1 = Translation(4.0, [3.0, 4.0])
    out = Translation(0.0, [0.0, 0.0])

    returned = kernel(10.0, x0, 100.0, x1, out)

    assert returned is out
    assert out.dx == pytest.approx(420.0)
    assert out.values == pytest.approx([310.0, 420.0])


def test_generator_specialist_bakes_delta_coefficients():
    provider = AlgebraistGeneratorSpecialist(
        translation=Translation(0.0, [0.0, 0.0]),
        allocator=Allocator(),
        layout=layout(),
    )
    stencil = SchemeStencil(scale=2.0, coefficients=(0.5, 0.25))
    source = provider.source_string(stencil)

    assert "_a0 = step * 1.0" in source
    assert "_a1 = step * 0.5" in source

    kernel = provider.provide(stencil)
    x0 = Translation(2.0, [1.0, 2.0])
    x1 = Translation(4.0, [3.0, 4.0])
    out = Translation(0.0, [0.0, 0.0])

    returned = kernel(10.0, x0, x1, out)

    assert returned is out
    assert out.dx == pytest.approx(40.0)
    assert out.values == pytest.approx([25.0, 40.0])


def test_generator_specialist_applies_delta_to_origin():
    provider = AlgebraistGeneratorSpecialist(
        translation=Translation(0.0, [0.0, 0.0]),
        allocator=Allocator(),
        layout=layout(),
    )
    stencil = SchemeStencil(scale=1.0, coefficients=(0.5, 0.25), apply=True)
    kernel = provider.provide(stencil)

    origin = State(100.0, [10.0, 20.0])
    result = State(0.0, [0.0, 0.0])
    x0 = Translation(2.0, [1.0, 2.0])
    x1 = Translation(4.0, [3.0, 4.0])

    returned = kernel(10.0, origin, x0, x1, result)

    assert returned is result
    assert result.x == pytest.approx(120.0)
    assert result.values == pytest.approx([22.5, 40.0])


def test_unravel_rejects_large_shapes():
    with pytest.raises(ValueError, match="use AlgebraistLayoutLooped"):
        AlgebraistLayoutUnravel(shape=(17,))
