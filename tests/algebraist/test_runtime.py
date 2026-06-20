from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark.engines.shared.algebraist.arity import AlgebraistArity
from stark.engines.shared.algebraist.frame import (
    AlgebraistFrame,
    AlgebraistFrameField,
    AlgebraistFrameNormExcluded,
    AlgebraistFrameScalar,
)
from stark.engines.shared.algebraist.runtime import (
    AlgebraistRuntimeLinearCombine,
    AlgebraistRuntimeNorm,
    AlgebraistRuntimeSpecialist,
)
from stark.methods.schemes.specialization.stencil import SchemeStencil


@dataclass
class State:
    value: float = 0.0


@dataclass
class Translation:
    value: float = 0.0

    def __call__(self, origin: State, result: State) -> None:
        result.value = origin.value + self.value

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: Translation) -> Translation:
        return Translation(self.value + other.value)

    def __rmul__(self, scalar: float) -> Translation:
        return Translation(scalar * self.value)


class Allocator:
    def allocate_translation(self) -> Translation:
        return Translation()


def scale(a: float, x: Translation, out: Translation) -> Translation:
    out.value = a * x.value
    return out


def combine2(
    a0: float,
    x0: Translation,
    a1: float,
    x1: Translation,
    out: Translation,
) -> Translation:
    out.value = a0 * x0.value + a1 * x1.value
    return out


def combine3(
    a0: float,
    x0: Translation,
    a1: float,
    x1: Translation,
    a2: float,
    x2: Translation,
    out: Translation,
) -> Translation:
    out.value = a0 * x0.value + a1 * x1.value + a2 * x2.value
    return out


class TranslationWithLinearCombine(Translation):
    def __init__(self, value: float = 0.0) -> None:
        super().__init__(value)
        self.linear_combine = (scale, combine2, combine3)


def test_arity_validates_value() -> None:
    with pytest.raises(ValueError):
        AlgebraistArity(0)
    with pytest.raises(TypeError):
        AlgebraistArity(True)


def test_runtime_general_uses_return_fallback_without_linear_combine() -> None:
    general = AlgebraistRuntimeLinearCombine(
        translation=Translation(),
        allocator=Allocator(),
    )
    combine = general.provide(AlgebraistArity(3))
    out = Translation()

    result = combine(
        2.0,
        Translation(1.0),
        3.0,
        Translation(2.0),
        4.0,
        Translation(3.0),
        out,
    )

    assert result.value == pytest.approx(20.0)
    assert out.value == pytest.approx(0.0)


def test_runtime_general_synthesizes_higher_arity_from_direct_combine2() -> None:
    general = AlgebraistRuntimeLinearCombine(
        translation=TranslationWithLinearCombine(),
        allocator=Allocator(),
    )
    combine = general.provide(AlgebraistArity(4))
    out = Translation()

    result = combine(
        1.0,
        Translation(1.0),
        2.0,
        Translation(2.0),
        3.0,
        Translation(3.0),
        4.0,
        Translation(4.0),
        out,
    )

    assert result is out
    assert out.value == pytest.approx(30.0)


def test_runtime_general_accepts_explicit_linear_combine_override() -> None:
    general = AlgebraistRuntimeLinearCombine(
        translation=Translation(),
        allocator=Allocator(),
        linear_combine=(scale, combine2),
    )
    combine = general.provide(AlgebraistArity(2))
    out = Translation()

    result = combine(2.0, Translation(3.0), 4.0, Translation(5.0), out)

    assert result is out
    assert out.value == pytest.approx(26.0)


def test_runtime_general_as_tuple_provides_requested_arity_family() -> None:
    general = AlgebraistRuntimeLinearCombine(
        translation=TranslationWithLinearCombine(),
        allocator=Allocator(),
    )
    family = general.as_tuple(max_arity=5)

    assert len(family) == 5
    assert family[0] is general.provide(AlgebraistArity(1))
    assert family[4] is general.provide(AlgebraistArity(5))


def test_runtime_specialist_binds_delta_coefficients() -> None:
    specialist = AlgebraistRuntimeSpecialist(
        translation=TranslationWithLinearCombine(),
        allocator=Allocator(),
    )
    kernel = specialist.provide(SchemeStencil(scale=0.5, coefficients=(1.0, 2.0)))
    out = Translation()

    result = kernel(4.0, Translation(3.0), Translation(5.0), out)

    assert result is out
    assert out.value == pytest.approx(26.0)


def test_runtime_specialist_applies_delta_to_origin_state() -> None:
    specialist: AlgebraistRuntimeSpecialist[State, Translation] = AlgebraistRuntimeSpecialist(
        translation=TranslationWithLinearCombine(),
        allocator=Allocator(),
    )
    kernel = specialist.provide(
        SchemeStencil(scale=1.0, coefficients=(0.25, 0.75), apply=True)
    )

    origin = State(10.0)
    result = State()
    returned = kernel(2.0, origin, Translation(4.0), Translation(8.0), result)

    assert returned is result
    assert result.value == pytest.approx(24.0)


def test_runtime_specialist_binds_empty_stencil_as_zero_delta() -> None:
    specialist = AlgebraistRuntimeSpecialist(
        translation=TranslationWithLinearCombine(),
        allocator=Allocator(),
    )
    kernel = specialist.provide(SchemeStencil(coefficients=()))
    out = Translation(5.0)

    result = kernel(2.0, out)

    assert result is out
    assert out.value == pytest.approx(0.0)


def test_runtime_norm_uses_layout_norm_fields() -> None:
    norm = AlgebraistRuntimeNorm(
        frame=AlgebraistFrame(
            (
                AlgebraistFrameField(
                    translation_path="value",
                    state_path="value",
                    policy=AlgebraistFrameScalar(),
                ),
                AlgebraistFrameField(
                    translation_path="ignored",
                    state_path="ignored",
                    policy=AlgebraistFrameScalar(),
                    norm=AlgebraistFrameNormExcluded(),
                ),
            )
        ),
        field_norms=(abs, abs),
    ).provide()

    translation = type("RuntimeNormTranslation", (), {})()
    translation.value = -3.0
    translation.ignored = 100.0

    assert norm(translation) == pytest.approx(3.0)
