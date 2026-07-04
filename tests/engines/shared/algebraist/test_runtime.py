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
from tests.support import (
    DummyRuntimeAllocator,
    DummyRuntimeState,
    DummyRuntimeTranslation,
    DummyRuntimeTranslationWithLinearCombine,
    dummy_runtime_combine2,
    dummy_runtime_scale,
)


@dataclass
class NormTranslation:
    """Translation-shaped object for layout norm tests with excluded fields."""

    value: float = 0.0
    ignored: float = 0.0


def scalar_field_norm(value: object) -> float:
    """Return an absolute scalar norm with the callable shape runtime expects."""

    if not isinstance(value, int | float):
        raise TypeError("scalar field norm expects a numeric value.")
    return abs(float(value))


def test_arity_validates_value() -> None:
    with pytest.raises(ValueError):
        AlgebraistArity(0)
    with pytest.raises(TypeError):
        AlgebraistArity(True)


def test_runtime_general_uses_return_fallback_without_linear_combine() -> None:
    general = AlgebraistRuntimeLinearCombine(
        translation=DummyRuntimeTranslation(),
        allocator=DummyRuntimeAllocator(),
    )
    combine = general.provide(AlgebraistArity(3))
    out = DummyRuntimeTranslation()

    result = combine(
        2.0,
        DummyRuntimeTranslation(1.0),
        3.0,
        DummyRuntimeTranslation(2.0),
        4.0,
        DummyRuntimeTranslation(3.0),
        out,
    )

    assert result.value == pytest.approx(20.0)
    assert out.value == pytest.approx(0.0)


def test_runtime_general_synthesizes_higher_arity_from_direct_combine2() -> None:
    general = AlgebraistRuntimeLinearCombine(
        translation=DummyRuntimeTranslationWithLinearCombine(),
        allocator=DummyRuntimeAllocator(),
    )
    combine = general.provide(AlgebraistArity(4))
    out = DummyRuntimeTranslation()

    result = combine(
        1.0,
        DummyRuntimeTranslation(1.0),
        2.0,
        DummyRuntimeTranslation(2.0),
        3.0,
        DummyRuntimeTranslation(3.0),
        4.0,
        DummyRuntimeTranslation(4.0),
        out,
    )

    assert result is out
    assert out.value == pytest.approx(30.0)


def test_runtime_general_accepts_explicit_linear_combine_override() -> None:
    general = AlgebraistRuntimeLinearCombine(
        translation=DummyRuntimeTranslation(),
        allocator=DummyRuntimeAllocator(),
        linear_combine=(dummy_runtime_scale, dummy_runtime_combine2),
    )
    combine = general.provide(AlgebraistArity(2))
    out = DummyRuntimeTranslation()

    result = combine(2.0, DummyRuntimeTranslation(3.0), 4.0, DummyRuntimeTranslation(5.0), out)

    assert result is out
    assert out.value == pytest.approx(26.0)


def test_runtime_general_as_tuple_provides_requested_arity_family() -> None:
    general = AlgebraistRuntimeLinearCombine(
        translation=DummyRuntimeTranslationWithLinearCombine(),
        allocator=DummyRuntimeAllocator(),
    )
    family = general.as_tuple(max_arity=5)

    assert len(family) == 5
    assert family[0] is general.provide(AlgebraistArity(1))
    assert family[4] is general.provide(AlgebraistArity(5))


def test_runtime_specialist_binds_delta_coefficients() -> None:
    specialist = AlgebraistRuntimeSpecialist(
        translation=DummyRuntimeTranslationWithLinearCombine(),
        allocator=DummyRuntimeAllocator(),
    )
    kernel = specialist.provide_delta(SchemeStencil(scale=0.5, coefficients=(1.0, 2.0)))
    out = DummyRuntimeTranslation()

    result = kernel(4.0, DummyRuntimeTranslation(3.0), DummyRuntimeTranslation(5.0), out)

    assert result is out
    assert out.value == pytest.approx(26.0)


def test_runtime_specialist_applies_delta_to_origin_state() -> None:
    specialist: AlgebraistRuntimeSpecialist[
        DummyRuntimeState,
        DummyRuntimeTranslation,
    ] = AlgebraistRuntimeSpecialist(
        translation=DummyRuntimeTranslationWithLinearCombine(),
        allocator=DummyRuntimeAllocator(),
    )
    kernel = specialist.provide_apply(
        SchemeStencil(scale=1.0, coefficients=(0.25, 0.75), apply=True)
    )

    origin = DummyRuntimeState(10.0)
    result = DummyRuntimeState()
    returned = kernel(
        2.0,
        origin,
        DummyRuntimeTranslation(4.0),
        DummyRuntimeTranslation(8.0),
        result,
    )

    assert returned is result
    assert result.value == pytest.approx(24.0)


def test_runtime_specialist_binds_empty_stencil_as_zero_delta() -> None:
    specialist = AlgebraistRuntimeSpecialist(
        translation=DummyRuntimeTranslationWithLinearCombine(),
        allocator=DummyRuntimeAllocator(),
    )
    kernel = specialist.provide_delta(SchemeStencil(coefficients=()))
    out = DummyRuntimeTranslation(5.0)

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
        field_norms=(scalar_field_norm, scalar_field_norm),
    ).provide()

    translation = NormTranslation(value=-3.0, ignored=100.0)

    assert norm(translation) == pytest.approx(3.0)
