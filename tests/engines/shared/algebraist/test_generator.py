from __future__ import annotations

import pytest

from stark.engines.shared.algebraist.arity import AlgebraistArity
from stark.engines.shared.algebraist.generator import (
    AlgebraistGeneratorLinearCombine,
    AlgebraistGeneratorNorm,
    AlgebraistGeneratorSpecialist,
)
from stark.engines.shared.algebraist.frame import (
    AlgebraistFrame,
    AlgebraistField,
    AlgebraistFrameLooped,
    AlgebraistNormExcluded,
    AlgebraistNormMax,
    AlgebraistNormRMS,
    AlgebraistFrameScalar,
    AlgebraistFrameUnravel,
)
from stark.methods.schemes.specialization.stencil import SchemeStencil
from tests.support import (
    DummyStructuredAllocator,
    DummyStructuredState,
    DummyStructuredTranslation,
)


def frame() -> AlgebraistFrame:
    return AlgebraistFrame(
        fields=(
            AlgebraistField(
                translation_path="dx",
                state_path="x",
                policy=AlgebraistFrameScalar(),
            ),
            AlgebraistField(
                translation_path="values",
                state_path="values",
                policy=AlgebraistFrameUnravel(shape=(2,)),
            ),
        )
    )


def test_generator_general_combines_scalar_and_unravel_fields():
    provider = AlgebraistGeneratorLinearCombine(
        translation=DummyStructuredTranslation(0.0, [0.0, 0.0]),
        allocator=DummyStructuredAllocator(),
        frame=frame(),
    )
    kernel = provider.provide(AlgebraistArity(2))

    x0 = DummyStructuredTranslation(2.0, [1.0, 2.0])
    x1 = DummyStructuredTranslation(4.0, [3.0, 4.0])
    out = DummyStructuredTranslation(0.0, [0.0, 0.0])

    returned = kernel(10.0, x0, 100.0, x1, out)

    assert returned is out
    assert out.dx == pytest.approx(420.0)
    assert out.values == pytest.approx([310.0, 420.0])


def test_generator_specialist_bakes_delta_coefficients():
    provider = AlgebraistGeneratorSpecialist(
        translation=DummyStructuredTranslation(0.0, [0.0, 0.0]),
        allocator=DummyStructuredAllocator(),
        frame=frame(),
    )
    stencil = SchemeStencil(scale=2.0, coefficients=(0.5, 0.25))
    source = provider.source_string(stencil)

    assert "_a0 = step * 1.0" in source
    assert "_a1 = step * 0.5" in source

    kernel = provider.provide_delta(stencil)
    x0 = DummyStructuredTranslation(2.0, [1.0, 2.0])
    x1 = DummyStructuredTranslation(4.0, [3.0, 4.0])
    out = DummyStructuredTranslation(0.0, [0.0, 0.0])

    returned = kernel(10.0, x0, x1, out)

    assert returned is out
    assert out.dx == pytest.approx(40.0)
    assert out.values == pytest.approx([25.0, 40.0])


def test_generator_specialist_applies_delta_to_origin():
    provider = AlgebraistGeneratorSpecialist(
        translation=DummyStructuredTranslation(0.0, [0.0, 0.0]),
        allocator=DummyStructuredAllocator(),
        frame=frame(),
    )
    stencil = SchemeStencil(scale=1.0, coefficients=(0.5, 0.25), apply=True)
    kernel = provider.provide_apply(stencil)

    origin = DummyStructuredState(100.0, [10.0, 20.0])
    result = DummyStructuredState(0.0, [0.0, 0.0])
    x0 = DummyStructuredTranslation(2.0, [1.0, 2.0])
    x1 = DummyStructuredTranslation(4.0, [3.0, 4.0])

    returned = kernel(10.0, origin, x0, x1, result)

    assert returned is result
    assert result.x == pytest.approx(120.0)
    assert result.values == pytest.approx([22.5, 40.0])


def test_generator_looped_fields_match_vector_algebra():
    provider = AlgebraistGeneratorLinearCombine(
        translation=DummyStructuredTranslation(0.0, [0.0, 0.0, 0.0]),
        allocator=DummyStructuredAllocator(size=3),
        frame=AlgebraistFrame(
            fields=(
                AlgebraistField(
                    translation_path="values",
                    state_path="values",
                    policy=AlgebraistFrameLooped(shape=(3,)),
                ),
            )
        ),
    )
    kernel = provider.provide(AlgebraistArity(2))

    x0 = DummyStructuredTranslation(0.0, [1.0, 2.0, 3.0])
    x1 = DummyStructuredTranslation(0.0, [4.0, 5.0, 6.0])
    out = DummyStructuredTranslation(0.0, [0.0, 0.0, 0.0])

    returned = kernel(2.0, x0, 3.0, x1, out)

    assert returned is out
    assert out.values == pytest.approx([14.0, 19.0, 24.0])


def test_generator_specialist_looped_update_matches_tableau_algebra():
    provider = AlgebraistGeneratorSpecialist(
        translation=DummyStructuredTranslation(0.0, [0.0, 0.0, 0.0]),
        allocator=DummyStructuredAllocator(size=3),
        frame=AlgebraistFrame(
            fields=(
                AlgebraistField(
                    translation_path="values",
                    state_path="values",
                    policy=AlgebraistFrameLooped(shape=(3,)),
                ),
            )
        ),
    )
    kernel = provider.provide_apply(
        SchemeStencil(coefficients=(0.5, 0.25), apply=True)
    )

    origin = DummyStructuredState(0.0, [10.0, 20.0, 30.0])
    x0 = DummyStructuredTranslation(0.0, [1.0, 2.0, 3.0])
    x1 = DummyStructuredTranslation(0.0, [4.0, 5.0, 6.0])
    result = DummyStructuredState(0.0, [0.0, 0.0, 0.0])

    returned = kernel(2.0, origin, x0, x1, result)

    assert returned is result
    assert result.values == pytest.approx([13.0, 24.5, 36.0])


def test_generator_unit_apply_omits_runtime_step_and_unit_multiply():
    provider = AlgebraistGeneratorSpecialist(
        translation=DummyStructuredTranslation(0.0, [0.0, 0.0, 0.0]),
        allocator=DummyStructuredAllocator(size=3),
        frame=AlgebraistFrame(
            fields=(
                AlgebraistField(
                    translation_path="values",
                    state_path="values",
                    policy=AlgebraistFrameLooped(shape=(3,)),
                ),
            )
        ),
    )
    source = provider.source_unit_apply()

    assert "def kernel(origin, x0, result):" in source
    assert "step" not in source
    assert "1.0 * x0_values" not in source
    assert "result_values[i0] = origin_values[i0] + x0_values[i0]" in source


def test_generator_unit_apply_matches_translation_call_algebra():
    provider = AlgebraistGeneratorSpecialist(
        translation=DummyStructuredTranslation(0.0, [0.0, 0.0, 0.0]),
        allocator=DummyStructuredAllocator(size=3),
        frame=AlgebraistFrame(
            fields=(
                AlgebraistField(
                    translation_path="values",
                    state_path="values",
                    policy=AlgebraistFrameLooped(shape=(3,)),
                ),
            )
        ),
    )
    kernel = provider.provide_unit_apply()

    origin = DummyStructuredState(0.0, [10.0, 20.0, 30.0])
    delta = DummyStructuredTranslation(0.0, [1.0, 2.0, 3.0])
    result = DummyStructuredState(0.0, [0.0, 0.0, 0.0])

    returned = kernel(origin, delta, result)

    assert returned is result
    assert result.values == pytest.approx([11.0, 22.0, 33.0])


def test_generator_norm_uses_included_looped_fields():
    provider = AlgebraistGeneratorNorm(
        translation=DummyStructuredTranslation(0.0, [0.0, 0.0, 0.0]),
        frame=AlgebraistFrame(
            fields=(
                AlgebraistField(
                    translation_path="values",
                    state_path="values",
                    policy=AlgebraistFrameLooped(shape=(3,)),
                ),
                AlgebraistField(
                    translation_path="dx",
                    state_path="x",
                    policy=AlgebraistFrameScalar(),
                ),
            ),
            norms=(AlgebraistNormRMS(), AlgebraistNormExcluded()),
        ),
    )
    source = provider.source_string()

    assert "def kernel(translation):" in source
    assert "translation.values" in source
    assert "translation.dx" not in source

    kernel = provider.provide()

    assert kernel(DummyStructuredTranslation(100.0, [2.0, 3.0, 6.0])) == pytest.approx(
        ((4.0 + 9.0 + 36.0) / 3.0) ** 0.5
    )


def test_generator_norm_uses_field_max_policy():
    provider = AlgebraistGeneratorNorm(
        translation=DummyStructuredTranslation(0.0, [0.0, 0.0, 0.0]),
        frame=AlgebraistFrame(
            fields=(
                AlgebraistField(
                    translation_path="values",
                    state_path="values",
                    policy=AlgebraistFrameLooped(shape=(3,)),
                ),
            ),
            norms=(AlgebraistNormMax(),),
        ),
    )
    source = provider.source_string()

    assert "field_norm" in source

    kernel = provider.provide()

    assert kernel(DummyStructuredTranslation(0.0, [2.0, -7.0, 6.0])) == pytest.approx(7.0)


def test_unravel_rejects_large_shapes():
    with pytest.raises(ValueError, match="use AlgebraistFrameLooped"):
        AlgebraistFrameUnravel(shape=(17,))
