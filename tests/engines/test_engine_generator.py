from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import ClassVar

import pytest

from stark.engines.generator import (
    Generator,
    GeneratorElementwiseSource,
    GeneratorPolicy,
    GeneratorRequestApplyTranslation,
    GeneratorRequestInnerProduct,
    GeneratorRequestLinearCombine,
    GeneratorRequestLinearCombineTable,
    GeneratorRequestNorm,
)
from stark.engines.generator.linear_fixed_source import GeneratorLinearFixedSource
from stark.methods.schemes.specialization import SchemeStencil
from stark.problem.frame import Field, FieldPolicy, Frame
from tests.support import DummyScalarAllocator, DummyScalarState, DummyScalarTranslation


@dataclass(frozen=True, slots=True)
class UnknownGeneratorRequest:
    operation: str
    kind: str = "default"


@dataclass(slots=True)
class TwoFieldState:
    a: list[float]
    b: list[float]


@dataclass(slots=True)
class TwoFieldTranslation:
    da: list[float]
    db: list[float]

    def __call__(self, origin: TwoFieldState, result: TwoFieldState) -> None:
        result.a[:] = [
            origin_value + translation_value
            for origin_value, translation_value in zip(origin.a, self.da, strict=True)
        ]
        result.b[:] = [
            origin_value + translation_value
            for origin_value, translation_value in zip(origin.b, self.db, strict=True)
        ]

    def norm(self) -> float:
        return sqrt(
            sum(value * value for value in self.da)
            + sum(value * value for value in self.db)
        )

    def __add__(self, other: TwoFieldTranslation) -> TwoFieldTranslation:
        return TwoFieldTranslation(
            [
                left + right
                for left, right in zip(self.da, other.da, strict=True)
            ],
            [
                left + right
                for left, right in zip(self.db, other.db, strict=True)
            ],
        )

    def __rmul__(self, scalar: float) -> TwoFieldTranslation:
        return TwoFieldTranslation(
            [scalar * value for value in self.da],
            [scalar * value for value in self.db],
        )


@dataclass(frozen=True, slots=True)
class CustomInnerProduct:
    kind: ClassVar[str] = "custom"

    def __call__(self, left: object, right: object) -> float:
        del left, right
        return 0.0


@dataclass(frozen=True, slots=True)
class CustomNorm:
    kind: ClassVar[str] = "custom"

    def __call__(self, translation_field: object) -> float:
        del translation_field
        return 0.0


@dataclass(frozen=True, slots=True)
class CustomFieldPolicy:
    kind: str = "chunked"


def test_scheme_stencil_is_linear_fixed_generator_request() -> None:
    stencil = SchemeStencil((1.0, 2.0), scale=0.5, apply=True)

    assert stencil.operation == "linear_fixed"
    assert stencil.coefficients == (1.0, 2.0)
    assert stencil.scale == 0.5
    assert stencil.apply is True


def test_generator_linear_fixed_delta_uses_emitted_source() -> None:
    generator = Generator[DummyScalarState, DummyScalarTranslation](
        Frame(Field("value", policy=FieldPolicy.scalar())),
        allocator=DummyScalarAllocator(),
    )

    kernel = generator(SchemeStencil((2.0, -1.0), scale=0.5))
    out = DummyScalarTranslation()

    kernel(
        0.25,
        DummyScalarTranslation(8.0),
        DummyScalarTranslation(2.0),
        out,
    )

    assert out.value == pytest.approx(1.75)


def test_generator_linear_fixed_apply_uses_emitted_source() -> None:
    generator = Generator[DummyScalarState, DummyScalarTranslation](
        Frame(Field("value", policy=FieldPolicy.scalar())),
        allocator=DummyScalarAllocator(),
    )

    kernel = generator(SchemeStencil((1.0,), apply=True))
    result = DummyScalarState()

    kernel(
        0.25,
        DummyScalarState(10.0),
        DummyScalarTranslation(8.0),
        result,
    )

    assert result.value == pytest.approx(12.0)


def test_generator_apply_translation_uses_emitted_source() -> None:
    generator = Generator[DummyScalarState, DummyScalarTranslation](
        Frame(Field("value", policy=FieldPolicy.scalar())),
        allocator=DummyScalarAllocator(),
    )

    kernel = generator(GeneratorRequestApplyTranslation())
    result = DummyScalarState()

    kernel(
        DummyScalarState(10.0),
        DummyScalarTranslation(8.0),
        result,
    )

    assert result.value == pytest.approx(18.0)


def test_generator_linear_combine_uses_emitted_source() -> None:
    generator = Generator[DummyScalarState, DummyScalarTranslation](
        Frame(Field("value", policy=FieldPolicy.scalar())),
        allocator=DummyScalarAllocator(),
    )

    kernel = generator(GeneratorRequestLinearCombine(arity=2))
    out = DummyScalarTranslation()

    kernel(
        0.25,
        DummyScalarTranslation(8.0),
        -0.125,
        DummyScalarTranslation(4.0),
        out,
    )

    assert out.value == pytest.approx(1.5)


def test_generator_linear_combine_table_expands_arities() -> None:
    generator = Generator[DummyScalarState, DummyScalarTranslation](
        Frame(Field("value", policy=FieldPolicy.scalar())),
        allocator=DummyScalarAllocator(),
    )

    table = generator(GeneratorRequestLinearCombineTable(max_arity=4))
    out = DummyScalarTranslation()

    assert len(table) == 4

    table[3](
        0.5,
        DummyScalarTranslation(2.0),
        0.25,
        DummyScalarTranslation(4.0),
        -1.0,
        DummyScalarTranslation(1.0),
        2.0,
        DummyScalarTranslation(3.0),
        out,
    )

    assert out.value == pytest.approx(7.0)


def test_generator_norm_uses_emitted_source() -> None:
    generator = Generator[DummyScalarState, DummyScalarTranslation](
        Frame(Field("value", policy=FieldPolicy.scalar()))
    )

    kernel = generator(GeneratorRequestNorm())

    assert kernel(DummyScalarTranslation(3.0)) == pytest.approx(3.0)


def test_generator_inner_product_uses_emitted_source() -> None:
    generator = Generator[DummyScalarState, DummyScalarTranslation](
        Frame(Field("value", policy=FieldPolicy.scalar()))
    )

    kernel = generator(GeneratorRequestInnerProduct())

    assert kernel(
        DummyScalarTranslation(2.0),
        DummyScalarTranslation(4.0),
    ) == pytest.approx(8.0)


def test_generator_generated_linear_fixed_delta_uses_emitted_source() -> None:
    generator = Generator[DummyScalarState, DummyScalarTranslation](
        Frame(Field("value", policy=FieldPolicy.scalar())),
        policy=GeneratorPolicy(),
    )

    kernel = generator(SchemeStencil((2.0, -1.0), scale=0.5))
    out = DummyScalarTranslation()

    kernel(
        0.25,
        DummyScalarTranslation(8.0),
        DummyScalarTranslation(2.0),
        out,
    )

    assert out.value == pytest.approx(1.75)


def test_generator_generated_linear_fixed_apply_uses_emitted_source() -> None:
    generator = Generator[DummyScalarState, DummyScalarTranslation](
        Frame(Field("value", policy=FieldPolicy.scalar())),
        policy=GeneratorPolicy(),
    )

    kernel = generator(SchemeStencil((1.0,), apply=True))
    result = DummyScalarState()

    kernel(
        0.25,
        DummyScalarState(10.0),
        DummyScalarTranslation(8.0),
        result,
    )

    assert result.value == pytest.approx(12.0)


def test_generator_generated_linear_fixed_groups_same_shape_looped_fields() -> None:
    frame = Frame(
        (
            Field("a", translation="da", shape=(2,), policy=FieldPolicy.looped()),
            Field("b", translation="db", shape=(2,), policy=FieldPolicy.looped()),
        )
    )
    policy = GeneratorPolicy()
    stencil = SchemeStencil((1.0,))

    source = GeneratorLinearFixedSource(frame, policy=policy)(stencil)

    assert source.count("for i0 in range(2):") == 1

    generator = Generator[TwoFieldState, TwoFieldTranslation](
        frame,
        policy=policy,
    )
    kernel = generator(stencil)
    out = TwoFieldTranslation([0.0, 0.0], [0.0, 0.0])

    kernel(
        0.5,
        TwoFieldTranslation([2.0, 4.0], [6.0, 8.0]),
        out,
    )

    assert out.da == pytest.approx([1.0, 2.0])
    assert out.db == pytest.approx([3.0, 4.0])


def test_generator_generated_linear_combine_groups_same_shape_looped_fields() -> None:
    frame = Frame(
        (
            Field("a", translation="da", shape=(2,), policy=FieldPolicy.looped()),
            Field("b", translation="db", shape=(2,), policy=FieldPolicy.looped()),
        )
    )
    policy = GeneratorPolicy()

    source = GeneratorLinearFixedSource(frame, policy=policy).emit(
        kind="general",
        coefficients=None,
        arity=2,
    )

    assert source.count("for i0 in range(2):") == 1

    generator = Generator[TwoFieldState, TwoFieldTranslation](
        frame,
        policy=policy,
    )
    kernel = generator(GeneratorRequestLinearCombine(arity=2))
    out = TwoFieldTranslation([0.0, 0.0], [0.0, 0.0])

    kernel(
        0.5,
        TwoFieldTranslation([2.0, 4.0], [6.0, 8.0]),
        0.25,
        TwoFieldTranslation([4.0, 8.0], [8.0, 12.0]),
        out,
    )

    assert out.da == pytest.approx([2.0, 4.0])
    assert out.db == pytest.approx([5.0, 7.0])


def test_generator_elementwise_linear_combine_emits_backend_kernel_source() -> None:
    frame = Frame(
        (
            Field("a", translation="da", shape=(2,), policy=FieldPolicy.looped()),
            Field("b", translation="db", shape=(2,), policy=FieldPolicy.looped()),
        )
    )
    policy = GeneratorPolicy(traversal="elementwise", expression="elementwise")

    source = GeneratorLinearFixedSource(frame, policy=policy).emit(
        kind="general",
        coefficients=None,
        arity=2,
    )

    assert "ElementwiseKernel" in source
    assert "stark_elementwise_general_2_0" in source
    assert "stark_elementwise_general_2_1" in source
    assert "def kernel(a0, x0, a1, x1, out):" in source
    assert "_kernel_0(a0, x0.da, a1, x1.da, out.da)" in source
    assert "_kernel_1(a0, x0.db, a1, x1.db, out.db)" in source


def test_generator_generated_apply_translation_groups_same_shape_looped_fields() -> None:
    frame = Frame(
        (
            Field("a", translation="da", shape=(2,), policy=FieldPolicy.looped()),
            Field("b", translation="db", shape=(2,), policy=FieldPolicy.looped()),
        )
    )
    policy = GeneratorPolicy()

    source = GeneratorLinearFixedSource(frame, policy=policy).unit_apply()

    assert source.count("for i0 in range(2):") == 1

    generator = Generator[TwoFieldState, TwoFieldTranslation](
        frame,
        policy=policy,
    )
    kernel = generator(GeneratorRequestApplyTranslation())
    result = TwoFieldState([0.0, 0.0], [0.0, 0.0])

    kernel(
        TwoFieldState([1.0, 2.0], [3.0, 4.0]),
        TwoFieldTranslation([0.5, 1.0], [1.5, 2.0]),
        result,
    )

    assert result.a == pytest.approx([1.5, 3.0])
    assert result.b == pytest.approx([4.5, 6.0])


def test_generator_elementwise_apply_translation_emits_backend_kernel_source() -> None:
    frame = Frame(
        (
            Field("a", translation="da", shape=(2,), policy=FieldPolicy.looped()),
            Field("b", translation="db", shape=(2,), policy=FieldPolicy.looped()),
        )
    )
    policy = GeneratorPolicy(traversal="elementwise", expression="elementwise")

    source = GeneratorLinearFixedSource(frame, policy=policy).unit_apply()

    assert "ElementwiseKernel" in source
    assert "stark_elementwise_apply_translation_1_0" in source
    assert "stark_elementwise_apply_translation_1_1" in source
    assert "def kernel(origin, x0, result):" in source
    assert "_kernel_0(origin.a, x0.da, result.a)" in source
    assert "_kernel_1(origin.b, x0.db, result.b)" in source


def test_generator_elementwise_source_reports_unshaped_fields() -> None:
    source = GeneratorElementwiseSource(Frame(Field("value", policy=FieldPolicy.scalar())))

    with pytest.raises(ValueError) as error:
        source.linear_combine(arity=1)

    message = str(error.value)
    assert "Elementwise generated algebra requires shaped looped frame fields" in message
    assert "field 'value'" in message


def test_generator_generated_linear_fixed_reports_unsupported_policy() -> None:
    generator = Generator[DummyScalarState, DummyScalarTranslation](
        Frame(Field("value", policy=CustomFieldPolicy())),
        policy=GeneratorPolicy(),
    )

    with pytest.raises(ValueError) as error:
        generator(SchemeStencil((1.0,)))

    message = str(error.value)
    assert "field.policy.kind='chunked'" in message
    assert "field 'value'" in message


def test_generator_generated_norm_uses_emitted_source() -> None:
    generator = Generator[DummyScalarState, DummyScalarTranslation](
        Frame(Field("value", policy=FieldPolicy.scalar())),
        policy=GeneratorPolicy(),
    )

    kernel = generator(GeneratorRequestNorm())

    assert kernel(DummyScalarTranslation(3.0)) == pytest.approx(3.0)


def test_generator_generated_norm_groups_same_shape_looped_fields() -> None:
    frame = Frame(
        (
            Field("a", translation="da", shape=(2,), policy=FieldPolicy.looped()),
            Field("b", translation="db", shape=(2,), policy=FieldPolicy.looped()),
        )
    )
    generator = Generator[TwoFieldState, TwoFieldTranslation](
        frame,
        policy=GeneratorPolicy(),
    )

    source = generator.norm.source(GeneratorRequestNorm())

    assert source.count("for i0 in range(2):") == 1

    kernel = generator(GeneratorRequestNorm())

    assert kernel(TwoFieldTranslation([3.0, 4.0], [0.0, 6.0])) == pytest.approx(
        sqrt(30.5)
    )


def test_generator_generated_norm_reports_unsupported_kind() -> None:
    generator = Generator[DummyScalarState, DummyScalarTranslation](
        Frame(
            Field("value", policy=FieldPolicy.scalar()),
            norms=CustomNorm(),
        ),
        policy=GeneratorPolicy(),
    )

    with pytest.raises(ValueError) as error:
        generator(GeneratorRequestNorm())

    message = str(error.value)
    assert "norm.kind='custom'" in message
    assert "field 'value'" in message
    assert "CustomNorm" in message


def test_generator_generated_inner_product_uses_emitted_source() -> None:
    generator = Generator[DummyScalarState, DummyScalarTranslation](
        Frame(Field("value", policy=FieldPolicy.scalar())),
        policy=GeneratorPolicy(),
    )

    kernel = generator(GeneratorRequestInnerProduct())

    assert kernel(
        DummyScalarTranslation(2.0),
        DummyScalarTranslation(4.0),
    ) == pytest.approx(8.0)


def test_generator_generated_inner_product_reports_unsupported_kind() -> None:
    generator = Generator[DummyScalarState, DummyScalarTranslation](
        Frame(
            Field("value", policy=FieldPolicy.scalar()),
            inner_products=CustomInnerProduct(),
        ),
        policy=GeneratorPolicy(),
    )

    with pytest.raises(ValueError) as error:
        generator(GeneratorRequestInnerProduct())

    message = str(error.value)
    assert "inner_product.kind='custom'" in message
    assert "field 'value'" in message
    assert "CustomInnerProduct" in message


def test_generator_unknown_operation_raises() -> None:
    generator = Generator[DummyScalarState, DummyScalarTranslation](Frame("value"))

    with pytest.raises(NotImplementedError, match="unknown"):
        generator(UnknownGeneratorRequest("unknown"))
