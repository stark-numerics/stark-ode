from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import prod
from typing import Any, Generic, cast

from stark.core.contracts.engines.accelerator import Accelerator
from stark.core.contracts.problem.field import FieldLike, FieldPolicyLike
from stark.core.contracts.problem.frame import FrameLike
from stark.core.contracts.problem.inner_product import InnerProductNamed
from stark.core.contracts.problem.translation import TranslationType
from stark.engines.accelerators.none import AcceleratorNone
from stark.engines.generator.compiler import GeneratorCompiler
from stark.engines.generator.policy import GeneratorPolicy, GeneratorPolicyLike
from stark.engines.generator.request import GeneratorRequestInnerProductLike

GENERATED_INNER_PRODUCT_KINDS = frozenset({"l2", "rms"})


def included_inner_product_entries(
    frame: FrameLike,
) -> tuple[tuple[FieldLike[Any, Any], InnerProductNamed[Any]], ...]:
    """Return field/inner-product pairs that participate in inner products."""

    return tuple(
        (field, inner_product)
        for field, inner_product in zip(frame.fields, frame.inner_products, strict=True)
        if inner_product.kind != "excluded"
    )


@dataclass(frozen=True, slots=True)
class GeneratorInnerProduct(Generic[TranslationType]):
    """Generate or bind frame-aware inner-product kernels."""

    frame: FrameLike
    accelerator: Accelerator = field(default_factory=AcceleratorNone)
    policy: GeneratorPolicyLike = field(default_factory=GeneratorPolicy)

    def __call__(self, request: GeneratorRequestInnerProductLike) -> Callable[..., object]:
        return self.generated(request)

    def generated(self, request: GeneratorRequestInnerProductLike) -> Callable[..., object]:
        return self.compile(self.source(request))

    def source(self, request: GeneratorRequestInnerProductLike) -> str:
        del request
        entries = included_inner_product_entries(self.frame)
        parameters: list[str] = []
        for field, _inner_product in entries:
            parameters.append(f"left_{field.translation_name}")
            parameters.append(f"right_{field.translation_name}")
        lines = [f"def _kernel_flat({', '.join(parameters)}):", "    total = 0.0"]

        for field, inner_product in entries:
            left_name = f"left_{field.translation_name}"
            right_name = f"right_{field.translation_name}"
            policy = field.policy
            match policy.kind:
                case "scalar":
                    self.ensure_supported_inner_product(field, inner_product)
                    lines.append(f"    total += {left_name} * {right_name}")
                case "looped":
                    shape = self.field_shape(field, policy)
                    lines.extend(
                        self.looped_field_lines(
                            field_name=field.state_name,
                            left_name=left_name,
                            right_name=right_name,
                            shape=shape,
                            inner_product=inner_product,
                            vectorized=self.uses_vectorized_arrays(),
                        )
                    )
                case "unravel":
                    shape = self.field_shape(field, policy)
                    lines.extend(
                        self.unravelled_field_lines(
                            field_name=field.state_name,
                            left_name=left_name,
                            right_name=right_name,
                            shape=shape,
                            inner_product=inner_product,
                        )
                    )
                case _:
                    raise ValueError(
                        "Generated inner-product source does not yet support "
                        f"field.policy.kind={policy.kind!r} "
                        f"for field {field.state_name!r}. "
                        "Supported generated policies today: 'looped', 'scalar', 'unravel'."
                    )

        lines.append("    return total")
        lines.append("")
        lines.append("def kernel(left, right):")
        arguments: list[str] = []
        for field, _inner_product in entries:
            arguments.append(field.translation_expression("left"))
            arguments.append(field.translation_expression("right"))
        lines.append(f"    return _kernel_flat({', '.join(arguments)})")
        return "\n".join(lines) + "\n"

    def compile(self, source: str) -> Callable[[TranslationType, TranslationType], float]:
        return cast(
            Callable[[TranslationType, TranslationType], float],
            GeneratorCompiler(self.accelerator).compile(source),
        )

    def uses_vectorized_arrays(self) -> bool:
        return self.policy.traversal in {"vectorized", "elementwise", "backend_kernel"} or (
            self.policy.expression in {"array_expression", "elementwise", "backend_kernel"}
        )

    @staticmethod
    def field_shape(
        field: FieldLike[Any, Any],
        policy: FieldPolicyLike,
    ) -> tuple[int, ...]:
        shape = field.shape
        if shape is None:
            raise ValueError(
                "Generated inner-product source needs a concrete shape for "
                f"field {field.state_name!r} with policy "
                f"{policy.kind!r}."
            )
        return tuple(shape)

    @staticmethod
    def ensure_supported_inner_product(
        field: FieldLike[Any, Any],
        inner_product: InnerProductNamed[Any],
    ) -> None:
        inner_product_kind = inner_product.kind
        if inner_product_kind in GENERATED_INNER_PRODUCT_KINDS:
            return

        supported = ", ".join(repr(kind) for kind in sorted(GENERATED_INNER_PRODUCT_KINDS))
        raise ValueError(
            "Generated inner-product source does not yet support "
            f"inner_product.kind={inner_product_kind!r} "
            f"for field {field.state_name!r} "
            f"({type(inner_product).__name__}). "
            f"Supported generated kinds today: {supported}."
        )

    @staticmethod
    def looped_field_lines(
        *,
        field_name: str,
        left_name: str,
        right_name: str,
        shape: tuple[int, ...],
        inner_product: InnerProductNamed[Any],
        vectorized: bool,
    ) -> list[str]:
        inner_product_kind = inner_product.kind
        if inner_product_kind == "l2":
            scale = 1.0
        elif inner_product_kind == "rms":
            scale = float(prod(shape))
        else:
            supported = ", ".join(repr(kind) for kind in sorted(GENERATED_INNER_PRODUCT_KINDS))
            raise ValueError(
                "Generated inner-product source does not yet support "
                f"inner_product.kind={inner_product_kind!r} "
                f"for field {field_name!r} "
                f"({type(inner_product).__name__}). "
                f"Supported generated kinds today: {supported}."
            )

        if vectorized:
            return [f"    total += ({left_name} * {right_name}).sum() / {scale!r}"]

        index_names = tuple(f"i{index}" for index in range(len(shape)))
        lines: list[str] = []
        for depth, (index_name, bound) in enumerate(zip(index_names, shape, strict=True)):
            indent = "    " * (depth + 1)
            lines.append(f"{indent}for {index_name} in range({bound}):")
        index = "".join(f"[{index_name}]" for index_name in index_names)
        assignment_indent = "    " * (len(shape) + 1)
        lines.append(
            f"{assignment_indent}total += ({left_name}{index} * {right_name}{index}) / {scale!r}"
        )
        return lines

    @staticmethod
    def unravelled_field_lines(
        *,
        field_name: str,
        left_name: str,
        right_name: str,
        shape: tuple[int, ...],
        inner_product: InnerProductNamed[Any],
    ) -> list[str]:
        from itertools import product

        inner_product_kind = inner_product.kind
        if inner_product_kind == "l2":
            scale = 1.0
        elif inner_product_kind == "rms":
            scale = float(prod(shape))
        else:
            supported = ", ".join(repr(kind) for kind in sorted(GENERATED_INNER_PRODUCT_KINDS))
            raise ValueError(
                "Generated inner-product source does not yet support "
                f"inner_product.kind={inner_product_kind!r} "
                f"for field {field_name!r} "
                f"({type(inner_product).__name__}). "
                f"Supported generated kinds today: {supported}."
            )
        lines: list[str] = []
        for index_tuple in product(*(range(dimension) for dimension in shape)):
            index = "".join(f"[{index}]" for index in index_tuple)
            lines.append(f"    total += ({left_name}{index} * {right_name}{index}) / {scale!r}")
        return lines


__all__ = [
    "GENERATED_INNER_PRODUCT_KINDS",
    "GeneratorInnerProduct",
    "included_inner_product_entries",
]
