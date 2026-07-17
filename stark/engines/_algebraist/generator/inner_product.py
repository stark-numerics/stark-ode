from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import prod
from typing import Generic, TypeVar, cast

from stark.engines.accelerators.none import AcceleratorNone
from stark.engines._algebraist.generator.compiler import AlgebraistGeneratorCompiler
from stark.engines._algebraist.generator.target import (
    AlgebraistGeneratorTarget,
    AlgebraistGeneratorTargetFunctional,
    AlgebraistGeneratorTargetMutable,
    AlgebraistGeneratorTargetMutableVectorized,
)
from stark.core.contracts.accelerator import Accelerator
from stark.core.contracts.frame import FrameLike
from stark.engines._algebraist.inner_product import included_inner_product_entries

TranslationType = TypeVar("TranslationType")


@dataclass(frozen=True, slots=True)
class AlgebraistGeneratorInnerProduct(Generic[TranslationType]):
    """Generated provider of frame-aware translation inner products."""

    translation: TranslationType
    frame: FrameLike
    accelerator: Accelerator = field(default_factory=AcceleratorNone)
    target: AlgebraistGeneratorTarget = field(default_factory=AlgebraistGeneratorTargetMutable)

    def source_string(self, request: None = None) -> str:
        del request
        source = getattr(self.target, "source_inner_product", None)
        if callable(source):
            return cast(str, source(self.frame))
        inner_product_entries = included_inner_product_entries(self.frame)
        parameters = []
        for field, _inner_product in inner_product_entries:
            parameters.append(f"left_{field.translation_name}")
            parameters.append(f"right_{field.translation_name}")
        lines = [f"def _kernel_flat({', '.join(parameters)}):", "    total = 0.0"]

        for field, inner_product in inner_product_entries:
            left_name = f"left_{field.translation_name}"
            right_name = f"right_{field.translation_name}"
            policy = field.policy
            policy_kind = getattr(policy, "kind", None)
            if policy_kind == "scalar":
                if getattr(inner_product, "kind", None) not in {"l2", "rms"}:
                    raise ValueError(
                        "Generated inner product requires L2 or RMS inner product fields."
                    )
                lines.append(f"    total += {left_name} * {right_name}")
                continue
            if policy_kind == "looped":
                shape = getattr(policy, "shape", None)
                if shape is None:
                    shape = getattr(field, "shape", None)
                if shape is None:
                    raise ValueError("Generated inner product requires looped fields to declare shape.")
                lines.extend(
                    self._looped_field_lines(
                        left_name=left_name,
                        right_name=right_name,
                        shape=tuple(shape),
                        inner_product=inner_product,
                        vectorized=isinstance(
                            self.target,
                            (
                                AlgebraistGeneratorTargetFunctional,
                                AlgebraistGeneratorTargetMutableVectorized,
                            ),
                        ),
                    )
                )
                continue
            if policy_kind == "unravel":
                shape = getattr(policy, "shape", None)
                if shape is None:
                    shape = getattr(field, "shape", None)
                if shape is None:
                    raise ValueError("Generated inner product requires unravelled fields to declare shape.")
                lines.extend(
                    self._unravelled_field_lines(
                        left_name=left_name,
                        right_name=right_name,
                        shape=tuple(shape),
                        inner_product=inner_product,
                    )
                )
                continue
            raise ValueError(
                "Generated inner product requires scalar, looped, or unravelled fields."
            )

        lines.append("    return total")
        lines.append("")
        lines.append("def kernel(left, right):")
        arguments = []
        for field, _inner_product in inner_product_entries:
            arguments.append(field.translation_expression("left"))
            arguments.append(field.translation_expression("right"))
        lines.append(f"    return _kernel_flat({', '.join(arguments)})")
        return "\n".join(lines) + "\n"

    def compile(self, source: str) -> Callable[[TranslationType, TranslationType], float]:
        return cast(
            Callable[[TranslationType, TranslationType], float],
            AlgebraistGeneratorCompiler(self.accelerator).compile(source),
        )

    def provide(self, request: None = None) -> Callable[[TranslationType, TranslationType], float]:
        return self.compile(self.source_string(request))

    @staticmethod
    def _looped_field_lines(
        *,
        left_name: str,
        right_name: str,
        shape: tuple[int, ...],
        inner_product: object,
        vectorized: bool = False,
    ) -> list[str]:
        inner_product_kind = getattr(inner_product, "kind", None)
        if inner_product_kind == "l2":
            scale = 1.0
        elif inner_product_kind == "rms":
            scale = float(prod(shape))
        else:
            raise ValueError(
                "Generated inner product requires L2 or RMS inner product fields."
            )

        if vectorized:
            return [f"    total += ({left_name} * {right_name}).sum() / {scale!r}"]

        index_names = tuple(f"i{index}" for index in range(len(shape)))
        lines = []
        for depth, (index_name, bound) in enumerate(zip(index_names, shape, strict=True)):
            indent = "    " * (depth + 1)
            lines.append(f"{indent}for {index_name} in range({bound}):")
        index = "".join(f"[{index_name}]" for index_name in index_names)
        assignment_indent = "    " * (len(shape) + 1)
        lines.append(f"{assignment_indent}total += ({left_name}{index} * {right_name}{index}) / {scale!r}")
        return lines

    @staticmethod
    def _unravelled_field_lines(
        *,
        left_name: str,
        right_name: str,
        shape: tuple[int, ...],
        inner_product: object,
    ) -> list[str]:
        from itertools import product

        inner_product_kind = getattr(inner_product, "kind", None)
        if inner_product_kind == "l2":
            scale = 1.0
        elif inner_product_kind == "rms":
            scale = float(prod(shape))
        else:
            raise ValueError(
                "Generated inner product requires L2 or RMS inner product fields."
            )
        lines = []
        for index_tuple in product(*(range(dimension) for dimension in shape)):
            index = "".join(f"[{index}]" for index in index_tuple)
            lines.append(f"    total += ({left_name}{index} * {right_name}{index}) / {scale!r}")
        return lines


__all__ = ["AlgebraistGeneratorInnerProduct"]
