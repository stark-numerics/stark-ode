from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import prod
from typing import Generic, TypeVar, cast

from stark.engines.accelerators.none import AcceleratorNone
from stark.engines.algebraist.generator.compiler import AlgebraistGeneratorCompiler
from stark.engines.algebraist.frame import (
    AlgebraistFrame,
    AlgebraistFrameLooped,
    AlgebraistFrameNormMax,
    AlgebraistFrameNormRMS,
    AlgebraistFrameScalar,
    AlgebraistFrameUnravel,
)
from stark.core.contracts.accelerator import Accelerator

TranslationType = TypeVar("TranslationType")


@dataclass(frozen=True, slots=True)
class AlgebraistGeneratorInnerProduct(Generic[TranslationType]):
    """Generated provider of frame-aware translation inner products."""

    translation: TranslationType
    frame: AlgebraistFrame
    accelerator: Accelerator = field(default_factory=AcceleratorNone)

    def source_string(self, request: None = None) -> str:
        del request
        parameters = []
        for field in self.frame.norm_fields:
            parameters.append(f"left_{field.translation_name}")
            parameters.append(f"right_{field.translation_name}")
        lines = [f"def _kernel_flat({', '.join(parameters)}):", "    total = 0.0"]

        for field in self.frame.norm_fields:
            left_name = f"left_{field.translation_name}"
            right_name = f"right_{field.translation_name}"
            policy = field.policy
            norm = field.norm
            if isinstance(policy, AlgebraistFrameScalar):
                if not isinstance(norm, (AlgebraistFrameNormRMS, AlgebraistFrameNormMax)):
                    raise ValueError("Generated inner product requires RMS or max norm fields.")
                lines.append(f"    total += {left_name} * {right_name}")
                continue
            if isinstance(policy, AlgebraistFrameLooped):
                if policy.shape is None:
                    raise ValueError("Generated inner product requires looped fields to declare shape.")
                lines.extend(
                    self._looped_field_lines(
                        left_name=left_name,
                        right_name=right_name,
                        shape=policy.shape,
                        norm=norm,
                    )
                )
                continue
            if isinstance(policy, AlgebraistFrameUnravel):
                lines.extend(
                    self._unravelled_field_lines(
                        left_name=left_name,
                        right_name=right_name,
                        shape=policy.shape,
                        norm=norm,
                    )
                )
                continue
            raise ValueError(
                "Generated inner product requires scalar, looped, or unravelled norm fields."
            )

        lines.append("    return total")
        lines.append("")
        lines.append("def kernel(left, right):")
        arguments = []
        for field in self.frame.norm_fields:
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
        norm: object,
    ) -> list[str]:
        index_names = tuple(f"i{index}" for index in range(len(shape)))
        lines = []
        if isinstance(norm, AlgebraistFrameNormRMS):
            scale = float(prod(shape))
        elif isinstance(norm, AlgebraistFrameNormMax):
            scale = 1.0
        else:
            raise ValueError("Generated inner product requires RMS or max norm fields.")
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
        norm: object,
    ) -> list[str]:
        from itertools import product

        if isinstance(norm, AlgebraistFrameNormRMS):
            scale = float(prod(shape))
        elif isinstance(norm, AlgebraistFrameNormMax):
            scale = 1.0
        else:
            raise ValueError("Generated inner product requires RMS or max norm fields.")
        lines = []
        for index_tuple in product(*(range(dimension) for dimension in shape)):
            index = "".join(f"[{index}]" for index in index_tuple)
            lines.append(f"    total += ({left_name}{index} * {right_name}{index}) / {scale!r}")
        return lines


__all__ = ["AlgebraistGeneratorInnerProduct"]
