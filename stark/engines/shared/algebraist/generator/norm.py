from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import prod
from typing import Generic, TypeVar, cast

from stark.engines.shared.accelerators.none import AcceleratorNone
from stark.engines.shared.algebraist.generator.compiler import AlgebraistGeneratorCompiler
from stark.engines.shared.algebraist.generator.target import (
    AlgebraistGeneratorTarget,
    AlgebraistGeneratorTargetFunctional,
    AlgebraistGeneratorTargetMutable,
    AlgebraistGeneratorTargetMutableVectorized,
)
from stark.engines.shared.algebraist.frame import (
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
class AlgebraistGeneratorNorm(Generic[TranslationType]):
    """Generated provider of frame-aware translation norm kernels."""

    translation: TranslationType
    frame: AlgebraistFrame
    accelerator: Accelerator = field(default_factory=AcceleratorNone)
    target: AlgebraistGeneratorTarget = field(default_factory=AlgebraistGeneratorTargetMutable)

    def source_string(self, request: None = None) -> str:
        del request
        source = getattr(self.target, "source_norm", None)
        if callable(source):
            return cast(str, source(self.frame))
        parameters = [
            field.translation_name
            for field in self.frame.norm_fields
        ]
        lines = [f"def _kernel_flat({', '.join(parameters)}):", "    total = 0.0"]

        for field in self.frame.norm_fields:
            name = field.translation_name
            policy = field.policy
            norm = field.norm
            if isinstance(policy, AlgebraistFrameScalar):
                if not isinstance(norm, (AlgebraistFrameNormRMS, AlgebraistFrameNormMax)):
                    raise ValueError("Generated norm requires RMS or max norm fields.")
                lines.append(f"    total += abs({name}) ** 2")
                continue
            if isinstance(policy, AlgebraistFrameLooped):
                if policy.shape is None:
                    raise ValueError("Generated norm requires looped fields to declare shape.")
                lines.extend(
                    self._looped_field_lines(
                        name=name,
                        shape=tuple(policy.shape),
                        norm=norm,
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
            if isinstance(policy, AlgebraistFrameUnravel):
                lines.extend(
                    self._unravelled_field_lines(
                        name=name,
                        shape=tuple(policy.shape),
                        norm=norm,
                    )
                )
                continue
            raise ValueError(
                "Generated norm requires scalar, looped, or unravelled norm fields."
            )

        lines.append("    return total ** 0.5")
        lines.append("")
        lines.append("def kernel(translation):")
        arguments = [
            field.translation_expression("translation")
            for field in self.frame.norm_fields
        ]
        lines.append(f"    return _kernel_flat({', '.join(arguments)})")
        return "\n".join(lines) + "\n"

    def compile(self, source: str) -> Callable[[TranslationType], float]:
        return cast(
            Callable[[TranslationType], float],
            AlgebraistGeneratorCompiler(self.accelerator).compile(source),
        )

    def provide(self, request: None = None) -> Callable[[TranslationType], float]:
        return self.compile(self.source_string(request))

    @staticmethod
    def _looped_field_lines(
        *,
        name: str,
        shape: tuple[int, ...],
        norm: object,
        vectorized: bool = False,
    ) -> list[str]:
        if vectorized:
            if isinstance(norm, AlgebraistFrameNormRMS):
                return [f"    total += (abs({name}) ** 2).sum() / {float(prod(shape))!r}"]
            if isinstance(norm, AlgebraistFrameNormMax):
                return [
                    f"    field_norm = abs({name}).max()",
                    "    total += field_norm ** 2",
                ]
            raise ValueError("Generated norm requires RMS or max norm fields.")

        index_names = tuple(f"i{index}" for index in range(len(shape)))
        if isinstance(norm, AlgebraistFrameNormRMS):
            lines = ["    subtotal = 0.0"]
        elif isinstance(norm, AlgebraistFrameNormMax):
            lines = ["    field_norm = 0.0"]
        else:
            raise ValueError("Generated norm requires RMS or max norm fields.")
        for depth, (index_name, bound) in enumerate(zip(index_names, shape, strict=True)):
            indent = "    " * (depth + 1)
            lines.append(f"{indent}for {index_name} in range({bound}):")
        index = "".join(f"[{index_name}]" for index_name in index_names)
        assignment_indent = "    " * (len(shape) + 1)
        if isinstance(norm, AlgebraistFrameNormRMS):
            lines.append(f"{assignment_indent}subtotal += abs({name}{index}) ** 2")
            lines.append(f"    total += subtotal / {float(prod(shape))!r}")
        else:
            lines.append(f"{assignment_indent}item_norm = abs({name}{index})")
            lines.append(f"{assignment_indent}if item_norm > field_norm:")
            lines.append(f"{assignment_indent}    field_norm = item_norm")
            lines.append("    total += field_norm ** 2")
        return lines

    @staticmethod
    def _unravelled_field_lines(
        *,
        name: str,
        shape: tuple[int, ...],
        norm: object,
    ) -> list[str]:
        from itertools import product

        if isinstance(norm, AlgebraistFrameNormRMS):
            lines = ["    subtotal = 0.0"]
        elif isinstance(norm, AlgebraistFrameNormMax):
            lines = ["    field_norm = 0.0"]
        else:
            raise ValueError("Generated norm requires RMS or max norm fields.")
        for index_tuple in product(*(range(dimension) for dimension in shape)):
            index = "".join(f"[{index}]" for index in index_tuple)
            if isinstance(norm, AlgebraistFrameNormRMS):
                lines.append(f"    subtotal += abs({name}{index}) ** 2")
            else:
                lines.append(f"    item_norm = abs({name}{index})")
                lines.append("    if item_norm > field_norm:")
                lines.append("        field_norm = item_norm")
        if isinstance(norm, AlgebraistFrameNormRMS):
            lines.append(f"    total += subtotal / {float(prod(shape))!r}")
        else:
            lines.append("    total += field_norm ** 2")
        return lines


__all__ = ["AlgebraistGeneratorNorm"]
