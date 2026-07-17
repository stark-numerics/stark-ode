from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import prod
from typing import Any, Generic, cast

from stark.core.contracts.accelerator import Accelerator
from stark.core.contracts.field import FieldLike
from stark.core.contracts.frame import FrameLike
from stark.core.contracts.norm import NormLike
from stark.core.contracts.translation import TranslationType
from stark.engines.accelerators.none import AcceleratorNone
from stark.engines.generator.compiler import GeneratorCompiler
from stark.engines.generator.policy import GeneratorPolicy, GeneratorPolicyLike
from stark.engines.generator.request import GeneratorRequestNormLike

GENERATED_NORM_KINDS = frozenset({"max", "rms"})
NormLoopGroupKey = tuple[int, tuple[int, ...]]
NormLoopGroupItem = tuple[FieldLike[Any, Any], NormLike[Any]]


def included_norm_entries(
    frame: FrameLike,
) -> tuple[tuple[FieldLike[Any, Any], NormLike[Any]], ...]:
    """Return field/norm pairs that participate in norm calculations."""

    return tuple(
        (field, norm)
        for field, norm in zip(frame.fields, frame.norms, strict=True)
        if getattr(norm, "kind", None) != "excluded"
    )


@dataclass(frozen=True, slots=True)
class GeneratorNorm(Generic[TranslationType]):
    """Generate or bind frame-aware norm kernels."""

    frame: FrameLike
    accelerator: Accelerator = field(default_factory=AcceleratorNone)
    policy: GeneratorPolicyLike = field(default_factory=GeneratorPolicy)

    def __call__(self, request: GeneratorRequestNormLike) -> Callable[..., object]:
        return self.generated(request)

    def generated(self, request: GeneratorRequestNormLike) -> Callable[..., object]:
        return self.compile(self.source(request))

    def source(self, request: GeneratorRequestNormLike) -> str:
        del request
        entries = included_norm_entries(self.frame)
        parameters = [field.translation_name for field, _norm in entries]
        lines = [f"def _kernel_flat({', '.join(parameters)}):", "    total = 0.0"]
        looped_groups: dict[NormLoopGroupKey, list[NormLoopGroupItem]] = {}

        for field, norm in entries:
            name = field.translation_name
            policy = field.policy
            match getattr(policy, "kind", None):
                case "scalar":
                    self.ensure_supported_norm(field, norm)
                    lines.append(f"    total += abs({name}) ** 2")
                case "looped":
                    shape = self.field_shape(field, policy)
                    if self.uses_vectorized_arrays():
                        lines.extend(
                            self.looped_field_lines(
                                field_name=field.state_name,
                                name=name,
                                shape=shape,
                                norm=norm,
                                vectorized=True,
                            )
                        )
                    else:
                        key = self.loop_group_key(field, policy)
                        looped_groups.setdefault(key, []).append((field, norm))
                case "unravel":
                    shape = self.field_shape(field, policy)
                    lines.extend(
                        self.unravelled_field_lines(
                            field_name=field.state_name,
                            name=name,
                            shape=shape,
                            norm=norm,
                        )
                    )
                case _:
                    policy_kind = getattr(policy, "kind", None)
                    raise ValueError(
                        "Generated norm source does not yet support "
                        f"field.policy.kind={policy_kind!r} "
                        f"for field {field.state_name!r}. "
                        "Supported generated policies today: 'looped', 'scalar', 'unravel'."
                    )

        for key, group in looped_groups.items():
            lines.extend(self.looped_group_lines(key=key, group=tuple(group)))

        lines.append("    return total ** 0.5")
        lines.append("")
        lines.append("def kernel(translation):")
        arguments = [
            field.translation_expression("translation")
            for field, _norm in entries
        ]
        lines.append(f"    return _kernel_flat({', '.join(arguments)})")
        return "\n".join(lines) + "\n"

    def compile(self, source: str) -> Callable[[TranslationType], float]:
        return cast(
            Callable[[TranslationType], float],
            GeneratorCompiler(self.accelerator).compile(source),
        )

    def uses_vectorized_arrays(self) -> bool:
        return self.policy.traversal in {"vectorized", "elementwise", "backend_kernel"} or (
            self.policy.expression in {"array_expression", "elementwise", "backend_kernel"}
        )

    @staticmethod
    def field_shape(field: FieldLike[Any, Any], policy: object) -> tuple[int, ...]:
        shape = getattr(policy, "shape", None)
        if shape is None:
            shape = getattr(field, "shape", None)
        if shape is None:
            raise ValueError(
                "Generated norm source needs a concrete shape for "
                f"field {field.state_name!r} with policy "
                f"{getattr(policy, 'kind', None)!r}."
            )
        return tuple(shape)

    def loop_group_key(
        self,
        field: FieldLike[Any, Any],
        policy: object,
    ) -> NormLoopGroupKey:
        shape = self.field_shape(field, policy)
        rank = getattr(policy, "rank", None)
        if rank is None:
            rank = len(shape)
        return int(rank), shape

    @staticmethod
    def ensure_supported_norm(field: FieldLike[Any, Any], norm: object) -> None:
        norm_kind = getattr(norm, "kind", None)
        if norm_kind in GENERATED_NORM_KINDS:
            return

        supported = ", ".join(repr(kind) for kind in sorted(GENERATED_NORM_KINDS))
        raise ValueError(
            "Generated norm source does not yet support "
            f"norm.kind={norm_kind!r} "
            f"for field {field.state_name!r} "
            f"({type(norm).__name__}). "
            f"Supported generated kinds today: {supported}."
        )

    @staticmethod
    def looped_field_lines(
        *,
        field_name: str,
        name: str,
        shape: tuple[int, ...],
        norm: object,
        vectorized: bool,
    ) -> list[str]:
        norm_kind = getattr(norm, "kind", None)
        if vectorized:
            if norm_kind == "rms":
                return [f"    total += (abs({name}) ** 2).sum() / {float(prod(shape))!r}"]
            if norm_kind == "max":
                return [
                    f"    field_norm = abs({name}).max()",
                    "    total += field_norm ** 2",
                ]
            supported = ", ".join(repr(kind) for kind in sorted(GENERATED_NORM_KINDS))
            raise ValueError(
                "Generated norm source does not yet support "
                f"norm.kind={norm_kind!r} "
                f"for field {field_name!r} "
                f"({type(norm).__name__}). "
                f"Supported generated kinds today: {supported}."
            )

        index_names = tuple(f"i{index}" for index in range(len(shape)))
        if norm_kind == "rms":
            lines = ["    subtotal = 0.0"]
        elif norm_kind == "max":
            lines = ["    field_norm = 0.0"]
        else:
            supported = ", ".join(repr(kind) for kind in sorted(GENERATED_NORM_KINDS))
            raise ValueError(
                "Generated norm source does not yet support "
                f"norm.kind={norm_kind!r} "
                f"for field {field_name!r} "
                f"({type(norm).__name__}). "
                f"Supported generated kinds today: {supported}."
            )
        for depth, (index_name, bound) in enumerate(zip(index_names, shape, strict=True)):
            indent = "    " * (depth + 1)
            lines.append(f"{indent}for {index_name} in range({bound}):")
        index = "".join(f"[{index_name}]" for index_name in index_names)
        assignment_indent = "    " * (len(shape) + 1)
        if norm_kind == "rms":
            lines.append(f"{assignment_indent}subtotal += abs({name}{index}) ** 2")
            lines.append(f"    total += subtotal / {float(prod(shape))!r}")
        else:
            lines.append(f"{assignment_indent}item_norm = abs({name}{index})")
            lines.append(f"{assignment_indent}if item_norm > field_norm:")
            lines.append(f"{assignment_indent}    field_norm = item_norm")
            lines.append("    total += field_norm ** 2")
        return lines

    @staticmethod
    def looped_group_lines(
        *,
        key: NormLoopGroupKey,
        group: tuple[NormLoopGroupItem, ...],
    ) -> list[str]:
        rank, shape = key
        lines: list[str] = []
        for field, norm in group:
            norm_kind = getattr(norm, "kind", None)
            if norm_kind == "rms":
                lines.append(f"    subtotal_{field.translation_name} = 0.0")
            elif norm_kind == "max":
                lines.append(f"    field_norm_{field.translation_name} = 0.0")
            else:
                supported = ", ".join(repr(kind) for kind in sorted(GENERATED_NORM_KINDS))
                raise ValueError(
                    "Generated norm source does not yet support "
                    f"norm.kind={norm_kind!r} "
                    f"for field {field.state_name!r} "
                    f"({type(norm).__name__}). "
                    f"Supported generated kinds today: {supported}."
                )

        index_names = tuple(f"i{index}" for index in range(rank))
        for depth, (index_name, bound) in enumerate(zip(index_names, shape, strict=True)):
            indent = "    " * (depth + 1)
            lines.append(f"{indent}for {index_name} in range({bound}):")

        index = "".join(f"[{index_name}]" for index_name in index_names)
        assignment_indent = "    " * (rank + 1)
        for field, norm in group:
            name = field.translation_name
            norm_kind = getattr(norm, "kind", None)
            if norm_kind == "rms":
                lines.append(
                    f"{assignment_indent}subtotal_{name} += abs({name}{index}) ** 2"
                )
            else:
                lines.append(f"{assignment_indent}item_norm_{name} = abs({name}{index})")
                lines.append(
                    f"{assignment_indent}if item_norm_{name} > field_norm_{name}:"
                )
                lines.append(f"{assignment_indent}    field_norm_{name} = item_norm_{name}")

        for field, norm in group:
            name = field.translation_name
            norm_kind = getattr(norm, "kind", None)
            if norm_kind == "rms":
                lines.append(f"    total += subtotal_{name} / {float(prod(shape))!r}")
            else:
                lines.append(f"    total += field_norm_{name} ** 2")
        return lines

    @staticmethod
    def unravelled_field_lines(
        *,
        field_name: str,
        name: str,
        shape: tuple[int, ...],
        norm: object,
    ) -> list[str]:
        from itertools import product

        norm_kind = getattr(norm, "kind", None)
        if norm_kind == "rms":
            lines = ["    subtotal = 0.0"]
        elif norm_kind == "max":
            lines = ["    field_norm = 0.0"]
        else:
            supported = ", ".join(repr(kind) for kind in sorted(GENERATED_NORM_KINDS))
            raise ValueError(
                "Generated norm source does not yet support "
                f"norm.kind={norm_kind!r} "
                f"for field {field_name!r} "
                f"({type(norm).__name__}). "
                f"Supported generated kinds today: {supported}."
            )
        for index_tuple in product(*(range(dimension) for dimension in shape)):
            index = "".join(f"[{index}]" for index in index_tuple)
            if norm_kind == "rms":
                lines.append(f"    subtotal += abs({name}{index}) ** 2")
            else:
                lines.append(f"    item_norm = abs({name}{index})")
                lines.append("    if item_norm > field_norm:")
                lines.append("        field_norm = item_norm")
        if norm_kind == "rms":
            lines.append(f"    total += subtotal / {float(prod(shape))!r}")
        else:
            lines.append("    total += field_norm ** 2")
        return lines


__all__ = [
    "GENERATED_NORM_KINDS",
    "GeneratorNorm",
    "NormLoopGroupItem",
    "NormLoopGroupKey",
    "included_norm_entries",
]
