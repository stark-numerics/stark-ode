from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from math import prod
from types import MappingProxyType

from stark.contracts.acceleration import AcceleratorLike


@dataclass(frozen=True, slots=True)
class AlgebraistField:
    """Describe one generated translation field."""

    name: str
    style: str = "broadcast"
    rank: int | None = None
    shape: tuple[int, ...] | None = None
    apply_to: str | None = None
    include_in_norm: bool = True


class Algebraist:
    """
    Generate inspectable translation kernels from field metadata.

    Generated source is retained on `sources`, `kernel_sources`, and
    `wrapper_sources` so users can inspect exactly what was emitted.
    """

    __slots__ = (
        "accelerator",
        "fields",
        "fused_up_to",
        "generate_norm",
        "kernels",
        "kernel_sources",
        "wrappers",
        "wrapper_sources",
        "sources",
        "linear_combine",
        "apply",
        "norm",
    )

    def __init__(
        self,
        *,
        fields: Sequence[str | AlgebraistField],
        accelerator: AcceleratorLike | None = None,
        fused_up_to: int = 7,
        generate_norm: str | None = None,
    ) -> None:
        if not fields:
            raise ValueError("Algebraist needs at least one translation field.")
        if fused_up_to < 2:
            raise ValueError("Algebraist.fused_up_to must be at least 2.")
        if generate_norm not in {None, "l2", "rms"}:
            raise ValueError("Algebraist.generate_norm must be None, 'l2', or 'rms'.")

        self.accelerator = accelerator
        self.fields = tuple(coerce_field(field) for field in fields)
        self.fused_up_to = fused_up_to
        self.generate_norm = generate_norm

        kernels, kernel_sources = self.build_kernels()
        self.kernels = MappingProxyType(kernels)
        self.kernel_sources = MappingProxyType(kernel_sources)

        wrappers, wrapper_sources = self.build_wrappers()
        self.wrappers = MappingProxyType(wrappers)
        self.wrapper_sources = MappingProxyType(wrapper_sources)
        self.sources = MappingProxyType({**kernel_sources, **wrapper_sources})
        self.linear_combine = tuple(wrappers[name] for name in linear_combine_names(fused_up_to))
        self.apply = wrappers["apply"]
        self.norm = wrappers.get("norm")

    @property
    def linear_combination(self) -> tuple[Callable[..., object], ...]:
        return self.linear_combine

    def compile_examples(self, *probes: object) -> None:
        accelerator = self.accelerator
        if accelerator is None:
            return
        if len(probes) != len(self.fields):
            raise ValueError(f"Expected {len(self.fields)} probes, got {len(probes)}.")

        accelerator.compile_examples(self.kernels["scale_kernel"], combine_signature(1, probes))
        for term_count in range(2, self.fused_up_to + 1):
            accelerator.compile_examples(self.kernels[f"combine{term_count}_kernel"], combine_signature(term_count, probes))

        accelerator.compile_examples(self.kernels["apply_kernel"], apply_signature(probes))
        if self.generate_norm is not None:
            accelerator.compile_examples(self.kernels["norm_kernel"], tuple(probes))

    def build_kernels(self) -> tuple[dict[str, Callable[..., object]], dict[str, str]]:
        kernels: dict[str, Callable[..., object]] = {}
        sources: dict[str, str] = {}

        name, function, source = self.make_combine_kernel(1)
        kernels[name] = function
        sources[name] = source

        for term_count in range(2, self.fused_up_to + 1):
            name, function, source = self.make_combine_kernel(term_count)
            kernels[name] = function
            sources[name] = source

        name, function, source = self.make_apply_kernel()
        kernels[name] = function
        sources[name] = source

        if self.generate_norm is not None:
            name, function, source = self.make_norm_kernel(self.generate_norm)
            kernels[name] = function
            sources[name] = source

        return kernels, sources

    def build_wrappers(self) -> tuple[dict[str, Callable[..., object]], dict[str, str]]:
        wrappers: dict[str, Callable[..., object]] = {}
        sources: dict[str, str] = {}

        for term_count in range(1, self.fused_up_to + 1):
            name, function, source = self.make_combine_wrapper(term_count, wrappers)
            wrappers[name] = function
            sources[name] = source

        name, function, source = self.make_apply_wrapper()
        wrappers[name] = function
        sources[name] = source

        if self.generate_norm is not None:
            name, function, source = self.make_norm_wrapper()
            wrappers[name] = function
            sources[name] = source

        return wrappers, sources

    def make_combine_kernel(self, term_count: int) -> tuple[str, Callable[..., object], str]:
        kernel_name = combine_kernel_name(term_count)
        field_arguments = [f"out_{field.name}" for field in self.fields]

        term_arguments: list[str] = []
        for index in range(term_count):
                term_arguments.append(f"a{index}")
                for field in self.fields:
                    term_arguments.append(f"x{index}_{field.name}")

        body = "\n".join(field_combine_assignment(field, term_count) for field in self.fields)
        source = f"def {kernel_name}({', '.join(field_arguments + term_arguments)}):\n{body}\n"
        return kernel_name, build_function(kernel_name, source, accelerator=self.accelerator), source

    def make_combine_wrapper(
        self,
        term_count: int,
        known_wrappers: dict[str, Callable[..., object]],
    ) -> tuple[str, Callable[..., object], str]:
        wrapper_name = combine_wrapper_name(term_count)
        kernel_name = combine_kernel_name(term_count)

        parameters = ["out"]
        kernel_arguments = [f"out.{field.name}" for field in self.fields]
        for index in range(term_count):
            parameters.append(f"a{index}")
            parameters.append(f"x{index}")
            kernel_arguments.append(f"a{index}")
            kernel_arguments.extend(f"x{index}.{field.name}" for field in self.fields)
        source = (
            f"def {wrapper_name}({', '.join(parameters)}):\n"
            f"    kernel({', '.join(kernel_arguments)})\n"
            "    return out\n"
        )
        del known_wrappers
        return wrapper_name, build_function(wrapper_name, source, namespace={"kernel": self.kernels[kernel_name]}), source

    def make_apply_kernel(self) -> tuple[str, Callable[..., object], str]:
        kernel_name = "apply_kernel"
        origin_arguments = [f"origin_{state_field_name(field)}" for field in self.fields]
        delta_arguments = [f"delta_{field.name}" for field in self.fields]
        result_arguments = [f"result_{state_field_name(field)}" for field in self.fields]
        body = "\n".join(field_apply_assignment(field) for field in self.fields)
        source = f"def {kernel_name}({', '.join(origin_arguments + delta_arguments + result_arguments)}):\n{body}\n"
        return kernel_name, build_function(kernel_name, source, accelerator=self.accelerator), source

    def make_apply_wrapper(self) -> tuple[str, Callable[..., object], str]:
        arguments = [f"origin.{state_field_name(field)}" for field in self.fields]
        arguments.extend(f"translation.{field.name}" for field in self.fields)
        arguments.extend(f"result.{state_field_name(field)}" for field in self.fields)
        source = (
            "def apply(translation, origin, result):\n"
            f"    kernel({', '.join(arguments)})\n"
            "    return None\n"
        )
        return "apply", build_function("apply", source, namespace={"kernel": self.kernels["apply_kernel"]}), source

    def make_norm_kernel(self, kind: str) -> tuple[str, Callable[..., object], str]:
        signature = ", ".join(field.name for field in self.fields)
        body = "\n".join(field_norm_body(self.fields, kind))
        source = f"def norm_kernel({signature}):\n{body}\n"
        return "norm_kernel", build_function("norm_kernel", source, accelerator=self.accelerator), source

    def make_norm_wrapper(self) -> tuple[str, Callable[..., object], str]:
        arguments = ", ".join(f"translation.{field.name}" for field in self.fields)
        source = (
            "def norm(translation):\n"
            f"    return kernel({arguments})\n"
        )
        return "norm", build_function("norm", source, namespace={"kernel": self.kernels["norm_kernel"]}), source


def coerce_field(field: str | AlgebraistField) -> AlgebraistField:
    if not isinstance(field, AlgebraistField):
        field = AlgebraistField(str(field))
    return normalize_field(field)


def normalize_field(field: AlgebraistField) -> AlgebraistField:
    shape = tuple(field.shape) if field.shape is not None else None
    rank = field.rank

    if shape is not None:
        if not shape or any(dimension <= 0 for dimension in shape):
            raise ValueError(f"Field {field.name!r} has invalid shape {shape!r}.")
        if rank is None:
            rank = len(shape)
        elif rank != len(shape):
            raise ValueError(f"Field {field.name!r} rank {rank} does not match shape {shape!r}.")

    if field.style == "looped":
        if rank is None:
            raise ValueError(f"Looped field {field.name!r} needs an explicit rank or shape.")
        return replace(field, rank=rank, shape=shape)

    if field.style == "small_fixed":
        if shape is None:
            raise ValueError(f"Small-fixed field {field.name!r} needs an explicit shape.")
        if prod(shape) > 16:
            raise ValueError(f"Small-fixed field {field.name!r} is too large for unrolled code: {shape!r}.")
        return replace(field, rank=rank, shape=shape)

    if field.style != "broadcast":
        raise ValueError(f"Unknown AlgebraistField style {field.style!r} for field {field.name!r}.")
    return replace(field, rank=rank, shape=shape)


def linear_combine_names(fused_up_to: int) -> tuple[str, ...]:
    return tuple(combine_wrapper_name(term_count) for term_count in range(1, fused_up_to + 1))


def combine_wrapper_name(term_count: int) -> str:
    return "scale" if term_count == 1 else f"combine{term_count}"


def combine_kernel_name(term_count: int) -> str:
    return "scale_kernel" if term_count == 1 else f"combine{term_count}_kernel"


def state_field_name(field: AlgebraistField) -> str:
    return field.apply_to or field.name


def combine_signature(term_count: int, probes: Sequence[object]) -> tuple[object, ...]:
    arguments: list[object] = list(probes)
    for _ in range(term_count):
        arguments.append(1.0)
        arguments.extend(probes)
    return tuple(arguments)


def apply_signature(probes: Sequence[object]) -> tuple[object, ...]:
    return tuple(probes) + tuple(probes) + tuple(probes)


def field_combine_assignment(field: AlgebraistField, term_count: int) -> str:
    if field.style == "small_fixed":
        return small_fixed_combine_assignment(field, term_count)
    if field.style == "looped":
        return looped_combine_assignment(field, term_count)
    return broadcast_combine_assignment(field, term_count)


def field_apply_assignment(field: AlgebraistField) -> str:
    if field.style == "small_fixed":
        return small_fixed_apply_assignment(field)
    if field.style == "looped":
        return looped_apply_assignment(field)
    return broadcast_apply_assignment(field)


def broadcast_combine_assignment(field: AlgebraistField, term_count: int) -> str:
    value = " + ".join(f"a{index} * x{index}_{field.name}" for index in range(term_count))
    return f"    out_{field.name}[...] = {value}"


def looped_combine_assignment(field: AlgebraistField, term_count: int) -> str:
    rank = field.rank
    assert rank is not None
    index_names = [f"i{index}" for index in range(rank)]
    shape_terms = [f"shape_{index}" for index in range(rank)]
    if rank == 1:
        shape_binding = f"    {shape_terms[0]} = out_{field.name}.shape[0]"
    else:
        shape_binding = f"    {', '.join(shape_terms)} = out_{field.name}.shape"

    lines = [shape_binding]
    indent = "    "
    for index_name, shape_name in zip(index_names, shape_terms, strict=True):
        lines.append(f"{indent}for {index_name} in range({shape_name}):")
        indent += "    "

    location = ", ".join(index_names)
    value = " + ".join(f"a{index} * x{index}_{field.name}[{location}]" for index in range(term_count))
    lines.append(f"{indent}out_{field.name}[{location}] = {value}")
    return "\n".join(lines)


def small_fixed_combine_assignment(field: AlgebraistField, term_count: int) -> str:
    assert field.shape is not None
    lines: list[str] = []
    for location in fixed_locations(field.shape):
        value = " + ".join(f"a{index} * x{index}_{field.name}[{location}]" for index in range(term_count))
        lines.append(f"    out_{field.name}[{location}] = {value}")
    return "\n".join(lines)


def broadcast_apply_assignment(field: AlgebraistField) -> str:
    target = state_field_name(field)
    return f"    result_{target}[...] = origin_{target} + delta_{field.name}"


def looped_apply_assignment(field: AlgebraistField) -> str:
    target = state_field_name(field)
    rank = field.rank
    assert rank is not None
    index_names = [f"i{index}" for index in range(rank)]
    shape_terms = [f"shape_{index}" for index in range(rank)]
    if rank == 1:
        shape_binding = f"    {shape_terms[0]} = delta_{field.name}.shape[0]"
    else:
        shape_binding = f"    {', '.join(shape_terms)} = delta_{field.name}.shape"

    lines = [shape_binding]
    indent = "    "
    for index_name, shape_name in zip(index_names, shape_terms, strict=True):
        lines.append(f"{indent}for {index_name} in range({shape_name}):")
        indent += "    "

    location = ", ".join(index_names)
    lines.append(f"{indent}result_{target}[{location}] = origin_{target}[{location}] + delta_{field.name}[{location}]")
    return "\n".join(lines)


def small_fixed_apply_assignment(field: AlgebraistField) -> str:
    target = state_field_name(field)
    assert field.shape is not None
    lines = [
        f"    result_{target}[{location}] = origin_{target}[{location}] + delta_{field.name}[{location}]"
        for location in fixed_locations(field.shape)
    ]
    return "\n".join(lines)


def field_norm_body(fields: Sequence[AlgebraistField], kind: str) -> list[str]:
    lines = ["    total = 0.0"]
    if kind == "rms":
        lines.append("    count = 0")

    for field in fields:
        if not field.include_in_norm:
            continue
        if field.style == "small_fixed":
            lines.extend(small_fixed_norm_body(field, kind))
        elif field.style == "looped":
            lines.extend(looped_norm_body(field, kind))
        else:
            lines.extend(broadcast_norm_body(field, kind))

    if kind == "rms":
        lines.append("    return 0.0 if count == 0 else (total / count) ** 0.5")
    else:
        lines.append("    return total ** 0.5")
    return lines


def broadcast_norm_body(field: AlgebraistField, kind: str) -> list[str]:
    lines = [f"    for value in {field.name}.ravel():", "        total += value * value"]
    if kind == "rms":
        lines.append(f"    count += {field.name}.size")
    return lines


def looped_norm_body(field: AlgebraistField, kind: str) -> list[str]:
    rank = field.rank
    assert rank is not None
    index_names = [f"i{index}" for index in range(rank)]
    shape_terms = [f"shape_{index}" for index in range(rank)]
    if rank == 1:
        lines = [f"    {shape_terms[0]} = {field.name}.shape[0]"]
    else:
        lines = [f"    {', '.join(shape_terms)} = {field.name}.shape"]

    indent = "    "
    for index_name, shape_name in zip(index_names, shape_terms, strict=True):
        lines.append(f"{indent}for {index_name} in range({shape_name}):")
        indent += "    "

    location = ", ".join(index_names)
    lines.append(f"{indent}value = {field.name}[{location}]")
    lines.append(f"{indent}total += value * value")
    if kind == "rms":
        lines.append(f"    count += {field.name}.size")
    return lines


def small_fixed_norm_body(field: AlgebraistField, kind: str) -> list[str]:
    assert field.shape is not None
    lines: list[str] = []
    for location in fixed_locations(field.shape):
        lines.append(f"    value = {field.name}[{location}]")
        lines.append("    total += value * value")
    if kind == "rms":
        lines.append(f"    count += {prod(field.shape)}")
    return lines


def fixed_locations(shape: tuple[int, ...]) -> tuple[str, ...]:
    locations: list[str] = []

    def visit(prefix: tuple[int, ...], depth: int) -> None:
        if depth == len(shape):
            if len(prefix) == 1:
                locations.append(str(prefix[0]))
            else:
                locations.append(", ".join(str(item) for item in prefix))
            return
        for index in range(shape[depth]):
            visit(prefix + (index,), depth + 1)

    visit((), 0)
    return tuple(locations)


def build_function(
    name: str,
    source: str,
    *,
    accelerator: AcceleratorLike | None = None,
    namespace: dict[str, object] | None = None,
) -> Callable[..., object]:
    local_namespace: dict[str, object] = {} if namespace is None else dict(namespace)
    exec(source, local_namespace)
    function = local_namespace[name]
    if accelerator is None:
        return function
    return accelerator.decorate(cache=False)(function)


__all__ = [
    "Algebraist",
    "AlgebraistField",
    "apply_signature",
    "combine_signature",
    "coerce_field",
    "linear_combine_names",
    "state_field_name",
]
