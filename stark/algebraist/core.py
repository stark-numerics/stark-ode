from __future__ import annotations

from collections.abc import Callable, Sequence
from types import MappingProxyType

from stark.algebraist.build import build_function
from stark.algebraist.codegen import AlgebraistCodegen
from stark.algebraist.explicit import AlgebraistExplicitSchemeBinder
from stark.algebraist.fields import AlgebraistField
from stark.algebraist.names import (
    combine_kernel_name,
    combine_wrapper_name,
    linear_combine_names,
)
from stark.algebraist.paths import path_expression
from stark.algebraist.signatures import apply_signature, combine_signature
from stark.algebraist.tableau import AlgebraistTableauBinder, ButcherTableauLike
from stark.contracts.acceleration import AcceleratorLike


class Algebraist:
    """Generate inspectable translation kernels from field metadata.

    Generated source is retained on `sources`, `kernel_sources`, and
    `wrapper_sources` so users can inspect exactly what was emitted.
    """

    __slots__ = (
        "accelerator",
        "codegen",
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
        fields: Sequence[AlgebraistField],
        accelerator: AcceleratorLike | None = None,
        fused_up_to: int = 12,
        generate_norm: str | None = None,
    ) -> None:
        if not fields:
            raise ValueError("Algebraist needs at least one translation field.")
        if fused_up_to < 2:
            raise ValueError("Algebraist.fused_up_to must be at least 2.")
        if generate_norm not in {None, "l2", "rms"}:
            raise ValueError("Algebraist.generate_norm must be None, 'l2', or 'rms'.")

        self.accelerator = accelerator
        self.codegen = AlgebraistCodegen()
        self.fields = tuple(fields)
        self.fused_up_to = fused_up_to
        self.generate_norm = generate_norm

        kernels, kernel_sources = self.build_kernels()
        self.kernels = MappingProxyType(kernels)
        self.kernel_sources = MappingProxyType(kernel_sources)

        wrappers, wrapper_sources = self.build_wrappers()
        self.wrappers = MappingProxyType(wrappers)
        self.wrapper_sources = MappingProxyType(wrapper_sources)

        self.sources = MappingProxyType({**kernel_sources, **wrapper_sources})
        self.linear_combine = tuple(
            wrappers[name] for name in linear_combine_names(fused_up_to)
        )
        self.apply = wrappers["apply"]
        self.norm = wrappers.get("norm")

    def compile_examples(self, *probes: object) -> None:
        accelerator = self.accelerator
        if accelerator is None:
            return

        if len(probes) != len(self.fields):
            raise ValueError(f"Expected {len(self.fields)} probes, got {len(probes)}.")

        accelerator.compile_examples(
            self.kernels["scale_kernel"],
            combine_signature(1, probes),
        )

        for term_count in range(2, self.fused_up_to + 1):
            accelerator.compile_examples(
                self.kernels[f"combine{term_count}_kernel"],
                combine_signature(term_count, probes),
            )

        accelerator.compile_examples(
            self.kernels["apply_kernel"],
            apply_signature(probes),
        )

        if self.generate_norm is not None:
            accelerator.compile_examples(self.kernels["norm_kernel"], tuple(probes))

    def bind_tableau(self, tableau: ButcherTableauLike):
        return AlgebraistTableauBinder(self)(tableau)

    def bind_explicit_scheme(self, tableau: ButcherTableauLike):
        return AlgebraistExplicitSchemeBinder(self)(tableau)

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
            name, function, source = self.make_combine_wrapper(term_count)
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

    def make_combine_kernel(
        self,
        term_count: int,
    ) -> tuple[str, Callable[..., object], str]:
        kernel_name = combine_kernel_name(term_count)

        field_arguments = [
            f"out_{field.translation_name}" for field in self.fields
        ]
        term_arguments: list[str] = []

        for index in range(term_count):
            term_arguments.append(f"a{index}")
            for field in self.fields:
                term_arguments.append(f"x{index}_{field.translation_name}")

        body = "\n".join(
            self.codegen.combine_assignment(field, term_count)
            for field in self.fields
        )
        source = (
            f"def {kernel_name}({', '.join(field_arguments + term_arguments)}):\n"
            f"{body}\n"
        )

        return (
            kernel_name,
            build_function(kernel_name, source, accelerator=self.accelerator),
            source,
        )

    def make_combine_wrapper(
        self,
        term_count: int,
    ) -> tuple[str, Callable[..., object], str]:
        wrapper_name = combine_wrapper_name(term_count)
        kernel_name = combine_kernel_name(term_count)

        parameters = ["out"]
        kernel_arguments = [
            path_expression("out", field.translation_path)
            for field in self.fields
        ]

        for index in range(term_count):
            parameters.append(f"a{index}")
            parameters.append(f"x{index}")

            kernel_arguments.append(f"a{index}")
            kernel_arguments.extend(
                path_expression(f"x{index}", field.translation_path)
                for field in self.fields
            )

        source = (
            f"def {wrapper_name}({', '.join(parameters)}):\n"
            f" kernel({', '.join(kernel_arguments)})\n"
            " return out\n"
        )

        return (
            wrapper_name,
            build_function(
                wrapper_name,
                source,
                namespace={"kernel": self.kernels[kernel_name]},
            ),
            source,
        )

    def make_apply_kernel(self) -> tuple[str, Callable[..., object], str]:
        kernel_name = "apply_kernel"

        origin_arguments = [
            f"origin_{field.state_name}" for field in self.fields
        ]
        delta_arguments = [
            f"delta_{field.translation_name}" for field in self.fields
        ]
        result_arguments = [
            f"result_{field.state_name}" for field in self.fields
        ]

        body = "\n".join(
            self.codegen.apply_assignment(field)
            for field in self.fields
        )
        source = (
            f"def {kernel_name}("
            f"{', '.join(origin_arguments + delta_arguments + result_arguments)}"
            "):\n"
            f"{body}\n"
        )

        return (
            kernel_name,
            build_function(kernel_name, source, accelerator=self.accelerator),
            source,
        )

    def make_apply_wrapper(self) -> tuple[str, Callable[..., object], str]:
        arguments = [
            path_expression("origin", field.state_path)
            for field in self.fields
        ]
        arguments.extend(
            path_expression("translation", field.translation_path)
            for field in self.fields
        )
        arguments.extend(
            path_expression("result", field.state_path)
            for field in self.fields
        )

        source = (
            "def apply(translation, origin, result):\n"
            f" kernel({', '.join(arguments)})\n"
            " return None\n"
        )

        return (
            "apply",
            build_function(
                "apply",
                source,
                namespace={"kernel": self.kernels["apply_kernel"]},
            ),
            source,
        )

    def make_norm_kernel(
        self,
        kind: str,
    ) -> tuple[str, Callable[..., object], str]:
        signature = ", ".join(field.translation_name for field in self.fields)
        body = "\n".join(self.codegen.norm_body(self.fields, kind))
        source = f"def norm_kernel({signature}):\n{body}\n"

        return (
            "norm_kernel",
            build_function("norm_kernel", source, accelerator=self.accelerator),
            source,
        )

    def make_norm_wrapper(self) -> tuple[str, Callable[..., object], str]:
        arguments = ", ".join(
            path_expression("translation", field.translation_path)
            for field in self.fields
        )

        source = (
            "def norm(translation):\n"
            f" return kernel({arguments})\n"
        )

        return (
            "norm",
            build_function(
                "norm",
                source,
                namespace={"kernel": self.kernels["norm_kernel"]},
            ),
            source,
        )