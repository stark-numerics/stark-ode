from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from stark.core.contracts.problem.frame import FrameLike
from stark.engines.generator.expression import GeneratorExpression

ElementwiseKernelKind = Literal["general", "delta", "update", "apply_translation"]


@dataclass(frozen=True, slots=True)
class GeneratorElementwiseSource:
    """Emit backend elementwise kernels for shaped array fields.

    The policy-level concept is elementwise generation: one backend kernel per
    field, with scalar coefficients passed as kernel inputs when needed. The
    current concrete emitter targets CuPy's `ElementwiseKernel`; other
    backends can use the same operation shape later without changing request
    names.
    """

    frame: FrameLike
    module_alias: str = "cp"
    module_import: str = "import cupy as cp"
    kernel_factory: str = "ElementwiseKernel"
    kernel_prefix: str = "stark_elementwise"

    def linear_combine(self, *, arity: int) -> str:
        return self.emit(kind="general", source_count=arity)

    def linear_fixed(
        self,
        *,
        kind: Literal["delta", "update"],
        coefficients: tuple[float, ...],
        stencil_scale: float = 1.0,
    ) -> str:
        return self.emit(
            kind=kind,
            source_count=len(coefficients),
            coefficients=coefficients,
            stencil_scale=stencil_scale,
        )

    def apply_translation(self) -> str:
        return self.emit(
            kind="apply_translation",
            source_count=1,
            coefficients=(1.0,),
        )

    def emit(
        self,
        *,
        kind: ElementwiseKernelKind,
        source_count: int,
        coefficients: tuple[float, ...] | None = None,
        stencil_scale: float = 1.0,
    ) -> str:
        if source_count < 1:
            raise ValueError(
                f"elementwise generated algebra requires at least one source; got {source_count}."
            )

        lines = [self.module_import, ""]
        for index, field in enumerate(self.frame.fields):
            self.require_looped_shape(field)
            lines.extend(
                self.elementwise_kernel_lines(
                    field_index=index,
                    kind=kind,
                    source_count=source_count,
                    coefficients=coefficients,
                    stencil_scale=stencil_scale,
                )
            )
            lines.append("")

        lines.extend(
            self.wrapper_lines(
                kind=kind,
                source_count=source_count,
            )
        )
        return "\n".join(lines) + "\n"

    def elementwise_kernel_lines(
        self,
        *,
        field_index: int,
        kind: ElementwiseKernelKind,
        source_count: int,
        coefficients: tuple[float, ...] | None,
        stencil_scale: float,
    ) -> list[str]:
        operation = self.operation(
            kind=kind,
            source_count=source_count,
            coefficients=coefficients,
            stencil_scale=stencil_scale,
        )
        kernel_name = f"_kernel_{field_index}"
        backend_name = f"{self.kernel_prefix}_{kind}_{source_count}_{field_index}"
        return [
            f"{kernel_name} = {self.module_alias}.{self.kernel_factory}(",
            f"    {self.input_parameters(kind=kind, source_count=source_count)!r},",
            "    'T out',",
            f"    {operation!r},",
            f"    {backend_name!r},",
            ")",
        ]

    @staticmethod
    def input_parameters(*, kind: ElementwiseKernelKind, source_count: int) -> str:
        parameters: list[str] = []
        if kind in {"delta", "update"}:
            parameters.append("float64 step")
        if kind in {"update", "apply_translation"}:
            parameters.append("T origin")
        if kind == "general":
            for index in range(source_count):
                parameters.append(f"float64 a{index}")
                parameters.append(f"T x{index}")
        else:
            parameters.extend(f"T x{index}" for index in range(source_count))
        return ", ".join(parameters)

    @staticmethod
    def operation(
        *,
        kind: ElementwiseKernelKind,
        source_count: int,
        coefficients: tuple[float, ...] | None,
        stencil_scale: float,
    ) -> str:
        sources = tuple(f"x{index}" for index in range(source_count))
        if kind == "general":
            expression = GeneratorExpression.from_runtime_coefficients(
                coefficients=tuple(f"a{index}" for index in range(source_count)),
                sources=sources,
            ).source()
        elif kind == "apply_translation":
            expression = "x0"
        else:
            if coefficients is None:
                raise ValueError("elementwise linear-fixed emission requires coefficients.")
            scaled = tuple(stencil_scale * coefficient for coefficient in coefficients)
            expression = GeneratorExpression.from_fixed_coefficients(
                coefficients=scaled,
                sources=sources,
                inline_coefficients=True,
            ).source()
            expression = f"step * ({expression})"

        if kind in {"update", "apply_translation"}:
            expression = f"origin + {expression}"
        return f"out = {expression}"

    def wrapper_lines(
        self,
        *,
        kind: ElementwiseKernelKind,
        source_count: int,
    ) -> list[str]:
        lines = [f"def kernel({self.wrapper_signature(kind=kind, source_count=source_count)}):"]
        for field_index, field in enumerate(self.frame.fields):
            target_root = "result" if kind in {"update", "apply_translation"} else "out"
            arguments = self.wrapper_arguments(
                field=field,
                kind=kind,
                source_count=source_count,
                target_root=target_root,
            )
            lines.append(f"    _kernel_{field_index}({', '.join(arguments)})")
        lines.append(
            "    return result"
            if kind in {"update", "apply_translation"}
            else "    return out"
        )
        return lines

    @staticmethod
    def wrapper_signature(*, kind: ElementwiseKernelKind, source_count: int) -> str:
        if kind == "general":
            parameters: list[str] = []
            for index in range(source_count):
                parameters.append(f"a{index}")
                parameters.append(f"x{index}")
            parameters.append("out")
            return ", ".join(parameters)
        if kind == "delta":
            return ", ".join(("step", *(f"x{index}" for index in range(source_count)), "out"))
        if kind == "update":
            return ", ".join(
                ("step", "origin", *(f"x{index}" for index in range(source_count)), "result")
            )
        if kind == "apply_translation":
            return ", ".join(("origin", "x0", "result"))
        raise ValueError(f"Unknown elementwise kernel kind: {kind!r}.")

    @staticmethod
    def wrapper_arguments(
        *,
        field: Any,
        kind: ElementwiseKernelKind,
        source_count: int,
        target_root: str,
    ) -> list[str]:
        arguments: list[str] = []
        if kind in {"delta", "update"}:
            arguments.append("step")
        if kind in {"update", "apply_translation"}:
            arguments.append(field.state_expression("origin"))
        if kind == "general":
            for index in range(source_count):
                arguments.append(f"a{index}")
                arguments.append(field.translation_expression(f"x{index}"))
        else:
            for index in range(source_count):
                arguments.append(field.translation_expression(f"x{index}"))
        if kind in {"update", "apply_translation"}:
            arguments.append(field.state_expression(target_root))
        else:
            arguments.append(field.translation_expression(target_root))
        return arguments

    @staticmethod
    def require_looped_shape(field: Any) -> tuple[int, ...]:
        policy = field.policy
        shape = field.shape
        if policy.kind != "looped" or shape is None:
            raise ValueError(
                "Elementwise generated algebra requires shaped looped frame "
                f"fields; got field {field.state_name!r} with "
                f"policy.kind={policy.kind!r}."
            )
        return tuple(shape)


__all__ = ["ElementwiseKernelKind", "GeneratorElementwiseSource"]
