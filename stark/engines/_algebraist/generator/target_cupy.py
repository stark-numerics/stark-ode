from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

from stark.core.contracts.frame import FrameLike
from stark.engines._algebraist.arity import AlgebraistArity
from stark.engines._algebraist.inner_product import included_inner_product_entries
from stark.engines._algebraist.norm import included_norm_entries
from stark.engines._algebraist.generator.expression import AlgebraistGeneratorEmitterExpression
from stark.engines._algebraist.stencil import AlgebraistStencil


Kind = str


@dataclass(frozen=True, slots=True)
class AlgebraistGeneratorTargetCupy:
    """
    Emit CuPy-native algebra kernels for shaped array frames.

    CuPy arrays should not be driven through generated Python index loops:
    those loops perform scalar GPU accesses from Python and are extremely slow.
    This target emits `cupy.ElementwiseKernel` wrappers for linear-combine and
    state-apply work, leaving CuPy to compile the elementwise operation for the
    active device. Norms and inner products are emitted as CuPy reductions.
    """

    kernel_prefix: str = "stark_cupy_algebraist"

    def source_linear_combine(self, frame: FrameLike, request: AlgebraistArity) -> str:
        return self._source_algebra(frame=frame, kind="general", source_count=request.value)

    def source_linear_fixed(self, frame: FrameLike, stencil: AlgebraistStencil) -> str:
        coefficients = tuple(float(coefficient) for coefficient in stencil.coefficients)
        kind = "update" if stencil.apply else "delta"
        return self._source_algebra(
            frame=frame,
            kind=kind,
            source_count=len(coefficients),
            coefficients=coefficients,
            stencil_scale=float(stencil.scale),
        )

    def source_unit_apply(self, frame: FrameLike) -> str:
        return self._source_algebra(
            frame=frame,
            kind="unit_apply",
            source_count=1,
            coefficients=(1.0,),
        )

    def source_norm(self, frame: FrameLike) -> str:
        lines = ["import cupy as cp", "", "def kernel(translation):", "    total = 0.0"]
        for field, norm in included_norm_entries(frame):
            shape = self._require_looped_shape(field)
            name = field.translation_expression("translation")
            norm_kind = getattr(norm, "kind", None)
            if norm_kind == "rms":
                scale = float(prod(shape))
                lines.append(f"    total = total + cp.sum(cp.abs({name}) ** 2) / {scale!r}")
                continue
            if norm_kind == "max":
                lines.append(f"    field_norm = cp.max(cp.abs({name}))")
                lines.append("    total = total + field_norm * field_norm")
                continue
            raise ValueError("CuPy generated norm requires RMS or max norm fields.")
        lines.append("    return cp.sqrt(total)")
        return "\n".join(lines) + "\n"

    def source_inner_product(self, frame: FrameLike) -> str:
        lines = ["import cupy as cp", "", "def kernel(left, right):", "    total = 0.0"]
        for field, inner_product in included_inner_product_entries(frame):
            shape = self._require_looped_shape(field)
            left = field.translation_expression("left")
            right = field.translation_expression("right")
            inner_product_kind = getattr(inner_product, "kind", None)
            if inner_product_kind == "l2":
                scale = 1.0
            elif inner_product_kind == "rms":
                scale = float(prod(shape))
            else:
                raise ValueError(
                    "CuPy generated inner product requires L2 or RMS inner product fields."
                )
            lines.append(f"    total = total + cp.sum({left} * {right}) / {scale!r}")
        lines.append("    return total")
        return "\n".join(lines) + "\n"

    def _source_algebra(
        self,
        *,
        frame: FrameLike,
        kind: Kind,
        source_count: int,
        coefficients: tuple[float, ...] | None = None,
        stencil_scale: float = 1.0,
    ) -> str:
        lines = ["import cupy as cp", ""]
        for index, field in enumerate(frame.fields):
            self._require_looped_shape(field)
            lines.extend(
                self._elementwise_kernel_lines(
                    field_index=index,
                    field=field,
                    kind=kind,
                    source_count=source_count,
                    coefficients=coefficients,
                    stencil_scale=stencil_scale,
                )
            )
            lines.append("")
        lines.extend(
            self._wrapper_lines(
                frame=frame,
                kind=kind,
                source_count=source_count,
            )
        )
        return "\n".join(lines) + "\n"

    def _elementwise_kernel_lines(
        self,
        *,
        field_index: int,
        field: Any,
        kind: Kind,
        source_count: int,
        coefficients: tuple[float, ...] | None,
        stencil_scale: float,
    ) -> list[str]:
        kernel_name = f"_kernel_{field_index}"
        in_params = self._in_params(kind=kind, source_count=source_count)
        operation = self._operation(
            kind=kind,
            source_count=source_count,
            coefficients=coefficients,
            stencil_scale=stencil_scale,
        )
        cupy_name = f"{self.kernel_prefix}_{kind}_{source_count}_{field_index}"
        return [
            f"{kernel_name} = cp.ElementwiseKernel(",
            f"    {in_params!r},",
            "    'T out',",
            f"    {operation!r},",
            f"    {cupy_name!r},",
            ")",
        ]

    @staticmethod
    def _in_params(*, kind: Kind, source_count: int) -> str:
        parameters: list[str] = []
        if kind in {"delta", "update"}:
            parameters.append("float64 step")
        if kind in {"update", "unit_apply"}:
            parameters.append("T origin")
        if kind == "general":
            for index in range(source_count):
                parameters.append(f"float64 a{index}")
                parameters.append(f"T x{index}")
        else:
            parameters.extend(f"T x{index}" for index in range(source_count))
        return ", ".join(parameters)

    @staticmethod
    def _operation(
        *,
        kind: Kind,
        source_count: int,
        coefficients: tuple[float, ...] | None,
        stencil_scale: float,
    ) -> str:
        sources = tuple(f"x{index}" for index in range(source_count))
        if kind == "general":
            expression = AlgebraistGeneratorEmitterExpression.from_runtime_coefficients(
                coefficients=tuple(f"a{index}" for index in range(source_count)),
                sources=sources,
            ).source()
        elif kind == "unit_apply":
            expression = "x0"
        else:
            if coefficients is None:
                raise ValueError("fixed coefficients are required for CuPy linear_fixed emission.")
            scaled = tuple(stencil_scale * coefficient for coefficient in coefficients)
            expression = AlgebraistGeneratorEmitterExpression.from_fixed_coefficients(
                coefficients=scaled,
                sources=sources,
                inline_coefficients=True,
            ).source()
            expression = f"step * ({expression})"

        if kind in {"update", "unit_apply"}:
            expression = f"origin + {expression}"
        return f"out = {expression}"

    def _wrapper_lines(
        self,
        *,
        frame: FrameLike,
        kind: Kind,
        source_count: int,
    ) -> list[str]:
        if kind == "general":
            signature = ", ".join(
                (
                    *(
                        item
                        for index in range(source_count)
                        for item in (f"a{index}", f"x{index}")
                    ),
                    "out",
                )
            )
        elif kind == "delta":
            signature = ", ".join(("step", *(f"x{index}" for index in range(source_count)), "out"))
        elif kind == "update":
            signature = ", ".join(("step", "origin", *(f"x{index}" for index in range(source_count)), "result"))
        elif kind == "unit_apply":
            signature = ", ".join(("origin", *(f"x{index}" for index in range(source_count)), "result"))
        else:
            raise ValueError(f"Unknown CuPy algebra kind: {kind!r}")

        lines = [f"def kernel({signature}):"]
        for field_index, field in enumerate(frame.fields):
            target_root = "result" if kind in {"update", "unit_apply"} else "out"
            arguments = self._wrapper_arguments(
                field=field,
                kind=kind,
                source_count=source_count,
                target_root=target_root,
            )
            lines.append(f"    _kernel_{field_index}({', '.join(arguments)})")
        lines.append("    return result" if kind in {"update", "unit_apply"} else "    return out")
        return lines

    @staticmethod
    def _wrapper_arguments(
        *,
        field: Any,
        kind: Kind,
        source_count: int,
        target_root: str,
    ) -> list[str]:
        arguments: list[str] = []
        if kind in {"delta", "update"}:
            arguments.append("step")
        if kind in {"update", "unit_apply"}:
            arguments.append(field.state_expression("origin"))
        if kind == "general":
            for index in range(source_count):
                arguments.append(f"a{index}")
                arguments.append(field.translation_expression(f"x{index}"))
        else:
            for index in range(source_count):
                arguments.append(field.translation_expression(f"x{index}"))
        if kind in {"update", "unit_apply"}:
            arguments.append(field.state_expression(target_root))
        else:
            arguments.append(field.translation_expression(target_root))
        return arguments

    @staticmethod
    def _require_looped_shape(field: Any) -> tuple[int, ...]:
        policy = field.policy
        shape = getattr(policy, "shape", None)
        if shape is None:
            shape = getattr(field, "shape", None)
        if getattr(policy, "kind", None) != "looped" or shape is None:
            raise ValueError("CuPy generated algebra requires shaped looped frame fields.")
        return tuple(shape)


__all__ = ["AlgebraistGeneratorTargetCupy"]
