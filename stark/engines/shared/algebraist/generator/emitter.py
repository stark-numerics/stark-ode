from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Literal

from stark.core.contracts.frame import FrameLike
from stark.engines.shared.algebraist.arity import AlgebraistArity
from stark.engines.shared.algebraist.stencil import AlgebraistStencil
from stark.engines.shared.algebraist.generator.expression import AlgebraistGeneratorEmitterExpression
from stark.engines.shared.algebraist.generator.target import (
    AlgebraistGeneratorTarget,
    AlgebraistGeneratorTargetFunctional,
    AlgebraistGeneratorTargetMutable,
    AlgebraistGeneratorTargetMutableVectorized,
)

Kind = Literal["general", "delta", "update"]


@dataclass(frozen=True, slots=True)
class AlgebraistGeneratorEmitter:
    """Emit complete source strings for generated Algebraist kernels."""

    frame: FrameLike
    target: AlgebraistGeneratorTarget = field(default_factory=AlgebraistGeneratorTargetMutable)

    def _uses_functional_updates(self) -> bool:
        return isinstance(self.target, AlgebraistGeneratorTargetFunctional)

    def _uses_vectorized_arrays(self) -> bool:
        return isinstance(
            self.target,
            (
                AlgebraistGeneratorTargetFunctional,
                AlgebraistGeneratorTargetMutableVectorized,
            ),
        )

    @staticmethod
    def _policy_kind(policy: object) -> str | None:
        return getattr(policy, "kind", None)

    def general(self, request: AlgebraistArity) -> str:
        arity = request.value
        return self._emit(kind="general", arity=arity)

    def specialist(self, stencil: AlgebraistStencil) -> str:
        coefficients = tuple(float(coefficient) for coefficient in stencil.coefficients)
        kind: Kind = "update" if stencil.apply else "delta"
        return self._emit(kind=kind, coefficients=coefficients, stencil_scale=float(stencil.scale))

    def unit_apply(self) -> str:
        """Emit `result = origin + translation` without a runtime step argument."""

        return self._emit_unit_apply()

    def _emit(
        self,
        *,
        kind: Kind,
        arity: int | None = None,
        coefficients: tuple[float, ...] | None = None,
        stencil_scale: float = 1.0,
    ) -> str:
        if kind == "general":
            if arity is None:
                raise ValueError("general emission requires arity.")
            source_count = arity
            coefficient_names = tuple(f"a{index}" for index in range(source_count))
            hoist_lines: list[str] = []
        else:
            if coefficients is None:
                raise ValueError("specialist emission requires coefficients.")
            hoist_lines = [
                f"    _a{index} = step * {stencil_scale * coefficient!r}"
                for index, coefficient in enumerate(coefficients)
            ]
            source_count = len(coefficients)
            coefficient_names = tuple(f"_a{index}" for index in range(source_count))

        flat_parameters = self._flat_parameters(kind=kind, source_count=source_count)
        flat_lines: list[str] = [f"def _kernel_flat({', '.join(flat_parameters)}):"]
        if hoist_lines:
            flat_lines.extend(hoist_lines)

        scalar_fields: list[Any] = []
        array_fields: list[Any] = []
        for layout_field in self.frame.fields:
            field_lines, is_scalar = self._field_lines(
                kind=kind,
                field=layout_field,
                source_count=source_count,
                coefficient_names=coefficient_names,
                fixed_coefficients=coefficients,
            )
            flat_lines.extend(field_lines)
            if is_scalar:
                scalar_fields.append(layout_field)
            else:
                array_fields.append(layout_field)

        if self._uses_functional_updates():
            returns = [
                self._target_name(kind, field)
                for field in array_fields
            ]
            returns.extend(
                f"_scalar_{self._target_name(kind, field)}"
                for field in scalar_fields
            )
            if returns:
                rendered_returns = ", ".join(returns)
                if len(returns) == 1:
                    rendered_returns += ","
                flat_lines.append(f"    return ({rendered_returns})")
            else:
                flat_lines.append("    return None")
        elif scalar_fields:
            returns = ", ".join(f"_scalar_{self._target_name(kind, field)}" for field in scalar_fields)
            if len(scalar_fields) == 1:
                returns = returns + ","
            flat_lines.append(f"    return ({returns})")
        else:
            flat_lines.append("    return None")

        wrapper_lines = self._wrapper_lines(
            kind=kind,
            source_count=source_count,
            scalar_fields=tuple(scalar_fields),
            array_fields=tuple(array_fields),
        )
        return "\n".join(flat_lines + [""] + wrapper_lines) + "\n"

    def _emit_unit_apply(self) -> str:
        source_count = 1
        flat_parameters = self._flat_parameters_unit_apply(source_count=source_count)
        flat_lines: list[str] = [f"def _kernel_flat({', '.join(flat_parameters)}):"]

        scalar_fields: list[Any] = []
        array_fields: list[Any] = []
        for layout_field in self.frame.fields:
            field_lines, is_scalar = self._field_lines(
                kind="update",
                field=layout_field,
                source_count=source_count,
                coefficient_names=("_a0",),
                fixed_coefficients=(1.0,),
                inline_fixed_coefficients=True,
            )
            flat_lines.extend(field_lines)
            if is_scalar:
                scalar_fields.append(layout_field)
            else:
                array_fields.append(layout_field)

        if self._uses_functional_updates():
            returns = [
                self._target_name("update", field)
                for field in array_fields
            ]
            returns.extend(
                f"_scalar_{self._target_name('update', field)}"
                for field in scalar_fields
            )
            if returns:
                rendered_returns = ", ".join(returns)
                if len(returns) == 1:
                    rendered_returns += ","
                flat_lines.append(f"    return ({rendered_returns})")
            else:
                flat_lines.append("    return None")
        elif scalar_fields:
            returns = ", ".join(f"_scalar_{self._target_name('update', field)}" for field in scalar_fields)
            if len(scalar_fields) == 1:
                returns = returns + ","
            flat_lines.append(f"    return ({returns})")
        else:
            flat_lines.append("    return None")

        wrapper_lines = self._wrapper_lines_unit_apply(
            source_count=source_count,
            scalar_fields=tuple(scalar_fields),
            array_fields=tuple(array_fields),
        )
        return "\n".join(flat_lines + [""] + wrapper_lines) + "\n"

    def _flat_parameters(self, *, kind: Kind, source_count: int) -> list[str]:
        parameters: list[str] = []
        if kind != "general":
            parameters.append("step")
        else:
            parameters.extend(f"a{index}" for index in range(source_count))

        for field in self.frame.fields:
            if kind == "update":
                parameters.append(self._origin_name(field))
            parameters.extend(f"x{index}_{field.translation_name}" for index in range(source_count))
            if self._policy_kind(field.policy) != "scalar":
                parameters.append(self._target_name(kind, field))
        return parameters

    def _flat_parameters_unit_apply(self, *, source_count: int) -> list[str]:
        parameters: list[str] = []
        for field in self.frame.fields:
            parameters.append(self._origin_name(field))
            parameters.extend(f"x{index}_{field.translation_name}" for index in range(source_count))
            if self._policy_kind(field.policy) != "scalar":
                parameters.append(self._target_name("update", field))
        return parameters

    def _wrapper_lines(
        self,
        *,
        kind: Kind,
        source_count: int,
        scalar_fields: tuple[Any, ...],
        array_fields: tuple[Any, ...],
    ) -> list[str]:
        if kind == "general":
            signature = self._general_wrapper_signature(source_count)
        elif kind == "delta":
            signature = self._delta_wrapper_signature(source_count)
        else:
            signature = self._update_wrapper_signature(source_count)

        lines = [f"def kernel({signature}):"]
        flat_args = self._flat_arguments(kind=kind, source_count=source_count)
        call = f"_kernel_flat({', '.join(flat_args)})"
        if self._uses_functional_updates():
            lines.append(f"    _updates = {call}")
            update_index = 0
            target_root = "result" if kind == "update" else "out"
            for field in array_fields:
                expression = (
                    field.state_expression(target_root)
                    if kind == "update"
                    else field.translation_expression(target_root)
                )
                lines.append(f"    {expression} = _updates[{update_index}]")
                update_index += 1
            for field in scalar_fields:
                expression = (
                    field.state_expression(target_root)
                    if kind == "update"
                    else field.translation_expression(target_root)
                )
                lines.append(f"    {expression} = _updates[{update_index}]")
                update_index += 1
        elif scalar_fields:
            lines.append(f"    _scalars = {call}")
            target_root = "result" if kind == "update" else "out"
            for index, field in enumerate(scalar_fields):
                expression = (
                    field.state_expression(target_root)
                    if kind == "update"
                    else field.translation_expression(target_root)
                )
                lines.append(f"    {expression} = _scalars[{index}]")
        else:
            lines.append(f"    {call}")

        if kind == "update":
            lines.append("    return result")
        else:
            lines.append("    return out")
        return lines

    def _wrapper_lines_unit_apply(
        self,
        *,
        source_count: int,
        scalar_fields: tuple[Any, ...],
        array_fields: tuple[Any, ...],
    ) -> list[str]:
        lines = [f"def kernel({self._update_wrapper_signature_unit_apply(source_count)}):"]
        flat_args = self._flat_arguments_unit_apply(source_count=source_count)
        call = f"_kernel_flat({', '.join(flat_args)})"
        if self._uses_functional_updates():
            lines.append(f"    _updates = {call}")
            update_index = 0
            for field in array_fields:
                lines.append(f"    {field.state_expression('result')} = _updates[{update_index}]")
                update_index += 1
            for field in scalar_fields:
                lines.append(f"    {field.state_expression('result')} = _updates[{update_index}]")
                update_index += 1
        elif scalar_fields:
            lines.append(f"    _scalars = {call}")
            for index, field in enumerate(scalar_fields):
                lines.append(f"    {field.state_expression('result')} = _scalars[{index}]")
        else:
            lines.append(f"    {call}")
        lines.append("    return result")
        return lines

    @staticmethod
    def _general_wrapper_signature(source_count: int) -> str:
        parameters: list[str] = []
        for index in range(source_count):
            parameters.append(f"a{index}")
            parameters.append(f"x{index}")
        parameters.append("out")
        return ", ".join(parameters)

    @staticmethod
    def _delta_wrapper_signature(source_count: int) -> str:
        return ", ".join(("step", *(f"x{index}" for index in range(source_count)), "out"))

    @staticmethod
    def _update_wrapper_signature(source_count: int) -> str:
        return ", ".join(("step", "origin", *(f"x{index}" for index in range(source_count)), "result"))

    @staticmethod
    def _update_wrapper_signature_unit_apply(source_count: int) -> str:
        return ", ".join(("origin", *(f"x{index}" for index in range(source_count)), "result"))

    def _flat_arguments(self, *, kind: Kind, source_count: int) -> list[str]:
        arguments: list[str] = []
        if kind != "general":
            arguments.append("step")
        else:
            arguments.extend(f"a{index}" for index in range(source_count))

        for field in self.frame.fields:
            if kind == "update":
                arguments.append(field.state_expression("origin"))
            for index in range(source_count):
                arguments.append(field.translation_expression(f"x{index}"))
            if self._policy_kind(field.policy) != "scalar":
                target_root = "result" if kind == "update" else "out"
                expression = (
                    field.state_expression(target_root)
                    if kind == "update"
                    else field.translation_expression(target_root)
                )
                arguments.append(expression)
        return arguments

    def _flat_arguments_unit_apply(self, *, source_count: int) -> list[str]:
        arguments: list[str] = []
        for field in self.frame.fields:
            arguments.append(field.state_expression("origin"))
            for index in range(source_count):
                arguments.append(field.translation_expression(f"x{index}"))
            if self._policy_kind(field.policy) != "scalar":
                arguments.append(field.state_expression("result"))
        return arguments

    def _field_lines(
        self,
        *,
        kind: Kind,
        field: Any,
        source_count: int,
        coefficient_names: tuple[str, ...],
        fixed_coefficients: tuple[float, ...] | None,
        inline_fixed_coefficients: bool = False,
    ) -> tuple[list[str], bool]:
        target_name = self._target_name(kind, field)
        origin_name = self._origin_name(field)
        source_names = tuple(f"x{index}_{field.translation_name}" for index in range(source_count))
        if kind == "general":
            expression = AlgebraistGeneratorEmitterExpression.from_runtime_coefficients(
                coefficients=coefficient_names,
                sources=source_names,
            ).source()
        else:
            if fixed_coefficients is None:
                raise ValueError("fixed coefficients are required for specialist emission.")
            expression = AlgebraistGeneratorEmitterExpression.from_fixed_coefficients(
                coefficients=fixed_coefficients,
                sources=source_names,
                inline_coefficients=inline_fixed_coefficients,
            ).source()
        if kind == "update":
            expression = f"{origin_name} + {expression}"

        policy = field.policy
        policy_kind = self._policy_kind(policy)
        if policy_kind == "scalar":
            return [f"    _scalar_{target_name} = {expression}"], True
        if policy_kind == "broadcast":
            if self._uses_functional_updates():
                return [f"    {target_name} = {expression}"], False
            return [f"    {target_name}[...] = {expression}"], False
        if policy_kind == "looped":
            return self._looped_lines(field=field, policy=policy, target=target_name, origin=origin_name, sources=source_names, coefficients=coefficient_names, fixed_coefficients=fixed_coefficients, inline_fixed_coefficients=inline_fixed_coefficients, kind=kind), False
        if policy_kind == "unravel":
            return self._unravel_lines(field=field, policy=policy, target=target_name, origin=origin_name, sources=source_names, coefficients=coefficient_names, fixed_coefficients=fixed_coefficients, inline_fixed_coefficients=inline_fixed_coefficients, kind=kind), False
        raise TypeError(f"Unsupported Algebraist frame policy: {policy!r}")

    @staticmethod
    def _target_name(kind: Kind, field: Any) -> str:
        prefix = "result" if kind == "update" else "out"
        name = field.state_name if kind == "update" else field.translation_name
        return f"{prefix}_{name}"

    @staticmethod
    def _origin_name(field: Any) -> str:
        return f"origin_{field.state_name}"

    def _looped_lines(
        self,
        *,
        field: Any,
        policy: object,
        target: str,
        origin: str,
        sources: tuple[str, ...],
        coefficients: tuple[str, ...],
        fixed_coefficients: tuple[float, ...] | None,
        inline_fixed_coefficients: bool = False,
        kind: Kind,
    ) -> list[str]:
        shape = getattr(policy, "shape", None)
        if shape is None:
            shape = getattr(field, "shape", None)
        if self._uses_vectorized_arrays() and shape is not None:
            if kind == "general":
                expression = AlgebraistGeneratorEmitterExpression.from_runtime_coefficients(
                    coefficients=coefficients,
                    sources=sources,
                ).source()
            else:
                if fixed_coefficients is None:
                    raise ValueError("fixed coefficients are required for specialist emission.")
                expression = AlgebraistGeneratorEmitterExpression.from_fixed_coefficients(
                    coefficients=fixed_coefficients,
                    sources=sources,
                    inline_coefficients=inline_fixed_coefficients,
                ).source()
            if kind == "update":
                expression = f"{origin} + {expression}"
            if self._uses_functional_updates():
                return [f"    {target} = {expression}"]
            return [f"    {target}[...] = {expression}"]

        rank = getattr(policy, "rank", None)
        if rank is None and shape is not None:
            rank = len(shape)
        if rank is None:
            raise ValueError("looped policy rank was not normalized.")
        index_names = tuple(f"i{index}" for index in range(rank))
        lines: list[str] = []
        for depth, index_name in enumerate(index_names):
            indent = "    " * (depth + 1)
            if shape is not None:
                bound = shape[depth]
            else:
                bound = f"{target}.shape[{depth}]"
            lines.append(f"{indent}for {index_name} in range({bound}):")
        assignment_indent = "    " * (rank + 1)
        index = self._index_expression(index_names)
        expression = self._expression_for_index(
            kind=kind,
            index=index,
            sources=sources,
            coefficients=coefficients,
            fixed_coefficients=fixed_coefficients,
            inline_fixed_coefficients=inline_fixed_coefficients,
        )
        if kind == "update":
            expression = f"{origin}{index} + {expression}"
        if self._uses_functional_updates():
            lines.append(f"{assignment_indent}{target} = {target}.at{index}.set({expression})")
        else:
            lines.append(f"{assignment_indent}{target}{index} = {expression}")
        return lines

    def _unravel_lines(
        self,
        *,
        field: Any,
        policy: object,
        target: str,
        origin: str,
        sources: tuple[str, ...],
        coefficients: tuple[str, ...],
        fixed_coefficients: tuple[float, ...] | None,
        inline_fixed_coefficients: bool = False,
        kind: Kind,
    ) -> list[str]:
        shape = getattr(policy, "shape", None)
        if shape is None:
            shape = getattr(field, "shape", None)
        if shape is None:
            raise ValueError("unravel policy requires shape.")
        lines: list[str] = []
        for index_tuple in product(*(range(dimension) for dimension in shape)):
            index = self._index_expression(tuple(str(index) for index in index_tuple))
            expression = self._expression_for_index(
                kind=kind,
                index=index,
                sources=sources,
                coefficients=coefficients,
                fixed_coefficients=fixed_coefficients,
                inline_fixed_coefficients=inline_fixed_coefficients,
            )
            if kind == "update":
                expression = f"{origin}{index} + {expression}"
            if self._uses_functional_updates():
                lines.append(f"    {target} = {target}.at{index}.set({expression})")
            else:
                lines.append(f"    {target}{index} = {expression}")
        return lines

    @staticmethod
    def _expression_for_index(
        *,
        kind: Kind,
        index: str,
        sources: tuple[str, ...],
        coefficients: tuple[str, ...],
        fixed_coefficients: tuple[float, ...] | None,
        inline_fixed_coefficients: bool = False,
    ) -> str:
        indexed_sources = tuple(f"{source}{index}" for source in sources)
        if kind == "general":
            return AlgebraistGeneratorEmitterExpression.from_runtime_coefficients(
                coefficients=coefficients,
                sources=indexed_sources,
            ).source()
        if fixed_coefficients is None:
            raise ValueError("fixed coefficients are required for specialist emission.")
        return AlgebraistGeneratorEmitterExpression.from_fixed_coefficients(
            coefficients=fixed_coefficients,
            sources=indexed_sources,
            inline_coefficients=inline_fixed_coefficients,
        ).source()

    @staticmethod
    def _index_expression(index_names: tuple[str, ...]) -> str:
        return "".join(f"[{index_name}]" for index_name in index_names)
