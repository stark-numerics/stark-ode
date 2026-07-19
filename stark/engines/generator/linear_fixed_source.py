from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Literal

from stark.core.contracts.problem.field import FieldLike, FieldPolicyLike
from stark.core.contracts.problem.frame import FrameLike
from stark.engines.generator.elementwise import GeneratorElementwiseSource
from stark.engines.generator.expression import GeneratorExpression
from stark.engines.generator.policy import GeneratorPolicy, GeneratorPolicyLike
from stark.engines.generator.request import GeneratorRequestLinearFixedLike

LinearKernelKind = Literal["general", "delta", "update"]
LinearFixedKernelKind = Literal["delta", "update"]
LoopGroupKey = tuple[int, tuple[int, ...]]
LoopGroupItem = tuple[FieldLike[Any, Any], FieldPolicyLike]


@dataclass(frozen=True, slots=True)
class GeneratorLinearFixedSource:
    """Emit source strings for fixed-coefficient linear frame kernels."""

    frame: FrameLike
    policy: GeneratorPolicyLike = field(default_factory=GeneratorPolicy)

    def __call__(self, request: GeneratorRequestLinearFixedLike) -> str:
        coefficients = tuple(float(coefficient) for coefficient in request.coefficients)
        kind: LinearFixedKernelKind = "update" if request.apply else "delta"
        if self.uses_elementwise_backend():
            return GeneratorElementwiseSource(self.frame).linear_fixed(
                kind=kind,
                coefficients=coefficients,
                stencil_scale=float(request.scale),
            )
        return self.emit(
            kind=kind,
            coefficients=coefficients,
            stencil_scale=float(request.scale),
        )

    def unit_apply(self) -> str:
        if self.uses_elementwise_backend():
            return GeneratorElementwiseSource(self.frame).apply_translation()

        source_count = 1
        coefficient_names = ("_a0",)
        fixed_coefficients = (1.0,)
        flat_lines: list[str] = [
            f"def _kernel_flat({', '.join(self.flat_parameters_unit_apply(source_count=source_count))}):"
        ]

        scalar_fields: list[Any] = []
        array_fields: list[Any] = []
        looped_groups: dict[LoopGroupKey, list[LoopGroupItem]] = {}
        for field in self.frame.fields:
            policy = field.policy
            if self.can_group_looped_field(field, policy):
                key = self.loop_group_key(field, policy)
                looped_groups.setdefault(key, []).append((field, policy))
                array_fields.append(field)
                continue

            field_lines, is_scalar = self.field_lines(
                kind="update",
                field=field,
                source_count=source_count,
                coefficient_names=coefficient_names,
                fixed_coefficients=fixed_coefficients,
                inline_fixed_coefficients=True,
            )
            flat_lines.extend(field_lines)
            if is_scalar:
                scalar_fields.append(field)
            else:
                array_fields.append(field)

        for key, group in looped_groups.items():
            flat_lines.extend(
                self.looped_group_lines(
                    kind="update",
                    key=key,
                    group=tuple(group),
                    coefficients=coefficient_names,
                    fixed_coefficients=fixed_coefficients,
                    inline_fixed_coefficients=True,
                )
            )

        flat_lines.extend(self.return_lines("update", tuple(scalar_fields), tuple(array_fields)))
        wrapper_lines = self.wrapper_lines_unit_apply(
            source_count=source_count,
            scalar_fields=tuple(scalar_fields),
            array_fields=tuple(array_fields),
        )
        return "\n".join(flat_lines + [""] + wrapper_lines) + "\n"

    def emit(
        self,
        *,
        kind: LinearKernelKind,
        coefficients: tuple[float, ...] | None,
        stencil_scale: float = 1.0,
        arity: int | None = None,
    ) -> str:
        if kind == "general":
            if arity is None:
                raise ValueError("linear-combine source emission requires arity.")
            if self.uses_elementwise_backend():
                return GeneratorElementwiseSource(self.frame).linear_combine(arity=arity)
            source_count = arity
            coefficient_names = tuple(f"a{index}" for index in range(source_count))
            fixed_coefficients = None
            hoist_lines: list[str] = []
        else:
            if coefficients is None:
                raise ValueError("linear-fixed source emission requires coefficients.")
            source_count = len(coefficients)
            coefficient_names = tuple(f"_a{index}" for index in range(source_count))
            fixed_coefficients = coefficients
            hoist_lines = [
                f"    _a{index} = step * {stencil_scale * coefficient!r}"
                for index, coefficient in enumerate(coefficients)
            ]

        flat_lines: list[str] = [
            f"def _kernel_flat({', '.join(self.flat_parameters(kind=kind, source_count=source_count))}):"
        ]
        flat_lines.extend(hoist_lines)

        scalar_fields: list[Any] = []
        array_fields: list[Any] = []
        looped_groups: dict[LoopGroupKey, list[LoopGroupItem]] = {}
        for field in self.frame.fields:
            policy = field.policy
            if self.can_group_looped_field(field, policy):
                key = self.loop_group_key(field, policy)
                looped_groups.setdefault(key, []).append((field, policy))
                array_fields.append(field)
                continue

            field_lines, is_scalar = self.field_lines(
                kind=kind,
                field=field,
                source_count=source_count,
                coefficient_names=coefficient_names,
                fixed_coefficients=fixed_coefficients,
            )
            flat_lines.extend(field_lines)
            if is_scalar:
                scalar_fields.append(field)
            else:
                array_fields.append(field)

        for key, group in looped_groups.items():
            flat_lines.extend(
                self.looped_group_lines(
                    kind=kind,
                    key=key,
                    group=tuple(group),
                    coefficients=coefficient_names,
                    fixed_coefficients=fixed_coefficients,
                    inline_fixed_coefficients=False,
                )
            )

        flat_lines.extend(self.return_lines(kind, tuple(scalar_fields), tuple(array_fields)))
        wrapper_lines = self.wrapper_lines(
            kind=kind,
            source_count=source_count,
            scalar_fields=tuple(scalar_fields),
            array_fields=tuple(array_fields),
        )
        return "\n".join(flat_lines + [""] + wrapper_lines) + "\n"

    def return_lines(
        self,
        kind: LinearKernelKind,
        scalar_fields: tuple[Any, ...],
        array_fields: tuple[Any, ...],
    ) -> list[str]:
        if self.uses_functional_updates():
            returns = [self.target_name(kind, field) for field in array_fields]
            returns.extend(
                f"_scalar_{self.target_name(kind, field)}" for field in scalar_fields
            )
            if not returns:
                return ["    return None"]
            rendered = ", ".join(returns)
            if len(returns) == 1:
                rendered += ","
            return [f"    return ({rendered})"]

        if scalar_fields:
            returns = ", ".join(
                f"_scalar_{self.target_name(kind, field)}" for field in scalar_fields
            )
            if len(scalar_fields) == 1:
                returns += ","
            return [f"    return ({returns})"]

        return ["    return None"]

    def uses_functional_updates(self) -> bool:
        return self.policy.mutation == "functional"

    def uses_vectorized_arrays(self) -> bool:
        return self.policy.traversal in {"vectorized", "elementwise", "backend_kernel"} or (
            self.policy.expression in {"array_expression", "elementwise", "backend_kernel"}
        )

    def uses_elementwise_backend(self) -> bool:
        return self.policy.traversal == "elementwise" or self.policy.expression == "elementwise"

    @staticmethod
    def policy_kind(policy: FieldPolicyLike) -> str:
        return policy.kind

    def can_group_looped_field(
        self,
        field: FieldLike[Any, Any],
        policy: FieldPolicyLike,
    ) -> bool:
        del field
        return (
            policy.kind == "looped"
            and not self.uses_vectorized_arrays()
        )

    def loop_group_key(
        self,
        field: FieldLike[Any, Any],
        policy: FieldPolicyLike,
    ) -> LoopGroupKey:
        shape = self.field_shape(field, policy)
        return len(shape), shape

    @staticmethod
    def field_shape(
        field: FieldLike[Any, Any],
        policy: FieldPolicyLike,
    ) -> tuple[int, ...]:
        shape = field.shape
        if shape is None:
            raise ValueError(
                "Generated linear-fixed source needs a concrete shape for "
                f"field {field.state_name!r} with policy "
                f"{policy.kind!r}."
            )
        return tuple(shape)

    def flat_parameters(
        self,
        *,
        kind: LinearKernelKind,
        source_count: int,
    ) -> list[str]:
        parameters: list[str] = []
        if kind == "general":
            parameters.extend(f"a{index}" for index in range(source_count))
        else:
            parameters.append("step")
        for field in self.frame.fields:
            if kind == "update":
                parameters.append(self.origin_name(field))
            parameters.extend(
                f"x{index}_{field.translation_name}" for index in range(source_count)
            )
            if self.policy_kind(field.policy) != "scalar":
                parameters.append(self.target_name(kind, field))
        return parameters

    def flat_parameters_unit_apply(self, *, source_count: int) -> list[str]:
        parameters: list[str] = []
        for field in self.frame.fields:
            parameters.append(self.origin_name(field))
            parameters.extend(
                f"x{index}_{field.translation_name}" for index in range(source_count)
            )
            if self.policy_kind(field.policy) != "scalar":
                parameters.append(self.target_name("update", field))
        return parameters

    def wrapper_lines(
        self,
        *,
        kind: LinearKernelKind,
        source_count: int,
        scalar_fields: tuple[Any, ...],
        array_fields: tuple[Any, ...],
    ) -> list[str]:
        if kind == "general":
            signature = self.general_wrapper_signature(source_count)
        elif kind == "delta":
            signature = self.delta_wrapper_signature(source_count)
        else:
            signature = self.update_wrapper_signature(source_count)
        lines = [f"def kernel({signature}):"]
        call = f"_kernel_flat({', '.join(self.flat_arguments(kind=kind, source_count=source_count))})"
        if self.uses_functional_updates():
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

        lines.append("    return result" if kind == "update" else "    return out")
        return lines

    def wrapper_lines_unit_apply(
        self,
        *,
        source_count: int,
        scalar_fields: tuple[Any, ...],
        array_fields: tuple[Any, ...],
    ) -> list[str]:
        lines = [f"def kernel({self.unit_apply_wrapper_signature(source_count)}):"]
        call = (
            f"_kernel_flat({', '.join(self.flat_arguments_unit_apply(source_count=source_count))})"
        )
        if self.uses_functional_updates():
            lines.append(f"    _updates = {call}")
            update_index = 0
            for field in array_fields:
                lines.append(
                    f"    {field.state_expression('result')} = _updates[{update_index}]"
                )
                update_index += 1
            for field in scalar_fields:
                lines.append(
                    f"    {field.state_expression('result')} = _updates[{update_index}]"
                )
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
    def general_wrapper_signature(source_count: int) -> str:
        parameters: list[str] = []
        for index in range(source_count):
            parameters.append(f"a{index}")
            parameters.append(f"x{index}")
        parameters.append("out")
        return ", ".join(parameters)

    @staticmethod
    def delta_wrapper_signature(source_count: int) -> str:
        return ", ".join(("step", *(f"x{index}" for index in range(source_count)), "out"))

    @staticmethod
    def update_wrapper_signature(source_count: int) -> str:
        return ", ".join(
            ("step", "origin", *(f"x{index}" for index in range(source_count)), "result")
        )

    @staticmethod
    def unit_apply_wrapper_signature(source_count: int) -> str:
        return ", ".join(
            ("origin", *(f"x{index}" for index in range(source_count)), "result")
        )

    def flat_arguments(
        self,
        *,
        kind: LinearKernelKind,
        source_count: int,
    ) -> list[str]:
        arguments: list[str] = []
        if kind == "general":
            arguments.extend(f"a{index}" for index in range(source_count))
        else:
            arguments.append("step")
        for field in self.frame.fields:
            if kind == "update":
                arguments.append(field.state_expression("origin"))
            for index in range(source_count):
                arguments.append(field.translation_expression(f"x{index}"))
            if self.policy_kind(field.policy) != "scalar":
                target_root = "result" if kind == "update" else "out"
                expression = (
                    field.state_expression(target_root)
                    if kind == "update"
                    else field.translation_expression(target_root)
                )
                arguments.append(expression)
        return arguments

    def flat_arguments_unit_apply(self, *, source_count: int) -> list[str]:
        arguments: list[str] = []
        for field in self.frame.fields:
            arguments.append(field.state_expression("origin"))
            for index in range(source_count):
                arguments.append(field.translation_expression(f"x{index}"))
            if self.policy_kind(field.policy) != "scalar":
                arguments.append(field.state_expression("result"))
        return arguments

    def field_lines(
        self,
        *,
        kind: LinearKernelKind,
        field: Any,
        source_count: int,
        coefficient_names: tuple[str, ...],
        fixed_coefficients: tuple[float, ...] | None,
        inline_fixed_coefficients: bool = False,
    ) -> tuple[list[str], bool]:
        target = self.target_name(kind, field)
        origin = self.origin_name(field)
        sources = tuple(f"x{index}_{field.translation_name}" for index in range(source_count))
        expression = self.expression_for_sources(
            sources=sources,
            coefficients=coefficient_names,
            fixed_coefficients=fixed_coefficients,
            inline_coefficients=inline_fixed_coefficients,
        )
        if kind == "update":
            expression = f"{origin} + {expression}"

        policy = field.policy
        match self.policy_kind(policy):
            case "scalar":
                return [f"    _scalar_{target} = {expression}"], True
            case "broadcast":
                if self.uses_functional_updates():
                    return [f"    {target} = {expression}"], False
                return [f"    {target}[...] = {expression}"], False
            case "looped":
                return self.looped_lines(
                    field=field,
                    policy=policy,
                    target=target,
                    origin=origin,
                    sources=sources,
                    coefficients=coefficient_names,
                    fixed_coefficients=fixed_coefficients,
                    inline_fixed_coefficients=inline_fixed_coefficients,
                    kind=kind,
                ), False
            case "unravel":
                return self.unravel_lines(
                    field=field,
                    policy=policy,
                    target=target,
                    origin=origin,
                    sources=sources,
                    coefficients=coefficient_names,
                    fixed_coefficients=fixed_coefficients,
                    inline_fixed_coefficients=inline_fixed_coefficients,
                    kind=kind,
                ), False
            case _:
                raise ValueError(
                    "Generated linear-fixed source does not yet support "
                    f"field.policy.kind={self.policy_kind(policy)!r} "
                    f"for field {field.state_name!r}. "
                    "Supported generated policies today: "
                    "'broadcast', 'looped', 'scalar', 'unravel'."
                )

    @staticmethod
    def target_name(kind: LinearKernelKind, field: Any) -> str:
        prefix = "result" if kind == "update" else "out"
        name = field.state_name if kind == "update" else field.translation_name
        return f"{prefix}_{name}"

    @staticmethod
    def origin_name(field: Any) -> str:
        return f"origin_{field.state_name}"

    def looped_lines(
        self,
        *,
        field: Any,
        policy: FieldPolicyLike,
        target: str,
        origin: str,
        sources: tuple[str, ...],
        coefficients: tuple[str, ...],
        fixed_coefficients: tuple[float, ...] | None,
        inline_fixed_coefficients: bool,
        kind: LinearKernelKind,
    ) -> list[str]:
        shape = self.field_shape(field, policy)
        if self.uses_vectorized_arrays() and shape is not None:
            expression = self.expression_for_sources(
                sources=sources,
                coefficients=coefficients,
                fixed_coefficients=fixed_coefficients,
                inline_coefficients=inline_fixed_coefficients,
            )
            if kind == "update":
                expression = f"{origin} + {expression}"
            if self.uses_functional_updates():
                return [f"    {target} = {expression}"]
            return [f"    {target}[...] = {expression}"]

        rank = len(shape)
        index_names = tuple(f"i{index}" for index in range(rank))
        lines: list[str] = []
        for depth, index_name in enumerate(index_names):
            indent = "    " * (depth + 1)
            bound = shape[depth]
            lines.append(f"{indent}for {index_name} in range({bound}):")
        assignment_indent = "    " * (rank + 1)
        index = self.index_expression(index_names)
        expression = self.expression_for_index(
            index=index,
            sources=sources,
            coefficients=coefficients,
            fixed_coefficients=fixed_coefficients,
            inline_fixed_coefficients=inline_fixed_coefficients,
        )
        if kind == "update":
            expression = f"{origin}{index} + {expression}"
        if self.uses_functional_updates():
            lines.append(f"{assignment_indent}{target} = {target}.at{index}.set({expression})")
        else:
            lines.append(f"{assignment_indent}{target}{index} = {expression}")
        return lines

    def looped_group_lines(
        self,
        *,
        kind: LinearKernelKind,
        key: LoopGroupKey,
        group: tuple[LoopGroupItem, ...],
        coefficients: tuple[str, ...],
        fixed_coefficients: tuple[float, ...] | None,
        inline_fixed_coefficients: bool,
    ) -> list[str]:
        rank, shape = key
        index_names = tuple(f"i{index}" for index in range(rank))
        lines: list[str] = []
        for depth, (index_name, bound) in enumerate(zip(index_names, shape, strict=True)):
            indent = "    " * (depth + 1)
            lines.append(f"{indent}for {index_name} in range({bound}):")

        assignment_indent = "    " * (rank + 1)
        index = self.index_expression(index_names)
        for field, _policy in group:
            target = self.target_name(kind, field)
            origin = self.origin_name(field)
            sources = tuple(
                f"x{source_index}_{field.translation_name}"
                for source_index in range(len(coefficients))
            )
            expression = self.expression_for_index(
                index=index,
                sources=sources,
                coefficients=coefficients,
                fixed_coefficients=fixed_coefficients,
                inline_fixed_coefficients=inline_fixed_coefficients,
            )
            if kind == "update":
                expression = f"{origin}{index} + {expression}"
            if self.uses_functional_updates():
                lines.append(f"{assignment_indent}{target} = {target}.at{index}.set({expression})")
            else:
                lines.append(f"{assignment_indent}{target}{index} = {expression}")
        return lines

    def unravel_lines(
        self,
        *,
        field: Any,
        policy: FieldPolicyLike,
        target: str,
        origin: str,
        sources: tuple[str, ...],
        coefficients: tuple[str, ...],
        fixed_coefficients: tuple[float, ...] | None,
        inline_fixed_coefficients: bool,
        kind: LinearKernelKind,
    ) -> list[str]:
        shape = self.field_shape(field, policy)
        lines: list[str] = []
        for index_tuple in product(*(range(dimension) for dimension in shape)):
            index = self.index_expression(tuple(str(index) for index in index_tuple))
            expression = self.expression_for_index(
                index=index,
                sources=sources,
                coefficients=coefficients,
                fixed_coefficients=fixed_coefficients,
                inline_fixed_coefficients=inline_fixed_coefficients,
            )
            if kind == "update":
                expression = f"{origin}{index} + {expression}"
            if self.uses_functional_updates():
                lines.append(f"    {target} = {target}.at{index}.set({expression})")
            else:
                lines.append(f"    {target}{index} = {expression}")
        return lines

    @staticmethod
    def expression_for_index(
        *,
        index: str,
        sources: tuple[str, ...],
        coefficients: tuple[str, ...],
        fixed_coefficients: tuple[float, ...] | None,
        inline_fixed_coefficients: bool,
    ) -> str:
        indexed_sources = tuple(f"{source}{index}" for source in sources)
        return GeneratorLinearFixedSource.expression_for_sources(
            sources=indexed_sources,
            coefficients=coefficients,
            fixed_coefficients=fixed_coefficients,
            inline_coefficients=inline_fixed_coefficients,
        )

    @staticmethod
    def expression_for_sources(
        *,
        sources: tuple[str, ...],
        coefficients: tuple[str, ...],
        fixed_coefficients: tuple[float, ...] | None,
        inline_coefficients: bool,
    ) -> str:
        if fixed_coefficients is None:
            return GeneratorExpression.from_runtime_coefficients(
                coefficients=coefficients,
                sources=sources,
            ).source()
        return GeneratorExpression.from_fixed_coefficients(
            coefficients=fixed_coefficients,
            sources=sources,
            inline_coefficients=inline_coefficients,
        ).source()

    @staticmethod
    def index_expression(index_names: tuple[str, ...]) -> str:
        return "".join(f"[{index_name}]" for index_name in index_names)


__all__ = ["GeneratorLinearFixedSource", "LinearFixedKernelKind", "LinearKernelKind"]
