from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field as dataclass_field

from stark.algebraist.codegen import AlgebraistCodegen
from stark.algebraist.paths import path_expression


@dataclass(frozen=True, slots=True)
class AlgebraistImplicitCombination:
    role: str
    coefficients: tuple[float, ...]
    term_indices: tuple[int, ...] | None = None
    step_scale: bool = False

    def __post_init__(self) -> None:
        if not self.role or not self.role.isidentifier():
            raise ValueError(f"Invalid implicit Algebraist role {self.role!r}.")

        coefficients = tuple(self.coefficients)
        term_indices = (
            tuple(range(len(coefficients)))
            if self.term_indices is None
            else tuple(self.term_indices)
        )

        if len(coefficients) != len(term_indices):
            raise ValueError(
                "Implicit Algebraist coefficients and term indices must have "
                "matching lengths."
            )
        if any(index < 0 for index in term_indices):
            raise ValueError("Implicit Algebraist term indices must be non-negative.")

        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "term_indices", term_indices)

    @property
    def term_count(self) -> int:
        return len(self.term_indices)

    @classmethod
    def from_coefficients(
        cls,
        role: str,
        coefficients: Sequence[float],
        *,
        zero: float = 0.0,
    ) -> "AlgebraistImplicitCombination":
        kept_coefficients: list[float] = []
        term_indices: list[int] = []

        for index, coefficient in enumerate(coefficients):
            if _is_zero(coefficient, zero):
                continue

            kept_coefficients.append(coefficient)
            term_indices.append(index)

        return cls(role, tuple(kept_coefficients), tuple(term_indices))


@dataclass(frozen=True, slots=True)
class AlgebraistImplicitFixedSchemeBinding:
    known_shift_calls: tuple[Callable[..., object] | None, ...]
    final_delta_call: Callable[..., object] | None
    error_delta_call: Callable[..., object] | None
    known_shifts: tuple[AlgebraistImplicitCombination | None, ...]
    final_delta: AlgebraistImplicitCombination | None
    error_delta: AlgebraistImplicitCombination | None

    def require_known_shift_call(
        self,
        stage_index: int,
        scheme_name: str,
    ) -> Callable[..., object]:
        if stage_index < 0:
            raise ValueError(
                f"{scheme_name} requested invalid generated known-shift call "
                f"{stage_index}."
            )

        try:
            known_shift_call = self.known_shift_calls[stage_index]
        except IndexError as exc:
            raise ValueError(
                f"{scheme_name} requires generated known-shift call "
                f"{stage_index}, but only {len(self.known_shift_calls)} calls "
                "were bound."
            ) from exc

        if known_shift_call is None:
            raise ValueError(
                f"{scheme_name} requires generated known-shift call "
                f"{stage_index}, but that stage has no generated algebra."
            )

        return known_shift_call

    def require_final_delta_call(self, scheme_name: str) -> Callable[..., object]:
        final_delta_call = self.final_delta_call
        if final_delta_call is None:
            raise ValueError(f"{scheme_name} requires a generated final-delta call.")

        return final_delta_call

    def require_error_delta_call(self, scheme_name: str) -> Callable[..., object]:
        error_delta_call = self.error_delta_call
        if error_delta_call is None:
            raise ValueError(f"{scheme_name} requires a generated error-delta call.")

        return error_delta_call


@dataclass(frozen=True, slots=True)
class AlgebraistImplicitFixedSchemeBinder:
    algebraist: object
    codegen: AlgebraistCodegen = dataclass_field(default_factory=AlgebraistCodegen)

    def __call__(
        self,
        *,
        known_shifts: Sequence[AlgebraistImplicitCombination | None] = (),
        final_delta: AlgebraistImplicitCombination | None = None,
        error_delta: AlgebraistImplicitCombination | None = None,
    ) -> AlgebraistImplicitFixedSchemeBinding:
        known_shift_combinations = tuple(known_shifts)

        return AlgebraistImplicitFixedSchemeBinding(
            known_shift_calls=tuple(
                None if combination is None else self.optional_combination(combination)
                for combination in known_shift_combinations
            ),
            final_delta_call=(
                None if final_delta is None else self.optional_combination(final_delta)
            ),
            error_delta_call=(
                None if error_delta is None else self.optional_combination(error_delta)
            ),
            known_shifts=known_shift_combinations,
            final_delta=final_delta,
            error_delta=error_delta,
        )

    def optional_combination(
        self,
        combination: AlgebraistImplicitCombination,
    ) -> Callable[..., object] | None:
        if combination.term_count == 0:
            return None

        return self.combination(f"{combination.role}_combine", combination)

    def combination(
        self,
        name: str,
        combination: AlgebraistImplicitCombination,
    ) -> Callable[..., object]:
        kernel_name = f"{name}_kernel"
        kernel, _kernel_source = self.combination_kernel(kernel_name, combination)

        parameters: list[str] = []
        wrapper_arguments: list[str] = []

        if combination.step_scale:
            parameters.append("step")
            wrapper_arguments.append("step")

        for local_index, term_index in enumerate(combination.term_indices):
            del term_index
            parameters.append(f"k{local_index}")
            wrapper_arguments.extend(
                path_expression(f"k{local_index}", field.translation_path)
                for field in self.algebraist.fields
            )

        parameters.append("out")
        wrapper_arguments.extend(
            path_expression("out", field.translation_path)
            for field in self.algebraist.fields
        )

        wrapper_source = (
            f"def {name}({', '.join(parameters)}):\n"
            f" kernel({', '.join(wrapper_arguments)})\n"
            " return out\n"
        )

        return self.algebraist.compile_function(
            name,
            wrapper_source,
            namespace={"kernel": kernel},
            source_kind="wrapper",
        )

    def combination_kernel(
        self,
        name: str,
        combination: AlgebraistImplicitCombination,
    ) -> tuple[Callable[..., object], str]:
        field_arguments = [
            f"out_{field.translation_name}"
            for field in self.algebraist.fields
        ]

        term_arguments: list[str] = ["step"] if combination.step_scale else []
        for index in range(combination.term_count):
            term_arguments.extend(
                f"x{index}_{field.translation_name}"
                for field in self.algebraist.fields
            )

        body = "\n".join(
            (
                self.codegen.tableau_combine_assignment(
                    field,
                    combination.coefficients,
                    combination.term_count,
                )
                if combination.step_scale
                else self.codegen.weighted_combine_assignment(
                    field,
                    combination.coefficients,
                    combination.term_count,
                )
            )
            for field in self.algebraist.fields
        )
        source = (
            f"def {name}({', '.join(term_arguments + field_arguments)}):\n"
            f"{body}\n"
        )

        function = self.algebraist.compile_function(
            name,
            source,
            accelerate=True,
            source_kind="kernel",
        )
        return function, source


def _is_zero(value: float, zero: float) -> bool:
    if zero == 0.0:
        return value == 0.0

    return abs(value) <= zero


__all__ = [
    "AlgebraistImplicitCombination",
    "AlgebraistImplicitFixedSchemeBinder",
    "AlgebraistImplicitFixedSchemeBinding",
]
