from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field as dataclass_field

from stark.algebraist.classic.codegen import AlgebraistCodegen
from stark.algebraist.classic.paths import path_expression


@dataclass(frozen=True, slots=True)
class AlgebraistImExCombination:
    role: str
    explicit_coefficients: tuple[float, ...]
    implicit_coefficients: tuple[float, ...]
    explicit_indices: tuple[int, ...] | None = None
    implicit_indices: tuple[int, ...] | None = None
    step_scale: bool = True

    def __post_init__(self) -> None:
        if not self.role or not self.role.isidentifier():
            raise ValueError(f"Invalid IMEX Algebraist role {self.role!r}.")

        explicit_coefficients = tuple(self.explicit_coefficients)
        implicit_coefficients = tuple(self.implicit_coefficients)
        explicit_indices = (
            tuple(range(len(explicit_coefficients)))
            if self.explicit_indices is None
            else tuple(self.explicit_indices)
        )
        implicit_indices = (
            tuple(range(len(implicit_coefficients)))
            if self.implicit_indices is None
            else tuple(self.implicit_indices)
        )

        if len(explicit_coefficients) != len(explicit_indices):
            raise ValueError(
                "IMEX Algebraist explicit coefficients and indices must have "
                "matching lengths."
            )
        if len(implicit_coefficients) != len(implicit_indices):
            raise ValueError(
                "IMEX Algebraist implicit coefficients and indices must have "
                "matching lengths."
            )
        if any(index < 0 for index in explicit_indices + implicit_indices):
            raise ValueError("IMEX Algebraist term indices must be non-negative.")

        object.__setattr__(self, "explicit_coefficients", explicit_coefficients)
        object.__setattr__(self, "implicit_coefficients", implicit_coefficients)
        object.__setattr__(self, "explicit_indices", explicit_indices)
        object.__setattr__(self, "implicit_indices", implicit_indices)

    @property
    def coefficients(self) -> tuple[float, ...]:
        return self.explicit_coefficients + self.implicit_coefficients

    @property
    def term_count(self) -> int:
        return len(self.explicit_coefficients) + len(self.implicit_coefficients)

    @classmethod
    def from_coefficients(
        cls,
        role: str,
        *,
        explicit: Sequence[float],
        implicit: Sequence[float],
        zero: float = 0.0,
    ) -> "AlgebraistImExCombination":
        explicit_coefficients: list[float] = []
        explicit_indices: list[int] = []
        implicit_coefficients: list[float] = []
        implicit_indices: list[int] = []

        for index, coefficient in enumerate(explicit):
            if _is_zero(coefficient, zero):
                continue
            explicit_coefficients.append(coefficient)
            explicit_indices.append(index)

        for index, coefficient in enumerate(implicit):
            if _is_zero(coefficient, zero):
                continue
            implicit_coefficients.append(coefficient)
            implicit_indices.append(index)

        return cls(
            role,
            tuple(explicit_coefficients),
            tuple(implicit_coefficients),
            tuple(explicit_indices),
            tuple(implicit_indices),
        )


@dataclass(frozen=True, slots=True)
class AlgebraistImExAdaptiveSchemeBinding:
    stage_shift_calls: tuple[Callable[..., object] | None, ...]
    high_delta_call: Callable[..., object] | None
    low_delta_call: Callable[..., object] | None
    error_delta_call: Callable[..., object] | None
    stage_shifts: tuple[AlgebraistImExCombination | None, ...]
    high_delta: AlgebraistImExCombination | None
    low_delta: AlgebraistImExCombination | None
    error_delta: AlgebraistImExCombination | None

    def require_stage_shift_call(
        self,
        stage_index: int,
        scheme_name: str,
    ) -> Callable[..., object]:
        if stage_index < 0:
            raise ValueError(
                f"{scheme_name} requested invalid generated IMEX stage-shift "
                f"call {stage_index}."
            )

        try:
            stage_shift_call = self.stage_shift_calls[stage_index]
        except IndexError as exc:
            raise ValueError(
                f"{scheme_name} requires generated IMEX stage-shift call "
                f"{stage_index}, but only {len(self.stage_shift_calls)} calls "
                "were bound."
            ) from exc

        if stage_shift_call is None:
            raise ValueError(
                f"{scheme_name} requires generated IMEX stage-shift call "
                f"{stage_index}, but that stage has no generated algebra."
            )

        return stage_shift_call

    def require_high_delta_call(self, scheme_name: str) -> Callable[..., object]:
        high_delta_call = self.high_delta_call
        if high_delta_call is None:
            raise ValueError(f"{scheme_name} requires a generated IMEX high-delta call.")

        return high_delta_call

    def require_low_delta_call(self, scheme_name: str) -> Callable[..., object]:
        low_delta_call = self.low_delta_call
        if low_delta_call is None:
            raise ValueError(f"{scheme_name} requires a generated IMEX low-delta call.")

        return low_delta_call

    def require_error_delta_call(self, scheme_name: str) -> Callable[..., object]:
        error_delta_call = self.error_delta_call
        if error_delta_call is None:
            raise ValueError(f"{scheme_name} requires a generated IMEX error-delta call.")

        return error_delta_call


@dataclass(frozen=True, slots=True)
class AlgebraistImExAdaptiveSchemeBinder:
    algebraist: object
    codegen: AlgebraistCodegen = dataclass_field(default_factory=AlgebraistCodegen)

    def __call__(
        self,
        *,
        stage_shifts: Sequence[AlgebraistImExCombination | None] = (),
        high_delta: AlgebraistImExCombination | None = None,
        low_delta: AlgebraistImExCombination | None = None,
        error_delta: AlgebraistImExCombination | None = None,
    ) -> AlgebraistImExAdaptiveSchemeBinding:
        stage_shift_combinations = tuple(stage_shifts)

        return AlgebraistImExAdaptiveSchemeBinding(
            stage_shift_calls=tuple(
                None if combination is None else self.optional_combination(combination)
                for combination in stage_shift_combinations
            ),
            high_delta_call=(
                None if high_delta is None else self.optional_combination(high_delta)
            ),
            low_delta_call=(
                None if low_delta is None else self.optional_combination(low_delta)
            ),
            error_delta_call=(
                None if error_delta is None else self.optional_combination(error_delta)
            ),
            stage_shifts=stage_shift_combinations,
            high_delta=high_delta,
            low_delta=low_delta,
            error_delta=error_delta,
        )

    def optional_combination(
        self,
        combination: AlgebraistImExCombination,
    ) -> Callable[..., object] | None:
        if combination.term_count == 0:
            return None

        return self.combination(f"{combination.role}_combine", combination)

    def combination(
        self,
        name: str,
        combination: AlgebraistImExCombination,
    ) -> Callable[..., object]:
        kernel_name = f"{name}_kernel"
        kernel, _kernel_source = self.combination_kernel(kernel_name, combination)

        parameters: list[str] = []
        wrapper_arguments: list[str] = []

        if combination.step_scale:
            parameters.append("step")
            wrapper_arguments.append("step")

        for explicit_index in combination.explicit_indices:
            parameters.append(f"explicit_k{explicit_index}")
            wrapper_arguments.extend(
                path_expression(f"explicit_k{explicit_index}", field.translation_path)
                for field in self.algebraist.fields
            )

        for implicit_index in combination.implicit_indices:
            parameters.append(f"implicit_k{implicit_index}")
            wrapper_arguments.extend(
                path_expression(f"implicit_k{implicit_index}", field.translation_path)
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
        combination: AlgebraistImExCombination,
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
    "AlgebraistImExAdaptiveSchemeBinder",
    "AlgebraistImExAdaptiveSchemeBinding",
    "AlgebraistImExCombination",
]
