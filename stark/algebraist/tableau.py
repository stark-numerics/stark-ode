from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field as dataclass_field
from typing import Protocol

from stark.algebraist.codegen import AlgebraistCodegen
from stark.algebraist.paths import path_expression


class ButcherTableauLike(Protocol):
    c: tuple[float, ...]
    a: tuple[tuple[float, ...], ...]
    b: tuple[float, ...]
    order: int
    b_embedded: tuple[float, ...] | None
    embedded_order: int | None
    short_name: str | None
    full_name: str | None


@dataclass(frozen=True, slots=True)
class AlgebraistTableauCombination:
    role: str
    coefficients: tuple[float, ...]
    term_indices: tuple[int, ...]

    @property
    def term_count(self) -> int:
        return len(self.term_indices)


@dataclass(frozen=True, slots=True)
class AlgebraistTableau:
    stages: tuple[AlgebraistTableauCombination, ...]
    solution: AlgebraistTableauCombination
    error: AlgebraistTableauCombination | None
    order: int
    embedded_order: int | None = None
    short_name: str | None = None
    full_name: str | None = None

    @property
    def has_error(self) -> bool:
        return self.error is not None


@dataclass(frozen=True, slots=True)
class AlgebraistTableauBinding:
    stages: tuple[Callable[..., object], ...]
    solution: Callable[..., object]
    error: Callable[..., object] | None
    tableau: AlgebraistTableau


@dataclass(frozen=True, slots=True)
class AlgebraistTableauPlanner:
    zero: float = 0.0

    def __call__(self, tableau: ButcherTableauLike) -> AlgebraistTableau:
        self.validate_tableau(tableau)

        stages = tuple(
            self.combination(f"stage{index}", row)
            for index, row in enumerate(tableau.a)
        )
        solution = self.combination("solution", tableau.b)

        error = None
        if tableau.b_embedded is not None:
            error_coefficients = tuple(
                high - low
                for high, low in zip(tableau.b, tableau.b_embedded, strict=True)
            )
            error = self.combination("error", error_coefficients)

        return AlgebraistTableau(
            stages=stages,
            solution=solution,
            error=error,
            order=tableau.order,
            embedded_order=tableau.embedded_order,
            short_name=tableau.short_name,
            full_name=tableau.full_name,
        )

    def combination(
        self,
        role: str,
        coefficients: Sequence[float],
    ) -> AlgebraistTableauCombination:
        kept_coefficients: list[float] = []
        term_indices: list[int] = []

        for index, coefficient in enumerate(coefficients):
            if self.is_zero(coefficient):
                continue

            kept_coefficients.append(coefficient)
            term_indices.append(index)

        return AlgebraistTableauCombination(
            role=role,
            coefficients=tuple(kept_coefficients),
            term_indices=tuple(term_indices),
        )

    def is_zero(self, value: float) -> bool:
        if self.zero == 0.0:
            return value == 0.0

        return abs(value) <= self.zero

    @staticmethod
    def validate_tableau(tableau: ButcherTableauLike) -> None:
        stage_count = len(tableau.c)

        if len(tableau.a) != stage_count:
            raise ValueError("Butcher tableau stage rows must match c.")
        if len(tableau.b) != stage_count:
            raise ValueError("Butcher tableau solution weights must match c.")
        if tableau.b_embedded is not None and len(tableau.b_embedded) != stage_count:
            raise ValueError("Butcher tableau embedded weights must match c.")


@dataclass(frozen=True, slots=True)
class AlgebraistTableauBinder:
    algebraist: object
    planner: AlgebraistTableauPlanner = AlgebraistTableauPlanner()
    codegen: AlgebraistCodegen = dataclass_field(default_factory=AlgebraistCodegen)

    def __call__(self, tableau: ButcherTableauLike) -> AlgebraistTableauBinding:
        algebraist_tableau = self.planner(tableau)

        return AlgebraistTableauBinding(
            stages=tuple(
                self.combination(f"{combination.role}_combine", combination)
                for combination in algebraist_tableau.stages
            ),
            solution=self.combination(
                "solution_combine",
                algebraist_tableau.solution,
            ),
            error=(
                None
                if algebraist_tableau.error is None
                else self.combination("error_combine", algebraist_tableau.error)
            ),
            tableau=algebraist_tableau,
        )

    def combination(
        self,
        name: str,
        combination: AlgebraistTableauCombination,
    ) -> Callable[..., object]:
        if combination.term_count == 0:
            source = (
                f"def {name}(out, step):\n"
                " raise ValueError("
                "'Cannot call an empty Algebraist tableau combination.'"
                ")\n"
            )
            return self.algebraist.compile_function(name, source)

        kernel_name = f"{name}_kernel"
        kernel, _kernel_source = self.combination_kernel(kernel_name, combination)

        parameters = ["out", "step"]
        wrapper_arguments = [
            path_expression("out", field.translation_path)
            for field in self.algebraist.fields
        ]
        wrapper_arguments.append("step")

        for local_index, term_index in enumerate(combination.term_indices):
            del term_index
            parameters.append(f"k{local_index}")
            wrapper_arguments.extend(
                path_expression(f"k{local_index}", field.translation_path)
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
        )

    def combination_kernel(
        self,
        name: str,
        combination: AlgebraistTableauCombination,
    ) -> tuple[Callable[..., object], str]:
        field_arguments = [
            f"out_{field.translation_name}"
            for field in self.algebraist.fields
        ]

        term_arguments: list[str] = ["step"]
        for index in range(combination.term_count):
            term_arguments.extend(
                f"x{index}_{field.translation_name}"
                for field in self.algebraist.fields
            )

        body = "\n".join(
            self.codegen.tableau_combine_assignment(
                field,
                combination.coefficients,
                combination.term_count,
            )
            for field in self.algebraist.fields
        )
        source = (
            f"def {name}({', '.join(field_arguments + term_arguments)}):\n"
            f"{body}\n"
        )

        function = self.algebraist.compile_function(
            name,
            source,
            accelerate=True,
        )
        return function, source


__all__ = [
    "AlgebraistTableauBinder",
    "AlgebraistTableauBinding",
    "AlgebraistTableau",
    "AlgebraistTableauCombination",
    "AlgebraistTableauPlanner",
    "ButcherTableauLike",
]