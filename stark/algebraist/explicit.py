from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field as dataclass_field

from stark.algebraist.codegen import AlgebraistCodegen
from stark.algebraist.paths import path_expression
from stark.algebraist.tableau import (
    AlgebraistTableauBinder,
    AlgebraistTableauBinding,
    AlgebraistTableauCombination,
    ButcherTableauLike,
)


@dataclass(frozen=True, slots=True)
class AlgebraistExplicitSchemeBinding:
    stage_state_calls: tuple[Callable[..., object], ...]
    solution_state_call: Callable[..., object]
    solution_delta_call: Callable[..., object]
    error_delta_call: Callable[..., object] | None
    tableau_binding: AlgebraistTableauBinding


@dataclass(frozen=True, slots=True)
class AlgebraistExplicitSchemeBinder:
    algebraist: object
    tableau_binder: AlgebraistTableauBinder | None = None
    codegen: AlgebraistCodegen = dataclass_field(default_factory=AlgebraistCodegen)

    def __call__(
        self,
        tableau: ButcherTableauLike,
    ) -> AlgebraistExplicitSchemeBinding:
        tableau_binder = self.tableau_binder
        if tableau_binder is None:
            tableau_binder = AlgebraistTableauBinder(self.algebraist)

        tableau_binding = tableau_binder(tableau)

        return AlgebraistExplicitSchemeBinding(
            stage_state_calls=tuple(
                self.stage(f"{combination.role}_state", combination)
                for combination in tableau_binding.tableau.stages
            ),
            solution_state_call=self.stage("solution_state", tableau_binding.tableau.solution),
            solution_delta_call=tableau_binding.solution,
            error_delta_call=tableau_binding.error,
            tableau_binding=tableau_binding,
        )

    def stage(
        self,
        name: str,
        combination: AlgebraistTableauCombination,
    ) -> Callable[..., object]:
        if combination.term_count == 0:
            source = (
                f"def {name}(result, origin, step):\n"
                " raise ValueError('Cannot call an empty Algebraist explicit stage.')\n"
            )
            return self.algebraist.compile_function(name, source, source_kind="wrapper")

        kernel_name = f"{name}_kernel"
        kernel, _kernel_source = self.stage_kernel(kernel_name, combination)

        parameters = ["result", "origin", "step"]
        wrapper_arguments = [
            path_expression("result", field.state_path)
            for field in self.algebraist.fields
        ]
        wrapper_arguments.extend(
            path_expression("origin", field.state_path)
            for field in self.algebraist.fields
        )
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
            " return result\n"
        )

        return self.algebraist.compile_function(
            name,
            wrapper_source,
            namespace={"kernel": kernel},
            source_kind="wrapper",
        )

    def stage_kernel(
        self,
        name: str,
        combination: AlgebraistTableauCombination,
    ) -> tuple[Callable[..., object], str]:
        result_arguments = [
            f"result_{field.state_name}"
            for field in self.algebraist.fields
        ]
        origin_arguments = [
            f"origin_{field.state_name}"
            for field in self.algebraist.fields
        ]

        term_arguments: list[str] = ["step"]
        for index in range(combination.term_count):
            term_arguments.extend(
                f"x{index}_{field.translation_name}"
                for field in self.algebraist.fields
            )

        body = "\n".join(
            self.codegen.tableau_stage_assignment(
                field,
                combination.coefficients,
                combination.term_count,
            )
            for field in self.algebraist.fields
        )
        source = (
            f"def {name}("
            f"{', '.join(result_arguments + origin_arguments + term_arguments)}"
            "):\n"
            f"{body}\n"
        )

        function = self.algebraist.compile_function(
            name,
            source,
            accelerate=True,
            source_kind="kernel",
        )
        return function, source


__all__ = [
    "AlgebraistExplicitSchemeBinder",
    "AlgebraistExplicitSchemeBinding",
]