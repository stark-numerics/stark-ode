from __future__ import annotations

from collections.abc import Sequence
from math import prod

from stark.algebraist.fields import AlgebraistField
from stark.algebraist.policies import (
    AlgebraistBroadcast,
    AlgebraistLooped,
    AlgebraistSmallFixed,
)


class AlgebraistCodegen:
    """Generate source-body fragments for Algebraist field policies.

    The class owns field-policy dispatch for generated Algebraist kernels.
    It emits source fragments only; function assembly and compilation remain
    in the Algebraist core and binder layers.
    """

    def combine_assignment(self, field: AlgebraistField, term_count: int) -> str:
        policy = field.policy

        if isinstance(policy, AlgebraistSmallFixed):
            return self.small_fixed_combine_assignment(
                field,
                policy.shape,
                term_count,
            )

        if isinstance(policy, AlgebraistLooped):
            return self.looped_combine_assignment(
                field,
                self.looped_rank(field, policy),
                term_count,
            )

        if isinstance(policy, AlgebraistBroadcast):
            return self.broadcast_combine_assignment(field, term_count)

        raise TypeError(f"Unknown Algebraist policy {policy!r}.")

    def tableau_combine_assignment(
        self,
        field: AlgebraistField,
        coefficients: Sequence[float],
        term_count: int,
    ) -> str:
        policy = field.policy

        if isinstance(policy, AlgebraistSmallFixed):
            return self.small_fixed_tableau_combine_assignment(
                field,
                policy.shape,
                coefficients,
                term_count,
            )

        if isinstance(policy, AlgebraistLooped):
            return self.looped_tableau_combine_assignment(
                field,
                self.looped_rank(field, policy),
                coefficients,
                term_count,
            )

        if isinstance(policy, AlgebraistBroadcast):
            return self.broadcast_tableau_combine_assignment(
                field,
                coefficients,
                term_count,
            )

        raise TypeError(f"Unknown Algebraist policy {policy!r}.")

    def tableau_stage_assignment(
        self,
        field: AlgebraistField,
        coefficients: Sequence[float],
        term_count: int,
    ) -> str:
        policy = field.policy

        if isinstance(policy, AlgebraistSmallFixed):
            return self.small_fixed_tableau_stage_assignment(
                field,
                policy.shape,
                coefficients,
                term_count,
            )

        if isinstance(policy, AlgebraistLooped):
            return self.looped_tableau_stage_assignment(
                field,
                self.looped_rank(field, policy),
                coefficients,
                term_count,
            )

        if isinstance(policy, AlgebraistBroadcast):
            return self.broadcast_tableau_stage_assignment(
                field,
                coefficients,
                term_count,
            )

        raise TypeError(f"Unknown Algebraist policy {policy!r}.")

    def apply_assignment(self, field: AlgebraistField) -> str:
        policy = field.policy

        if isinstance(policy, AlgebraistSmallFixed):
            return self.small_fixed_apply_assignment(field, policy.shape)

        if isinstance(policy, AlgebraistLooped):
            return self.looped_apply_assignment(
                field,
                self.looped_rank(field, policy),
            )

        if isinstance(policy, AlgebraistBroadcast):
            return self.broadcast_apply_assignment(field)

        raise TypeError(f"Unknown Algebraist policy {policy!r}.")

    def norm_body(self, fields: Sequence[AlgebraistField], kind: str) -> list[str]:
        lines = [" total = 0.0"]
        if kind == "rms":
            lines.append(" count = 0")

        for field in fields:
            if not field.include_in_norm:
                continue

            policy = field.policy
            if isinstance(policy, AlgebraistSmallFixed):
                lines.extend(self.small_fixed_norm_body(field, policy.shape, kind))
            elif isinstance(policy, AlgebraistLooped):
                lines.extend(
                    self.looped_norm_body(
                        field,
                        self.looped_rank(field, policy),
                        kind,
                    )
                )
            elif isinstance(policy, AlgebraistBroadcast):
                lines.extend(self.broadcast_norm_body(field, kind))
            else:
                raise TypeError(f"Unknown Algebraist policy {policy!r}.")

        if kind == "rms":
            lines.append(" return 0.0 if count == 0 else (total / count) ** 0.5")
        else:
            lines.append(" return total ** 0.5")

        return lines

    @staticmethod
    def broadcast_combine_assignment(
        field: AlgebraistField,
        term_count: int,
    ) -> str:
        name = field.translation_name
        value = " + ".join(
            f"a{index} * x{index}_{name}" for index in range(term_count)
        )
        return f" out_{name}[...] = {value}"

    @staticmethod
    def broadcast_tableau_combine_assignment(
        field: AlgebraistField,
        coefficients: Sequence[float],
        term_count: int,
    ) -> str:
        name = field.translation_name
        value = " + ".join(
            f"(step * {coefficients[index]!r}) * x{index}_{name}"
            for index in range(term_count)
        )
        return f" out_{name}[...] = {value}"

    @staticmethod
    def broadcast_tableau_stage_assignment(
        field: AlgebraistField,
        coefficients: Sequence[float],
        term_count: int,
    ) -> str:
        translation = field.translation_name
        state = field.state_name
        value = " + ".join(
            f"(step * {coefficients[index]!r}) * x{index}_{translation}"
            for index in range(term_count)
        )
        return f" result_{state}[...] = origin_{state} + ({value})"

    @staticmethod
    def broadcast_apply_assignment(field: AlgebraistField) -> str:
        delta = field.translation_name
        target = field.state_name
        return f" result_{target}[...] = origin_{target} + delta_{delta}"

    @staticmethod
    def broadcast_norm_body(field: AlgebraistField, kind: str) -> list[str]:
        name = field.translation_name
        lines = [
            f" for value in {name}.ravel():",
            "  total += value * value",
        ]

        if kind == "rms":
            lines.append(f" count += {name}.size")

        return lines

    def looped_combine_assignment(
        self,
        field: AlgebraistField,
        rank: int,
        term_count: int,
    ) -> str:
        name = field.translation_name
        index_names = self.index_names(rank)
        shape_terms = self.shape_terms(rank)

        lines = [self.shape_binding(f"out_{name}", shape_terms)]
        indent = " "

        for index_name, shape_name in zip(index_names, shape_terms, strict=True):
            lines.append(f"{indent}for {index_name} in range({shape_name}):")
            indent += " "

        location = ", ".join(index_names)
        value = " + ".join(
            f"a{index} * x{index}_{name}[{location}]"
            for index in range(term_count)
        )
        lines.append(f"{indent}out_{name}[{location}] = {value}")

        return "\n".join(lines)

    def looped_tableau_combine_assignment(
        self,
        field: AlgebraistField,
        rank: int,
        coefficients: Sequence[float],
        term_count: int,
    ) -> str:
        name = field.translation_name
        index_names = self.index_names(rank)
        shape_terms = self.shape_terms(rank)

        lines = [self.shape_binding(f"out_{name}", shape_terms)]
        indent = " "

        for index_name, shape_name in zip(index_names, shape_terms, strict=True):
            lines.append(f"{indent}for {index_name} in range({shape_name}):")
            indent += " "

        location = ", ".join(index_names)
        value = " + ".join(
            f"(step * {coefficients[index]!r}) * x{index}_{name}[{location}]"
            for index in range(term_count)
        )
        lines.append(f"{indent}out_{name}[{location}] = {value}")

        return "\n".join(lines)

    def looped_tableau_stage_assignment(
        self,
        field: AlgebraistField,
        rank: int,
        coefficients: Sequence[float],
        term_count: int,
    ) -> str:
        translation = field.translation_name
        state = field.state_name
        index_names = self.index_names(rank)
        shape_terms = self.shape_terms(rank)

        lines = [self.shape_binding(f"result_{state}", shape_terms)]
        indent = " "

        for index_name, shape_name in zip(index_names, shape_terms, strict=True):
            lines.append(f"{indent}for {index_name} in range({shape_name}):")
            indent += " "

        location = ", ".join(index_names)
        value = " + ".join(
            f"(step * {coefficients[index]!r}) * x{index}_{translation}[{location}]"
            for index in range(term_count)
        )
        lines.append(
            f"{indent}result_{state}[{location}] = "
            f"origin_{state}[{location}] + ({value})"
        )

        return "\n".join(lines)

    def looped_apply_assignment(self, field: AlgebraistField, rank: int) -> str:
        delta = field.translation_name
        target = field.state_name
        index_names = self.index_names(rank)
        shape_terms = self.shape_terms(rank)

        lines = [self.shape_binding(f"delta_{delta}", shape_terms)]
        indent = " "

        for index_name, shape_name in zip(index_names, shape_terms, strict=True):
            lines.append(f"{indent}for {index_name} in range({shape_name}):")
            indent += " "

        location = ", ".join(index_names)
        lines.append(
            f"{indent}result_{target}[{location}] = "
            f"origin_{target}[{location}] + delta_{delta}[{location}]"
        )

        return "\n".join(lines)

    def looped_norm_body(
        self,
        field: AlgebraistField,
        rank: int,
        kind: str,
    ) -> list[str]:
        name = field.translation_name
        index_names = self.index_names(rank)
        shape_terms = self.shape_terms(rank)

        lines = [self.shape_binding(name, shape_terms)]
        indent = " "

        for index_name, shape_name in zip(index_names, shape_terms, strict=True):
            lines.append(f"{indent}for {index_name} in range({shape_name}):")
            indent += " "

        location = ", ".join(index_names)
        lines.append(f"{indent}value = {name}[{location}]")
        lines.append(f"{indent}total += value * value")

        if kind == "rms":
            lines.append(f" count += {name}.size")

        return lines

    def small_fixed_combine_assignment(
        self,
        field: AlgebraistField,
        shape: tuple[int, ...],
        term_count: int,
    ) -> str:
        name = field.translation_name
        lines: list[str] = []

        for location in self.fixed_locations(shape):
            value = " + ".join(
                f"a{index} * x{index}_{name}[{location}]"
                for index in range(term_count)
            )
            lines.append(f" out_{name}[{location}] = {value}")

        return "\n".join(lines)

    def small_fixed_tableau_stage_assignment(
        self,
        field: AlgebraistField,
        shape: tuple[int, ...],
        coefficients: Sequence[float],
        term_count: int,
    ) -> str:
        translation = field.translation_name
        state = field.state_name
        lines: list[str] = []

        for location in self.fixed_locations(shape):
            value = " + ".join(
                f"(step * {coefficients[index]!r}) * "
                f"x{index}_{translation}[{location}]"
                for index in range(term_count)
            )
            lines.append(
                f" result_{state}[{location}] = "
                f"origin_{state}[{location}] + ({value})"
            )

        return "\n".join(lines)

    def small_fixed_tableau_combine_assignment(
        self,
        field: AlgebraistField,
        shape: tuple[int, ...],
        coefficients: Sequence[float],
        term_count: int,
    ) -> str:
        name = field.translation_name
        lines: list[str] = []

        for location in self.fixed_locations(shape):
            value = " + ".join(
                f"(step * {coefficients[index]!r}) * x{index}_{name}[{location}]"
                for index in range(term_count)
            )
            lines.append(f" out_{name}[{location}] = {value}")

        return "\n".join(lines)

    def small_fixed_apply_assignment(
        self,
        field: AlgebraistField,
        shape: tuple[int, ...],
    ) -> str:
        delta = field.translation_name
        target = field.state_name

        return "\n".join(
            f" result_{target}[{location}] = "
            f"origin_{target}[{location}] + delta_{delta}[{location}]"
            for location in self.fixed_locations(shape)
        )

    def small_fixed_norm_body(
        self,
        field: AlgebraistField,
        shape: tuple[int, ...],
        kind: str,
    ) -> list[str]:
        name = field.translation_name
        lines: list[str] = []

        for location in self.fixed_locations(shape):
            lines.append(f" value = {name}[{location}]")
            lines.append(" total += value * value")

        if kind == "rms":
            lines.append(f" count += {prod(shape)}")

        return lines

    @staticmethod
    def index_names(rank: int) -> list[str]:
        return [f"i{index}" for index in range(rank)]

    @staticmethod
    def shape_terms(rank: int) -> list[str]:
        return [f"shape_{index}" for index in range(rank)]

    @staticmethod
    def shape_binding(array_name: str, shape_terms: Sequence[str]) -> str:
        if len(shape_terms) == 1:
            return f" {shape_terms[0]} = {array_name}.shape[0]"

        return f" {', '.join(shape_terms)} = {array_name}.shape"

    def fixed_locations(self, shape: tuple[int, ...]) -> tuple[str, ...]:
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

    @staticmethod
    def looped_rank(field: AlgebraistField, policy: AlgebraistLooped) -> int:
        if policy.rank is None:
            raise ValueError(
                f"Looped Algebraist field {field.translation_name!r} "
                "needs an explicit rank or shape."
            )

        return policy.rank