from __future__ import annotations

from dataclasses import dataclass
from math import isclose


@dataclass(frozen=True, slots=True)
class AlgebraistGeneratorEmitterTerm:
    """One source term in a generated emitter expression."""

    coefficient: str
    source: str


@dataclass(frozen=True, slots=True)
class AlgebraistGeneratorEmitterExpression:
    """Normalized expression consumed by generator emitters."""

    terms: tuple[AlgebraistGeneratorEmitterTerm, ...]

    @classmethod
    def from_runtime_coefficients(
        cls,
        *,
        coefficients: tuple[str, ...],
        sources: tuple[str, ...],
    ) -> "AlgebraistGeneratorEmitterExpression":
        if len(coefficients) != len(sources):
            raise ValueError("coefficients and sources must have the same length.")
        return cls(
            tuple(
                AlgebraistGeneratorEmitterTerm(coefficient=coefficient, source=source)
                for coefficient, source in zip(coefficients, sources, strict=True)
            )
        )

    @classmethod
    def from_fixed_coefficients(
        cls,
        *,
        coefficients: tuple[float, ...],
        sources: tuple[str, ...],
        coefficient_prefix: str = "_a",
        inline_coefficients: bool = False,
    ) -> "AlgebraistGeneratorEmitterExpression":
        if len(coefficients) != len(sources):
            raise ValueError("coefficients and sources must have the same length.")
        terms: list[AlgebraistGeneratorEmitterTerm] = []
        for index, coefficient in enumerate(coefficients):
            if isclose(coefficient, 0.0, abs_tol=0.0):
                continue
            if inline_coefficients:
                coefficient_source = repr(float(coefficient))
            else:
                coefficient_source = f"{coefficient_prefix}{index}"
            terms.append(
                AlgebraistGeneratorEmitterTerm(
                    coefficient=coefficient_source,
                    source=sources[index],
                )
            )
        return cls(tuple(terms))

    def source(self) -> str:
        if not self.terms:
            return "0.0"
        rendered = [self._term_source(term) for term in self.terms]
        return self._balanced_sum(rendered)

    @staticmethod
    def _term_source(term: AlgebraistGeneratorEmitterTerm) -> str:
        if term.coefficient == "1.0" or term.coefficient == "1":
            return term.source
        if term.coefficient == "-1.0" or term.coefficient == "-1":
            return f"(-{term.source})"
        return f"{term.coefficient} * {term.source}"

    @classmethod
    def _balanced_sum(cls, values: list[str]) -> str:
        if not values:
            return "0.0"
        if len(values) == 1:
            return values[0]
        midpoint = len(values) // 2
        left = cls._balanced_sum(values[:midpoint])
        right = cls._balanced_sum(values[midpoint:])
        return f"({left} + {right})"
