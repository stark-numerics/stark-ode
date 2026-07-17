from __future__ import annotations

from dataclasses import dataclass
from math import isclose


@dataclass(frozen=True, slots=True)
class GeneratorExpressionTerm:
    """One source term in a generated expression."""

    coefficient: str
    source: str


@dataclass(frozen=True, slots=True)
class GeneratorExpression:
    """Balanced expression tree rendered as Python source."""

    terms: tuple[GeneratorExpressionTerm, ...]

    @classmethod
    def from_runtime_coefficients(
        cls,
        *,
        coefficients: tuple[str, ...],
        sources: tuple[str, ...],
    ) -> "GeneratorExpression":
        if len(coefficients) != len(sources):
            raise ValueError("coefficients and sources must have the same length.")
        return cls(
            tuple(
                GeneratorExpressionTerm(coefficient=coefficient, source=source)
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
    ) -> "GeneratorExpression":
        if len(coefficients) != len(sources):
            raise ValueError("coefficients and sources must have the same length.")
        terms: list[GeneratorExpressionTerm] = []
        for index, coefficient in enumerate(coefficients):
            if isclose(coefficient, 0.0, abs_tol=0.0):
                continue
            if inline_coefficients:
                coefficient_source = repr(float(coefficient))
            else:
                coefficient_source = f"{coefficient_prefix}{index}"
            terms.append(
                GeneratorExpressionTerm(
                    coefficient=coefficient_source,
                    source=sources[index],
                )
            )
        return cls(tuple(terms))

    def source(self) -> str:
        if not self.terms:
            return "0.0"
        return self.balanced_sum([self.term_source(term) for term in self.terms])

    @staticmethod
    def term_source(term: GeneratorExpressionTerm) -> str:
        if term.coefficient in {"1", "1.0"}:
            return term.source
        if term.coefficient in {"-1", "-1.0"}:
            return f"(-{term.source})"
        return f"{term.coefficient} * {term.source}"

    @classmethod
    def balanced_sum(cls, values: list[str]) -> str:
        if not values:
            return "0.0"
        if len(values) == 1:
            return values[0]
        midpoint = len(values) // 2
        left = cls.balanced_sum(values[:midpoint])
        right = cls.balanced_sum(values[midpoint:])
        return f"({left} + {right})"


__all__ = ["GeneratorExpression", "GeneratorExpressionTerm"]
