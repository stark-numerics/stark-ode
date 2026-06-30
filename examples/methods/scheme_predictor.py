"""Seed implicit stage unknowns with scheme predictor workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stark.methods.schemes.predictor import (
    SchemePredictorKnown,
    SchemePredictorPrevious,
    SchemePredictorZero,
)


@dataclass
class ToyTranslation:
    value: float

    def __call__(self, origin, result) -> None:
        result.value = origin.value + self.value

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: ToyTranslation) -> ToyTranslation:
        return ToyTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> ToyTranslation:
        return ToyTranslation(scalar * self.value)


def scale(a: float, x: Any, out: Any) -> Any:
    out.value = a * x.value
    return out


if __name__ == "__main__":
    known = ToyTranslation(3.0)
    previous = ToyTranslation(2.0)

    for name, predictor in (
        ("known", SchemePredictorKnown()),
        ("zero", SchemePredictorZero()),
        ("previous", SchemePredictorPrevious()),
    ):
        delta = ToyTranslation(-99.0)
        predictor(known=known, previous=previous, delta=delta, scale=scale)
        print(f"SchemePredictor{name.title():8s} -> initial delta {delta.value:.1f}")

    print("Predictors belong to schemes: they seed the stage solve before the resolvent runs.")
