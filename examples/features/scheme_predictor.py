from __future__ import annotations

"""Seed implicit stage unknowns with scheme predictor workers."""

from dataclasses import dataclass

from stark.methods.schemes.predictors import (
    SchemePredictorKnown,
    SchemePredictorPrevious,
    SchemePredictorZero,
)


@dataclass
class Translation:
    value: float


def scale(coefficient: float, source: Translation, target: Translation) -> Translation:
    target.value = coefficient * source.value
    return target


known = Translation(3.0)
previous = Translation(2.0)

for name, predictor in (
    ("known", SchemePredictorKnown()),
    ("zero", SchemePredictorZero()),
    ("previous", SchemePredictorPrevious()),
):
    delta = Translation(-99.0)
    predictor(known=known, previous=previous, delta=delta, scale=scale)
    print(f"SchemePredictor{name.title():8s} -> initial delta {delta.value:.1f}")

print("Predictors belong to schemes: they seed the stage solve before the resolvent runs.")
