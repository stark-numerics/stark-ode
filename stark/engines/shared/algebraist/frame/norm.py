from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import sqrt
from numbers import Number
from typing import ClassVar, cast

from stark.core.contracts.norm import NormLike

ScalarValue = float | complex


def _values(translation_field) -> Iterable[ScalarValue]:
    if isinstance(translation_field, Number):
        return cast(tuple[ScalarValue], (translation_field,))

    flat = getattr(translation_field, "flat", None)
    if flat is not None:
        return cast(Iterable[ScalarValue], flat)

    try:
        return cast(Iterable[ScalarValue], iter(translation_field))
    except TypeError:
        return cast(tuple[ScalarValue], (translation_field,))


AlgebraistNormLike = NormLike[object]


@dataclass(frozen=True, slots=True)
class AlgebraistNormRMS:
    """Use a root-mean-square field norm."""

    kind: ClassVar[str] = "rms"

    def __call__(self, translation_field) -> float:
        total = 0.0
        count = 0
        for item in _values(translation_field):
            total += abs(item) ** 2
            count += 1
        if count == 0:
            return 0.0
        return sqrt(total / count)


@dataclass(frozen=True, slots=True)
class AlgebraistNormMax:
    """Use a maximum absolute-entry field norm."""

    kind: ClassVar[str] = "max"

    def __call__(self, translation_field) -> float:
        result = 0.0
        for item in _values(translation_field):
            item_norm = abs(item)
            if item_norm > result:
                result = item_norm
        return float(result)


@dataclass(frozen=True, slots=True)
class AlgebraistNormExcluded:
    """Exclude this field from frame-aware norms."""

    kind: ClassVar[str] = "excluded"

    def __call__(self, translation_field) -> float:
        return 0.0


__all__ = [
    "AlgebraistNormExcluded",
    "AlgebraistNormLike",
    "AlgebraistNormMax",
    "AlgebraistNormRMS",
]
