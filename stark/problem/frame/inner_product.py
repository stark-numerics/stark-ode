"""Inner product policies used by frame declarations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Number
from typing import Any, ClassVar, cast

from stark.core.contracts.inner_product import InnerProductNamed


def translation_field_pairs(left: Any, right: Any) -> Iterable[tuple[Any, Any]]:
    if isinstance(left, Number) and isinstance(right, Number):
        return ((left, right),)

    left_flat = getattr(left, "flat", None)
    right_flat = getattr(right, "flat", None)
    if left_flat is not None and right_flat is not None:
        return zip(left_flat, right_flat)

    try:
        left_iter = iter(cast(Iterable[Any], left))
        right_iter = iter(cast(Iterable[Any], right))
    except TypeError:
        return ((left, right),)
    return zip(left_iter, right_iter)


@dataclass(frozen=True, slots=True)
class InnerProductL2:
    """Use the unscaled L2 field inner product."""

    kind: ClassVar[str] = "l2"

    def __call__(self, left, right) -> float:
        total: Any = 0.0
        for left_item, right_item in translation_field_pairs(left, right):
            total += left_item * right_item
        return float(total)


@dataclass(frozen=True, slots=True)
class InnerProductRMS:
    """Use the mean-scaled L2 field inner product."""

    kind: ClassVar[str] = "rms"

    def __call__(self, left, right) -> float:
        total: Any = 0.0
        count = 0
        for left_item, right_item in translation_field_pairs(left, right):
            total += left_item * right_item
            count += 1
        if count == 0:
            return 0.0
        return float(total / count)


@dataclass(frozen=True, slots=True)
class InnerProductExcluded:
    """Exclude this field from frame-aware inner products."""

    kind: ClassVar[str] = "excluded"

    def __call__(self, left, right) -> float:
        return 0.0


__all__ = [
    "InnerProductExcluded",
    "InnerProductL2",
    "InnerProductNamed",
    "InnerProductRMS",
]
