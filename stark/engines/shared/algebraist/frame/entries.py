"""Algebraist views over frame-like declarations."""

from __future__ import annotations

from typing import Any

from stark.core.contracts.field import FieldLike
from stark.core.contracts.frame import FrameLike
from stark.core.contracts.inner_product import InnerProductNamed
from stark.core.contracts.norm import NormLike

AlgebraistFrameField = FieldLike[Any, Any]


def included_norm_entries(
    frame: FrameLike,
) -> tuple[tuple[AlgebraistFrameField, NormLike[Any]], ...]:
    """Return field/norm pairs that participate in norm calculations."""

    return tuple(
        (field, norm)
        for field, norm in zip(frame.fields, frame.norms, strict=True)
        if getattr(norm, "kind", None) != "excluded"
    )


def included_inner_product_entries(
    frame: FrameLike,
) -> tuple[tuple[AlgebraistFrameField, InnerProductNamed[Any]], ...]:
    """Return field/inner-product pairs that participate in inner products."""

    return tuple(
        (field, inner_product)
        for field, inner_product in zip(
            frame.fields,
            frame.inner_products,
            strict=True,
        )
        if getattr(inner_product, "kind", None) != "excluded"
    )


__all__ = [
    "AlgebraistFrameField",
    "included_inner_product_entries",
    "included_norm_entries",
]
