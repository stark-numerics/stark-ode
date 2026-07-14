from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from stark.core.contracts.field import FieldLike
from stark.core.contracts.frame import FrameLike
from stark.core.contracts.inner_product import InnerProductNamed
from stark.core.contracts.translation import TranslationTypeContravariant


AlgebraistInnerProductKernel = Callable[
    [TranslationTypeContravariant, TranslationTypeContravariant],
    float,
]


def included_inner_product_entries(
    frame: FrameLike,
) -> tuple[tuple[FieldLike[Any, Any], InnerProductNamed[Any]], ...]:
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


class AlgebraistInnerProduct(Protocol[TranslationTypeContravariant]):
    """Provider of frame-aware translation inner-product kernels."""

    def provide(
        self,
        request: None = None,
    ) -> AlgebraistInnerProductKernel[TranslationTypeContravariant]:
        ...


__all__ = [
    "AlgebraistInnerProduct",
    "AlgebraistInnerProductKernel",
    "included_inner_product_entries",
]
