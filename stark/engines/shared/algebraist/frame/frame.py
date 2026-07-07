from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, cast

from stark.core.contracts.field import FieldLike, FieldPath as FieldPathContract
from stark.core.contracts.inner_product import InnerProductNamed
from stark.core.contracts.norm import NormLike
from stark.engines.shared.algebraist.frame.entries import (
    included_inner_product_entries,
    included_norm_entries,
)
from stark.engines.shared.algebraist.frame.inner_product import AlgebraistInnerProductL2
from stark.engines.shared.algebraist.frame.norm import AlgebraistNormRMS

AlgebraistFrameField = FieldLike[Any, Any]


@dataclass(frozen=True, slots=True)
class AlgebraistFrame:
    """Structural description of the fields visible to an Algebraist."""

    fields: tuple[AlgebraistFrameField, ...]
    norms: tuple[NormLike[object], ...]
    inner_products: tuple[InnerProductNamed[object], ...]

    def __init__(
        self,
        fields: Iterable[AlgebraistFrameField],
        *,
        norms: NormLike[object] | Iterable[NormLike[object]] | None = None,
        inner_products: InnerProductNamed[object]
        | Iterable[InnerProductNamed[object]]
        | None = None,
    ) -> None:
        normalized = tuple(fields)

        if not normalized:
            raise ValueError("AlgebraistFrame requires at least one field.")
        normalized_norms = self._normalize_norms(norms, count=len(normalized))
        normalized_inner_products = self._normalize_inner_products(
            inner_products,
            count=len(normalized),
        )

        translation_paths = tuple(field.translation_path for field in normalized)
        state_paths = tuple(field.state_path for field in normalized)

        if len(set(translation_paths)) != len(translation_paths):
            raise ValueError("AlgebraistFrame fields must have unique translation paths.")

        if len(set(state_paths)) != len(state_paths):
            raise ValueError("AlgebraistFrame fields must have unique state paths.")

        object.__setattr__(self, "fields", normalized)
        object.__setattr__(self, "norms", normalized_norms)
        object.__setattr__(self, "inner_products", normalized_inner_products)

    @staticmethod
    def _normalize_norms(
        norms: NormLike[object] | Iterable[NormLike[object]] | None,
        *,
        count: int,
    ) -> tuple[NormLike[object], ...]:
        if norms is None:
            return tuple(AlgebraistNormRMS() for _index in range(count))
        if callable(norms) and hasattr(norms, "kind"):
            return tuple(norms for _index in range(count))

        normalized = tuple(cast(Iterable[NormLike[object]], norms))
        if len(normalized) != count:
            raise ValueError("AlgebraistFrame requires one norm per field.")
        return normalized

    @staticmethod
    def _normalize_inner_products(
        inner_products: InnerProductNamed[object]
        | Iterable[InnerProductNamed[object]]
        | None,
        *,
        count: int,
    ) -> tuple[InnerProductNamed[object], ...]:
        if inner_products is None:
            return tuple(AlgebraistInnerProductL2() for _index in range(count))
        if callable(inner_products) and hasattr(inner_products, "kind"):
            return tuple(inner_products for _index in range(count))

        normalized = tuple(cast(Iterable[InnerProductNamed[object]], inner_products))
        if len(normalized) != count:
            raise ValueError("AlgebraistFrame requires one inner product per field.")
        return normalized

    def __iter__(self) -> Iterator[AlgebraistFrameField]:
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    @property
    def norm_entries(self) -> tuple[tuple[AlgebraistFrameField, NormLike[object]], ...]:
        """Field/norm pairs included in generated/runtime norm handling."""

        return included_norm_entries(self)

    @property
    def inner_product_entries(
        self,
    ) -> tuple[tuple[AlgebraistFrameField, InnerProductNamed[object]], ...]:
        """Field/inner-product pairs included in frame-aware inner products."""

        return included_inner_product_entries(self)

    @property
    def translation_paths(self) -> tuple[FieldPathContract, ...]:
        """Translation-side paths in frame order."""

        return tuple(field.translation_path for field in self.fields)

    @property
    def state_paths(self) -> tuple[FieldPathContract, ...]:
        """State-side paths in frame order."""

        return tuple(field.state_path for field in self.fields)
