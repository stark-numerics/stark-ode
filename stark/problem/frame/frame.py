"""User-facing frame declarations for structured state.

A `Frame` names the state fields a model owns, the translation fields where
dynamics are written, each field's storage shape, and the frame-level policies
used by adaptive methods and algebra helpers. Engines translate this
declaration into backend-specific allocation and algebra kernels.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from stark.core.contracts.problem.field import FieldLike
from stark.core.contracts.problem.inner_product import InnerProductNamed
from stark.core.contracts.problem.norm import NormLike
from stark.problem.frame.field import Field
from stark.problem.frame.inner_product import InnerProductL2
from stark.problem.frame.norm import NormRMS
from stark.problem.frame.path import FieldPath, FieldPathLike


FrameFieldLike = FieldLike[Any, Any]


@dataclass(frozen=True, slots=True)
class Frame:
    """
    User-facing declaration of structured state and translation fields.

    A frame tells an engine which state paths exist, which translation paths
    hold their updates, what shape each field has, and how each field contributes
    to norms and inner products. It accepts explicit `FieldLike` objects,
    simple path names, or a mapping such as
    `{"y": {"translation": "dy", "shape": (2,)}}`.
    """

    fields: tuple[FrameFieldLike, ...]
    norms: tuple[NormLike[object], ...]
    inner_products: tuple[InnerProductNamed[object], ...]

    def __init__(
        self,
        fields: FrameFieldLike
        | FieldPathLike
        | Mapping[FieldPathLike, Any]
        | Iterable[FrameFieldLike | FieldPathLike | Mapping[str, Any]],
        norms: NormLike[object] | Iterable[NormLike[object]] | None = None,
        inner_products: InnerProductNamed[object]
        | Iterable[InnerProductNamed[object]]
        | None = None,
    ) -> None:
        if isinstance(fields, Mapping):
            field_mapping = cast(Mapping[FieldPathLike, Any], fields)
            entries = tuple(
                self._field_from_mapping_item(state, spec)
                for state, spec in field_mapping.items()
            )
        elif isinstance(fields, str) or self._is_field_like(fields):
            entries = (
                self._coerce_field(
                    cast(FrameFieldLike | FieldPathLike | Mapping[str, Any], fields)
                ),
            )
        else:
            field_iterable = cast(
                Iterable[FrameFieldLike | FieldPathLike | Mapping[str, Any]],
                fields,
            )
            entries = tuple(self._coerce_field(field) for field in field_iterable)
        if not entries:
            raise ValueError("Frame requires at least one field.")

        normalized_fields = tuple(field for field, _norm, _inner_product in entries)
        declared_norms = tuple(norm for _field, norm, _inner_product in entries)
        declared_inner_products = tuple(
            inner_product for _field, _norm, inner_product in entries
        )
        normalized_norms = (
            declared_norms
            if norms is None
            else self._normalize_norms(norms, count=len(normalized_fields))
        )
        normalized_inner_products = (
            declared_inner_products
            if inner_products is None
            else self._normalize_inner_products(
                inner_products,
                count=len(normalized_fields),
            )
        )

        object.__setattr__(self, "fields", normalized_fields)
        object.__setattr__(self, "norms", normalized_norms)
        object.__setattr__(self, "inner_products", normalized_inner_products)
        self._validate()

    @classmethod
    def scalar(
        cls,
        state: FieldPathLike,
        *,
        translation: FieldPathLike | None = None,
        norm: NormLike[object] | None = None,
        inner_product: InnerProductNamed[object] | None = None,
    ) -> "Frame":
        """Build a one-field frame for scalar-like state storage.

        This is a convenience spelling for the full mapping syntax. The
        resulting field still has shape `(1,)`, matching the single-entry
        array style used by the getting-started examples.
        """

        spec: dict[str, Any] = {
            "translation": translation if translation is not None else state,
            "shape": (1,),
        }
        if norm is not None:
            spec["norm"] = norm
        if inner_product is not None:
            spec["inner_product"] = inner_product
        return cls({state: spec})

    @classmethod
    def vector(
        cls,
        state: FieldPathLike,
        *,
        translation: FieldPathLike | None = None,
        length: int,
        norm: NormLike[object] | None = None,
        inner_product: InnerProductNamed[object] | None = None,
    ) -> "Frame":
        """Build a one-field frame for vector state storage.

        This is a convenience spelling for `Frame({state: {"translation": ...,
        "shape": (length,)}})`.
        """

        spec: dict[str, Any] = {
            "translation": translation if translation is not None else state,
            "shape": (length,),
        }
        if norm is not None:
            spec["norm"] = norm
        if inner_product is not None:
            spec["inner_product"] = inner_product
        return cls({state: spec})

    @classmethod
    def array(
        cls,
        state: FieldPathLike,
        *,
        translation: FieldPathLike | None = None,
        shape: tuple[int, ...] | list[int],
        norm: NormLike[object] | None = None,
        inner_product: InnerProductNamed[object] | None = None,
    ) -> "Frame":
        """Build a one-field frame for array state storage.

        Use `array` when the field is naturally an array and its dimensionality
        matters to the model. This is a convenience spelling for
        `Frame({state: {"translation": ..., "shape": shape}})`.
        """

        spec: dict[str, Any] = {
            "translation": translation if translation is not None else state,
            "shape": tuple(shape),
        }
        if norm is not None:
            spec["norm"] = norm
        if inner_product is not None:
            spec["inner_product"] = inner_product
        return cls({state: spec})

    @staticmethod
    def _coerce_field(
        field: FrameFieldLike | FieldPathLike | Mapping[str, Any],
    ) -> tuple[FrameFieldLike, NormLike[object], InnerProductNamed[object]]:
        if isinstance(field, Mapping):
            return Frame._field_from_spec(field)
        if Frame._is_path_like(field):
            return Field(cast(FieldPathLike, field)), NormRMS(), InnerProductL2()
        if Frame._is_field_like(field):
            return cast(FrameFieldLike, field), NormRMS(), InnerProductL2()
        raise TypeError("Frame fields must be Field-like, path-like, or mappings.")

    @staticmethod
    def _field_from_spec(
        spec: Mapping[str, Any],
    ) -> tuple[FrameFieldLike, NormLike[object], InnerProductNamed[object]]:
        if "state" not in spec:
            raise ValueError("Frame field mappings require a 'state' entry.")
        kwargs = Frame._field_kwargs(spec)
        norm = cast(NormLike[object], kwargs.pop("norm", NormRMS()))
        inner_product = cast(
            InnerProductNamed[object],
            kwargs.pop("inner_product", InnerProductL2()),
        )
        return Field(spec["state"], **kwargs), norm, inner_product

    @staticmethod
    def _field_from_mapping_item(
        state: FieldPathLike,
        spec: Any,
    ) -> tuple[FrameFieldLike, NormLike[object], InnerProductNamed[object]]:
        if spec is None:
            return Field(state), NormRMS(), InnerProductL2()
        if not isinstance(spec, Mapping):
            raise TypeError(
                "Frame mapping values must be field option mappings or None."
            )
        kwargs = Frame._field_kwargs(spec)
        norm = cast(NormLike[object], kwargs.pop("norm", NormRMS()))
        inner_product = cast(
            InnerProductNamed[object],
            kwargs.pop("inner_product", InnerProductL2()),
        )
        return Field(state, **kwargs), norm, inner_product

    @staticmethod
    def _is_path_like(value: Any) -> bool:
        if isinstance(value, str):
            return True
        if not isinstance(value, Sequence):
            return False
        return all(isinstance(part, str) for part in value)

    @staticmethod
    def _is_field_like(value: Any) -> bool:
        attributes = (
            "state",
            "translation",
            "shape",
            "policy",
            "state_path",
            "translation_path",
        )
        methods = ("state_expression", "translation_expression")
        return all(hasattr(value, name) for name in attributes) and all(
            callable(getattr(value, name, None)) for name in methods
        )

    @staticmethod
    def _field_kwargs(spec: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {
            "state",
            "translation",
            "shape",
            "norm",
            "inner_product",
            "policy",
        }
        unsupported = tuple(name for name in spec if name not in allowed)
        if unsupported:
            names = ", ".join(str(name) for name in unsupported)
            raise ValueError(f"Unsupported Frame field option(s): {names}.")
        return {
            name: spec[name]
            for name in ("translation", "shape", "norm", "inner_product", "policy")
            if name in spec
        }

    @staticmethod
    def _normalize_norms(
        norms: NormLike[object] | Iterable[NormLike[object]],
        *,
        count: int,
    ) -> tuple[NormLike[object], ...]:
        if callable(norms) and hasattr(norms, "kind"):
            return tuple(cast(NormLike[object], norms) for _index in range(count))

        normalized = tuple(cast(Iterable[NormLike[object]], norms))
        if len(normalized) != count:
            raise ValueError("Frame requires one norm per field.")
        return normalized

    @staticmethod
    def _normalize_inner_products(
        inner_products: InnerProductNamed[object] | Iterable[InnerProductNamed[object]],
        *,
        count: int,
    ) -> tuple[InnerProductNamed[object], ...]:
        if callable(inner_products) and hasattr(inner_products, "kind"):
            return tuple(
                cast(InnerProductNamed[object], inner_products)
                for _index in range(count)
            )

        normalized = tuple(
            cast(Iterable[InnerProductNamed[object]], inner_products)
        )
        if len(normalized) != count:
            raise ValueError("Frame requires one inner product per field.")
        return normalized

    def __iter__(self):
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    @property
    def translation_paths(self) -> tuple[FieldPath, ...]:
        return tuple(field.translation_path for field in self.fields)

    @property
    def state_paths(self) -> tuple[FieldPath, ...]:
        return tuple(field.state_path for field in self.fields)

    def _validate(self) -> None:
        translation_paths = self.translation_paths
        state_paths = self.state_paths

        if len(set(translation_paths)) != len(translation_paths):
            raise ValueError("Frame fields must have unique translation paths.")

        if len(set(state_paths)) != len(state_paths):
            raise ValueError("Frame fields must have unique state paths.")


__all__ = [
    "Frame",
    "Field",
    "FieldPath",
    "FieldPathLike",
]
