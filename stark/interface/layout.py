from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from stark.algebraist.layout import (
    AlgebraistLayout,
    AlgebraistLayoutBroadcast,
    AlgebraistLayoutField,
    AlgebraistLayoutLooped,
    AlgebraistLayoutNormPolicy,
    AlgebraistLayoutPolicy,
    AlgebraistLayoutScalar,
    AlgebraistLayoutUnravel,
)
from stark.algebraist.layout.path import AlgebraistLayoutPathLike
from stark.interface.norm import LayoutNormPolicy, LayoutNormRMS


@dataclass(frozen=True, slots=True)
class LayoutField:
    """One user-facing state field in a STARK layout."""

    state: AlgebraistLayoutPathLike
    translation: AlgebraistLayoutPathLike | None = None
    shape: tuple[int, ...] | list[int] | None = None
    norm: LayoutNormPolicy | AlgebraistLayoutNormPolicy = field(
        default_factory=LayoutNormRMS
    )

    def __post_init__(self) -> None:
        if self.translation is None:
            object.__setattr__(self, "translation", self.state)

    def to_algebraist_field(self) -> AlgebraistLayoutField:
        translation = self.translation
        if translation is None:
            translation = self.state
        return AlgebraistLayoutField(
            translation,
            self.state,
            policy=self.algebraist_policy(),
            norm=self.algebraist_norm(),
        )

    def algebraist_norm(self) -> AlgebraistLayoutNormPolicy:
        norm = self.norm
        if hasattr(norm, "to_algebraist_norm"):
            return norm.to_algebraist_norm()
        return norm

    def algebraist_policy(self) -> AlgebraistLayoutPolicy:
        if self.shape is not None:
            return AlgebraistLayoutLooped(shape=self.shape)
        return AlgebraistLayoutBroadcast()


@dataclass(frozen=True, slots=True)
class Layout:
    """
    User-facing declaration of structured state and translation fields.

    A layout tells an engine which state paths exist, which translation paths
    hold their updates, what shape each field has, and how each field contributes
    to norms. It accepts explicit `LayoutField` objects, simple path names,
    or a mapping such as `{"y": {"translation": "dy", "shape": (2,)}}`.
    """

    fields: tuple[LayoutField, ...]

    def __init__(
        self,
        fields: LayoutField
        | AlgebraistLayoutPathLike
        | Mapping[AlgebraistLayoutPathLike, Any]
        | Iterable[LayoutField | AlgebraistLayoutPathLike | Mapping[str, Any]],
    ) -> None:
        if isinstance(fields, Mapping):
            normalized = tuple(
                self._field_from_mapping_item(state, spec)
                for state, spec in fields.items()
            )
        elif isinstance(fields, LayoutField) or isinstance(fields, str):
            normalized = (self._coerce_field(fields),)
        else:
            normalized = tuple(self._coerce_field(field) for field in fields)
        if not normalized:
            raise ValueError("Layout requires at least one field.")

        object.__setattr__(self, "fields", normalized)
        self.to_algebraist_layout()

    @staticmethod
    def _coerce_field(
        field: LayoutField | AlgebraistLayoutPathLike | Mapping[str, Any],
    ) -> LayoutField:
        if isinstance(field, LayoutField):
            return field
        if isinstance(field, Mapping):
            return Layout._field_from_spec(field)
        return LayoutField(field)

    @staticmethod
    def _field_from_spec(spec: Mapping[str, Any]) -> LayoutField:
        if "state" not in spec:
            raise ValueError("Layout field mappings require a 'state' entry.")
        kwargs = Layout._field_kwargs(spec)
        return LayoutField(spec["state"], **kwargs)

    @staticmethod
    def _field_from_mapping_item(
        state: AlgebraistLayoutPathLike,
        spec: Any,
    ) -> LayoutField:
        if spec is None:
            return LayoutField(state)
        if not isinstance(spec, Mapping):
            raise TypeError(
                "Layout mapping values must be field option mappings or None."
            )
        kwargs = Layout._field_kwargs(spec)
        return LayoutField(state, **kwargs)

    @staticmethod
    def _field_kwargs(spec: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {"state", "translation", "shape", "norm"}
        unsupported = tuple(name for name in spec if name not in allowed)
        if unsupported:
            names = ", ".join(str(name) for name in unsupported)
            raise ValueError(f"Unsupported Layout field option(s): {names}.")
        return {
            name: spec[name]
            for name in ("translation", "shape", "norm")
            if name in spec
        }

    def __iter__(self):
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def to_algebraist_layout(self) -> AlgebraistLayout:
        return AlgebraistLayout(field.to_algebraist_field() for field in self.fields)


__all__ = [
    "Layout",
    "LayoutField",
]
