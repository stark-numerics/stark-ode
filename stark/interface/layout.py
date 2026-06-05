from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from stark.algebraist.layout import (
    AlgebraistLayout,
    AlgebraistLayoutBroadcast,
    AlgebraistLayoutField,
    AlgebraistLayoutLooped,
    AlgebraistLayoutPolicy,
    AlgebraistLayoutScalar,
    AlgebraistLayoutUnravel,
)
from stark.algebraist.layout.path import AlgebraistLayoutPathLike


@dataclass(frozen=True, slots=True)
class StarkField:
    """One user-facing state field in a STARK layout."""

    state: AlgebraistLayoutPathLike
    translation: AlgebraistLayoutPathLike | None = None
    shape: tuple[int, ...] | list[int] | None = None
    include_in_norm: bool = True

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
            include_in_norm=self.include_in_norm,
        )

    def algebraist_policy(self) -> AlgebraistLayoutPolicy:
        if self.shape is not None:
            return AlgebraistLayoutLooped(shape=self.shape)
        return AlgebraistLayoutBroadcast()


@dataclass(frozen=True, slots=True)
class StarkLayout:
    """User-facing declaration of state and translation fields."""

    fields: tuple[StarkField, ...]

    def __init__(
        self,
        fields: StarkField
        | AlgebraistLayoutPathLike
        | Iterable[StarkField | AlgebraistLayoutPathLike],
    ) -> None:
        if isinstance(fields, StarkField) or isinstance(fields, str):
            normalized = (fields if isinstance(fields, StarkField) else StarkField(fields),)
        else:
            normalized = tuple(
                field if isinstance(field, StarkField) else StarkField(field)
                for field in fields
            )
        if not normalized:
            raise ValueError("StarkLayout requires at least one field.")

        object.__setattr__(self, "fields", normalized)
        self.to_algebraist_layout()

    def __iter__(self):
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def to_algebraist_layout(self) -> AlgebraistLayout:
        return AlgebraistLayout(
            field.to_algebraist_field()
            for field in self.fields
        )


__all__ = [
    "StarkField",
    "StarkLayout",
]
