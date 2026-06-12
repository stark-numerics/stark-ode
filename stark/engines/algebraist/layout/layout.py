from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from stark.engines.algebraist.layout.field import AlgebraistLayoutField
from stark.engines.algebraist.layout.path import AlgebraistLayoutPath


@dataclass(frozen=True, slots=True)
class AlgebraistLayout:
    """Structural description of the fields visible to an Algebraist."""

    fields: tuple[AlgebraistLayoutField, ...]

    def __init__(self, fields: Iterable[AlgebraistLayoutField]) -> None:
        normalized = tuple(fields)

        if not normalized:
            raise ValueError("AlgebraistLayout requires at least one field.")

        translation_paths = tuple(field.translation_path for field in normalized)
        state_paths = tuple(field.state_path for field in normalized)

        if len(set(translation_paths)) != len(translation_paths):
            raise ValueError("AlgebraistLayout fields must have unique translation paths.")

        if len(set(state_paths)) != len(state_paths):
            raise ValueError("AlgebraistLayout fields must have unique state paths.")

        object.__setattr__(self, "fields", normalized)

    def __iter__(self) -> Iterator[AlgebraistLayoutField]:
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    @property
    def norm_fields(self) -> tuple[AlgebraistLayoutField, ...]:
        """Fields included in generated/runtime norm handling."""

        return tuple(field for field in self.fields if field.norm.include)

    @property
    def translation_paths(self) -> tuple[AlgebraistLayoutPath, ...]:
        """Translation-side paths in layout order."""

        return tuple(field.translation_path for field in self.fields)

    @property
    def state_paths(self) -> tuple[AlgebraistLayoutPath, ...]:
        """State-side paths in layout order."""

        return tuple(field.state_path for field in self.fields)
