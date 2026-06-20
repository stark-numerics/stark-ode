from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from stark.engines.shared.algebraist.frame.field import AlgebraistFrameField
from stark.engines.shared.algebraist.frame.path import AlgebraistFramePath


@dataclass(frozen=True, slots=True)
class AlgebraistFrame:
    """Structural description of the fields visible to an Algebraist."""

    fields: tuple[AlgebraistFrameField, ...]

    def __init__(self, fields: Iterable[AlgebraistFrameField]) -> None:
        normalized = tuple(fields)

        if not normalized:
            raise ValueError("AlgebraistFrame requires at least one field.")

        translation_paths = tuple(field.translation_path for field in normalized)
        state_paths = tuple(field.state_path for field in normalized)

        if len(set(translation_paths)) != len(translation_paths):
            raise ValueError("AlgebraistFrame fields must have unique translation paths.")

        if len(set(state_paths)) != len(state_paths):
            raise ValueError("AlgebraistFrame fields must have unique state paths.")

        object.__setattr__(self, "fields", normalized)

    def __iter__(self) -> Iterator[AlgebraistFrameField]:
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    @property
    def norm_fields(self) -> tuple[AlgebraistFrameField, ...]:
        """Fields included in generated/runtime norm handling."""

        return tuple(field for field in self.fields if field.norm.include)

    @property
    def translation_paths(self) -> tuple[AlgebraistFramePath, ...]:
        """Translation-side paths in frame order."""

        return tuple(field.translation_path for field in self.fields)

    @property
    def state_paths(self) -> tuple[AlgebraistFramePath, ...]:
        """State-side paths in frame order."""

        return tuple(field.state_path for field in self.fields)
