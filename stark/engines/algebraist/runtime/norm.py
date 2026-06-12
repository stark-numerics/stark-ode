from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import sqrt
from typing import Generic, TypeVar

from stark.engines.algebraist.frame import AlgebraistFrame

TranslationType = TypeVar("TranslationType")


@dataclass(frozen=True, slots=True)
class AlgebraistRuntimeNorm(Generic[TranslationType]):
    """Runtime provider of frame-aware translation norm kernels."""

    frame: AlgebraistFrame
    field_norms: Sequence[Callable[[object], float]]

    def __post_init__(self) -> None:
        if len(self.field_norms) != len(self.frame.fields):
            raise ValueError("Runtime norm requires one field norm per frame field.")

    def provide(self, request: None = None) -> Callable[[TranslationType], float]:
        del request
        fields_and_norms = tuple(zip(self.frame.fields, self.field_norms, strict=True))

        def kernel(translation: TranslationType) -> float:
            total = 0.0
            for field, norm in fields_and_norms:
                if not field.norm.include:
                    continue
                field_norm = norm(field.translation_path.get(translation))
                total += field_norm * field_norm
            return sqrt(total)

        return kernel


__all__ = ["AlgebraistRuntimeNorm"]
