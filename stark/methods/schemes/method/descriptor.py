from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SchemeDescriptor:
    short_name: str
    full_name: str

    def __repr__(self) -> str:
        return f"SchemeDescriptor(short_name={self.short_name!r}, full_name={self.full_name!r})"

    def __str__(self) -> str:
        return f"{self.short_name} ({self.full_name})"

    def display_tableau(self, tableau: Any) -> str:
        return tableau.display(short_name=self.short_name, full_name=self.full_name)

    def repr_for(self, class_name: str, tableau: Any) -> str:
        return "\n".join(
            [
                f"{class_name}(short_name={self.short_name!r}, full_name={self.full_name!r})",
                self.display_tableau(tableau),
            ]
        )


__all__ = ["SchemeDescriptor"]









