from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stark.schemes.descriptor import SchemeDescriptor
from stark.schemes.support.explicit import SchemeSupportExplicit


@dataclass(frozen=True, slots=True)
class SchemeDisplay:
    """Display adapter for a concrete scheme class.

    This keeps representation/tableau formatting reusable without making
    concrete schemes inherit algorithmic behaviour just to get display methods.
    """

    descriptor: SchemeDescriptor
    tableau: Any

    @property
    def short_name(self) -> str:
        return self.descriptor.short_name

    @property
    def full_name(self) -> str:
        return self.descriptor.full_name

    def display_tableau(self) -> str:
        return self.descriptor.display_tableau(self.tableau)

    def repr_for(self, class_name: str) -> str:
        return self.descriptor.repr_for(class_name, self.tableau)

    def str_for(self) -> str:
        return self.display_tableau()

    def format_for(self, format_spec: str) -> str:
        return format(self.str_for(), format_spec)


__all__ = ["SchemeDisplay"]