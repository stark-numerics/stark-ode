from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stark.schemes.descriptor import SchemeDescriptor


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


def with_scheme_display(cls):
    """Install the standard scheme display surface on a concrete scheme class."""

    @classmethod
    def with_scheme_display_for_class(inner_cls) -> SchemeDisplay:
        return SchemeDisplay(inner_cls.descriptor, inner_cls.tableau)

    @classmethod
    def display_tableau(inner_cls) -> str:
        return inner_cls.with_scheme_display().display_tableau()

    @property
    def short_name(self) -> str:
        return type(self).with_scheme_display().short_name

    @property
    def full_name(self) -> str:
        return type(self).with_scheme_display().full_name

    def repr_scheme(self) -> str:
        return type(self).with_scheme_display().repr_for(type(self).__name__)

    def str_scheme(self) -> str:
        return type(self).with_scheme_display().str_for()

    def format_scheme(self, format_spec: str) -> str:
        return type(self).with_scheme_display().format_for(format_spec)

    cls.with_scheme_display = with_scheme_display_for_class
    cls.display_tableau = display_tableau
    cls.short_name = short_name
    cls.full_name = full_name
    cls.__repr__ = repr_scheme
    cls.__str__ = str_scheme
    cls.__format__ = format_scheme
    return cls


__all__ = ["SchemeDisplay", "with_scheme_display"]
