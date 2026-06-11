from __future__ import annotations

from stark.methods.schemes.display.display import SchemeDisplay


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


__all__ = ["with_scheme_display"]
