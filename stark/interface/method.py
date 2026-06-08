from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from inspect import Parameter, signature
from types import MappingProxyType
from typing import Any


class StarkMethodError(ValueError):
    """Raised when a method recipe describes an inconsistent component stack."""


@dataclass(frozen=True, slots=True)
class StarkMethod:
    """
    Declarative recipe for the numerical method stack.

    A method names the scheme class to construct and, when needed, the
    resolvent and inverter classes that support it. The scheme is required; the
    resolvent is used for implicit or coupled solves; the inverter is used by
    linearising resolvents. The option mappings are passed to the corresponding
    component constructors after the current system, engine, and configuration
    have supplied the standard ingredients.
    """

    scheme: type[Any]
    resolvent: type[Any] | None = None
    inverter: type[Any] | None = None
    scheme_options: Mapping[str, Any] = field(default_factory=dict)
    resolvent_options: Mapping[str, Any] = field(default_factory=dict)
    inverter_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_component_class("scheme", self.scheme)
        _require_optional_component_class("resolvent", self.resolvent)
        _require_optional_component_class("inverter", self.inverter)

        object.__setattr__(self, "scheme_options", _freeze_options(self.scheme_options))
        object.__setattr__(self, "resolvent_options", _freeze_options(self.resolvent_options))
        object.__setattr__(self, "inverter_options", _freeze_options(self.inverter_options))

        scheme_accepts_resolvent = _constructor_accepts(self.scheme, "resolvent")
        scheme_requires_resolvent = _constructor_requires(self.scheme, "resolvent")

        if scheme_requires_resolvent and self.resolvent is None:
            raise StarkMethodError(
                f"{self.scheme.__name__} requires a resolvent; provide StarkMethod(..., resolvent=...)."
            )
        if self.resolvent is not None and not scheme_accepts_resolvent:
            raise StarkMethodError(
                f"{self.scheme.__name__} does not accept a resolvent; remove StarkMethod(..., resolvent=...)."
            )

        if self.resolvent is None:
            if self.resolvent_options:
                raise StarkMethodError("resolvent_options require a resolvent component.")
            if self.inverter is not None:
                raise StarkMethodError("An inverter requires a resolvent component.")
            if self.inverter_options:
                raise StarkMethodError("inverter_options require an inverter component.")
            return

        resolvent_accepts_inverter = _constructor_accepts(self.resolvent, "inverter")
        resolvent_requires_inverter = _constructor_requires(self.resolvent, "inverter")

        if resolvent_requires_inverter and self.inverter is None:
            raise StarkMethodError(
                f"{self.resolvent.__name__} requires an inverter; provide StarkMethod(..., inverter=...)."
            )
        if self.inverter is not None and not resolvent_accepts_inverter:
            raise StarkMethodError(
                f"{self.resolvent.__name__} does not accept an inverter; remove StarkMethod(..., inverter=...)."
            )
        if self.inverter is None and self.inverter_options:
            raise StarkMethodError("inverter_options require an inverter component.")


def _freeze_options(options: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(options))


def _require_component_class(role: str, component: object) -> None:
    if not isinstance(component, type):
        raise StarkMethodError(f"{role} must be a class, got {component!r}.")


def _require_optional_component_class(role: str, component: object | None) -> None:
    if component is not None:
        _require_component_class(role, component)


def _constructor_accepts(component: type[Any], parameter_name: str) -> bool:
    parameters = _constructor_parameters(component)
    if parameter_name in parameters:
        return True
    return any(parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values())


def _constructor_requires(component: type[Any], parameter_name: str) -> bool:
    parameter = _constructor_parameters(component).get(parameter_name)
    if parameter is None:
        return False
    if parameter.kind not in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY):
        return False
    return parameter.default is Parameter.empty


def _constructor_parameters(component: type[Any]) -> Mapping[str, Parameter]:
    try:
        return signature(component).parameters
    except (TypeError, ValueError) as exc:
        raise StarkMethodError(
            f"Cannot inspect constructor for {component.__name__}; provide a class with a visible signature."
        ) from exc


__all__ = ["StarkMethod", "StarkMethodError"]
