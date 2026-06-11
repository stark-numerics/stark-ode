from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from inspect import Parameter, signature
from types import MappingProxyType
from typing import Any


class MethodError(ValueError):
    """Raised when a method recipe describes an inconsistent component stack."""


@dataclass(frozen=True, slots=True)
class Method:
    """
    Declarative recipe for the numerical method stack.

    A method names the scheme to use and, when needed, the resolvent and
    inverter that support it. Passing a class means `System` should
    construct that component from the current system, engine, configuration,
    and option mapping. Passing an instance means the component is already
    configured and should be used exactly as supplied.
    """

    scheme: object
    resolvent: object | None = None
    inverter: object | None = None
    scheme_options: Mapping[str, Any] = field(default_factory=dict)
    resolvent_options: Mapping[str, Any] = field(default_factory=dict)
    inverter_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scheme_options", _freeze_options(self.scheme_options))
        object.__setattr__(self, "resolvent_options", _freeze_options(self.resolvent_options))
        object.__setattr__(self, "inverter_options", _freeze_options(self.inverter_options))

        _reject_instance_options("scheme", self.scheme, self.scheme_options)
        _reject_instance_options("resolvent", self.resolvent, self.resolvent_options)
        _reject_instance_options("inverter", self.inverter, self.inverter_options)

        if not isinstance(self.scheme, type):
            if self.resolvent is not None:
                raise MethodError("A ready scheme instance cannot also declare a resolvent.")
            if self.inverter is not None:
                raise MethodError("A ready scheme instance cannot also declare an inverter.")
            return

        scheme_accepts_resolvent = _constructor_accepts(self.scheme, "resolvent")
        scheme_requires_resolvent = _constructor_requires(self.scheme, "resolvent")

        if scheme_requires_resolvent and self.resolvent is None:
            raise MethodError(
                f"{self.scheme.__name__} requires a resolvent; provide Method(..., resolvent=...)."
            )
        if self.resolvent is not None and not scheme_accepts_resolvent:
            raise MethodError(
                f"{self.scheme.__name__} does not accept a resolvent; remove Method(..., resolvent=...)."
            )

        if self.resolvent is None:
            if self.resolvent_options:
                raise MethodError("resolvent_options require a resolvent component.")
            if self.inverter is not None:
                raise MethodError("An inverter requires a resolvent component.")
            if self.inverter_options:
                raise MethodError("inverter_options require an inverter component.")
            return

        if not isinstance(self.resolvent, type):
            if self.inverter is not None:
                raise MethodError("A ready resolvent instance cannot also declare an inverter.")
            if self.inverter_options:
                raise MethodError("inverter_options require an inverter component.")
            return

        resolvent_accepts_inverter = _constructor_accepts(self.resolvent, "inverter")
        resolvent_requires_inverter = _constructor_requires(self.resolvent, "inverter")

        if resolvent_requires_inverter and self.inverter is None:
            raise MethodError(
                f"{self.resolvent.__name__} requires an inverter; provide Method(..., inverter=...)."
            )
        if self.inverter is not None and not resolvent_accepts_inverter:
            raise MethodError(
                f"{self.resolvent.__name__} does not accept an inverter; remove Method(..., inverter=...)."
            )
        if self.inverter is None and self.inverter_options:
            raise MethodError("inverter_options require an inverter component.")


def _freeze_options(options: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(options))


def _reject_instance_options(
    role: str,
    component: object | None,
    options: Mapping[str, Any],
) -> None:
    if component is not None and not isinstance(component, type) and options:
        raise MethodError(f"{role}_options require a {role} class, not a ready instance.")


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
        raise MethodError(
            f"Cannot inspect constructor for {component.__name__}; provide a class with a visible signature."
        ) from exc


__all__ = ["Method", "MethodError"]
