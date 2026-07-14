from __future__ import annotations

from typing import Any, Protocol, SupportsFloat, TypeVar, cast


ScalarValueTypeContravariant = TypeVar(
    "ScalarValueTypeContravariant",
    contravariant=True,
)


class CarrierScalarLike(Protocol[ScalarValueTypeContravariant]):
    """Convert backend scalar values to Python floats at host boundaries."""

    def to_float(self, value: ScalarValueTypeContravariant) -> float:
        """Return `value` as a Python float."""
        ...


class CarrierScalarPython:
    """Scalar converter for values that directly support `float(value)`."""

    def to_float(self, value: SupportsFloat) -> float:
        return float(value)


class CarrierScalarItem:
    """Scalar converter for backend values that may need `.item()` first."""

    def to_float(self, value: Any) -> float:
        item = getattr(value, "item", None)
        if callable(item):
            value = item()
        return float(cast(SupportsFloat, value))


__all__ = [
    "CarrierScalarLike",
    "CarrierScalarItem",
    "CarrierScalarPython",
    "ScalarValueTypeContravariant",
]
