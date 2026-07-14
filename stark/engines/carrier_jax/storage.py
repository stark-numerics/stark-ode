"""Storage checks for JAX-backed carrier values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, cast

import jax.numpy as jnp  # type: ignore[import-not-found]


class HintJaxArraySetter(Protocol):
    """Subset of JAX indexed-update objects used by carrier bases."""

    def set(self, value: Any) -> HintJaxArray: ...


class HintJaxArrayAt(Protocol):
    """Subset of JAX `.at[...]` update syntax used by carrier bases."""

    def __getitem__(self, index: Any) -> HintJaxArraySetter: ...


class HintJaxArray(Protocol):
    """Small structural type for the JAX array surface STARK uses."""

    @property
    def at(self) -> HintJaxArrayAt: ...

    @property
    def dtype(self) -> object: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    def reshape(self, shape: tuple[int, ...]) -> HintJaxArray: ...
    def __add__(self, other: Any) -> HintJaxArray: ...
    def __radd__(self, other: Any) -> HintJaxArray: ...
    def __mul__(self, other: Any) -> HintJaxArray: ...
    def __rmul__(self, other: Any) -> HintJaxArray: ...
    def __pow__(self, other: Any) -> HintJaxArray: ...
    def __getitem__(self, index: Any) -> Any: ...
    def __float__(self) -> float: ...


class HintJaxArrayTemplate(Protocol):
    """Structural template accepted when creating JAX carrier storage."""

    @property
    def dtype(self) -> object: ...

    @property
    def shape(self) -> tuple[int, ...]: ...


class HintJaxNumpyModule(Protocol):
    """Subset of the JAX NumPy module used by carrier storage."""

    def asarray(self, value: Any) -> HintJaxArray: ...


jax_numpy = cast(HintJaxNumpyModule, jnp)
CarrierJaxValue: TypeAlias = HintJaxArray


@dataclass(frozen=True)
class CarrierStorageJax:
    """Template metadata for JAX states and translations."""

    shape: tuple[int, ...]
    dtype: object

    @classmethod
    def from_template(cls, template: HintJaxArrayTemplate) -> "CarrierStorageJax":
        array = jax_numpy.asarray(template)
        return cls(shape=array.shape, dtype=array.dtype)

    def is_state(self, value: CarrierJaxValue) -> bool:
        return self.matches_template(value)

    def is_translation(self, value: CarrierJaxValue) -> bool:
        return self.matches_template(value)

    def matches_template(self, value: CarrierJaxValue) -> bool:
        return value.shape == self.shape
