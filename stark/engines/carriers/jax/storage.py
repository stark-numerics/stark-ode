"""Storage checks for JAX-backed carrier values."""

from dataclasses import dataclass
from typing import Protocol, TypeAlias, cast

import jax.numpy as jnp  # type: ignore[import-not-found]


class HintJaxArray(Protocol):
    """Small structural type for the JAX array attributes STARK uses."""

    shape: tuple[int, ...]
    dtype: object


class HintJaxNumpyModule(Protocol):
    """Subset of the JAX NumPy module used by carrier storage."""

    def asarray(self, value: HintJaxArray) -> HintJaxArray: ...


jax_numpy = cast(HintJaxNumpyModule, jnp)
CarrierJaxValue: TypeAlias = HintJaxArray


@dataclass(frozen=True)
class CarrierStorageJax:
    """Template metadata for JAX states and translations."""

    shape: tuple[int, ...]
    dtype: object

    @classmethod
    def from_template(cls, template: CarrierJaxValue) -> "CarrierStorageJax":
        array = jax_numpy.asarray(template)
        return cls(shape=array.shape, dtype=array.dtype)

    def is_state(self, value: CarrierJaxValue) -> bool:
        return self.matches_template(value)

    def is_translation(self, value: CarrierJaxValue) -> bool:
        return self.matches_template(value)

    def matches_template(self, value: CarrierJaxValue) -> bool:
        return value.shape == self.shape
