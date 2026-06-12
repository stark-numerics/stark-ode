"""Norm policies for JAX-backed carrier values."""

from dataclasses import dataclass
from typing import Protocol, cast

import jax.numpy as jnp  # type: ignore[import-not-found]

from stark.engines.carriers.jax.storage import CarrierJaxValue


class HintJaxNumpyModule(Protocol):
    """Subset of JAX NumPy reduction APIs used by carrier norms."""

    def abs(self, value: CarrierJaxValue) -> CarrierJaxValue: ...
    def max(self, value: CarrierJaxValue) -> CarrierJaxValue: ...
    def mean(self, value: CarrierJaxValue) -> CarrierJaxValue: ...
    def sqrt(self, value: CarrierJaxValue) -> CarrierJaxValue: ...


jax_numpy = cast(HintJaxNumpyModule, jnp)


@dataclass(frozen=True)
class CarrierNormJaxRMS:
    """Root-mean-square norm for JAX arrays."""

    def __call__(self, value: CarrierJaxValue) -> float:
        absolute = jax_numpy.abs(value)
        return float(jax_numpy.sqrt(jax_numpy.mean(absolute ** 2)))


@dataclass(frozen=True)
class CarrierNormJaxMax:
    """Maximum absolute-entry norm for JAX arrays."""

    def __call__(self, value: CarrierJaxValue) -> float:
        return float(jax_numpy.max(jax_numpy.abs(value)))
