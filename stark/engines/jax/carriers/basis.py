"""Coordinate bases for JAX-backed carrier values."""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass
from math import prod
from typing import Protocol, cast

import jax.numpy as jnp  # type: ignore[import-not-found]

from stark.engines.jax.carriers.storage import CarrierJaxValue, CarrierStorageJax


class HintJaxNumpyModule(Protocol):
    """Subset of JAX NumPy used by the basis helper."""

    def zeros(self, shape: tuple[int, ...], dtype: object) -> CarrierJaxValue: ...
    def asarray(self, value: Sequence[float], dtype: object) -> CarrierJaxValue: ...


jax_numpy = cast(HintJaxNumpyModule, jnp)


@dataclass(frozen=True)
class CarrierBasisJax:
    """Canonical coordinate basis for a JAX carrier."""

    storage: CarrierStorageJax

    @property
    def dimension(self) -> int:
        return prod(self.storage.shape)

    def vector(self, index: int, output: CarrierJaxValue) -> CarrierJaxValue:
        del output
        zero = jax_numpy.zeros(self.storage.shape, dtype=self.storage.dtype)
        flat = zero.reshape((-1,))
        return flat.at[index].set(1.0).reshape(self.storage.shape)

    def coordinate(self, index: int, translation: CarrierJaxValue) -> float:
        return float(translation.reshape((-1,))[index])

    def coordinates(
        self,
        translation: CarrierJaxValue,
        output: MutableSequence[float],
    ) -> MutableSequence[float]:
        flat = translation.reshape((-1,))
        for index in range(self.dimension):
            output[index] = float(flat[index])
        return output

    def synthesize(
        self,
        coordinates: Sequence[float],
        output: CarrierJaxValue,
    ) -> CarrierJaxValue:
        del output
        return jax_numpy.asarray(coordinates, dtype=self.storage.dtype).reshape(self.storage.shape)


__all__ = ["CarrierBasisJax"]
