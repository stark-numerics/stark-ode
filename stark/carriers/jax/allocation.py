from dataclasses import dataclass
from typing import Protocol, cast

import jax.numpy as jnp  # type: ignore[import-not-found]

from stark.carriers.jax.storage import CarrierJaxValue, CarrierStorageJax


class HintJaxNumpyModule(Protocol):
    def zeros(self, shape: tuple[int, ...], dtype: object) -> CarrierJaxValue: ...
    def array(self, value: CarrierJaxValue) -> CarrierJaxValue: ...


jax_numpy = cast(HintJaxNumpyModule, jnp)


@dataclass(frozen=True)
class CarrierAllocationJax:
    storage: CarrierStorageJax

    def zero_state(self) -> CarrierJaxValue:
        return jax_numpy.zeros(self.storage.shape, dtype=self.storage.dtype)

    def zero_translation(self) -> CarrierJaxValue:
        return self.zero_state()

    def allocate_translation(self) -> CarrierJaxValue:
        return self.zero_translation()

    def copy_state(self, value: CarrierJaxValue) -> CarrierJaxValue:
        return jax_numpy.array(value)

    def copy_translation(self, value: CarrierJaxValue) -> CarrierJaxValue:
        return self.copy_state(value)