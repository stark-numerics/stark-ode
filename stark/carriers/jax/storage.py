from dataclasses import dataclass
from typing import Protocol, TypeAlias, cast

import jax.numpy as jnp # type: ignore[import-not-found]


class HintJaxArray(Protocol):
    shape: tuple[int, ...]
    dtype: object

class HintJaxNumpyModule(Protocol):
    def asarray(self, value: HintJaxArray) -> HintJaxArray: ...

jax_numpy = cast(HintJaxNumpyModule, jnp)
CarrierJaxValue: TypeAlias = HintJaxArray

@dataclass(frozen=True)
class CarrierStorageJax:
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