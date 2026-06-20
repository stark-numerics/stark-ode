"""Validation and coercion for JAX-backed carrier values."""

from dataclasses import dataclass
from typing import Protocol, cast

import jax.numpy as jnp  # type: ignore[import-not-found]

from stark.engines.jax.carriers.storage import CarrierJaxValue, CarrierStorageJax


class HintJaxNumpyModule(Protocol):
    """Subset of JAX NumPy validation APIs used by this carrier."""

    def asarray(self, value: CarrierJaxValue) -> CarrierJaxValue: ...


jax_numpy = cast(HintJaxNumpyModule, jnp)


@dataclass(frozen=True)
class CarrierValidationJax:
    """Validate that JAX values match the carrier template."""

    storage: CarrierStorageJax

    def validate_state(self, value: CarrierJaxValue) -> CarrierJaxValue:
        return self.validate_array(value, "state")

    def validate_translation(self, value: CarrierJaxValue) -> CarrierJaxValue:
        return self.validate_array(value, "translation")

    def coerce_translation(self, value: CarrierJaxValue) -> CarrierJaxValue:
        return self.validate_translation(jax_numpy.asarray(value))

    def validate_array(self, value: CarrierJaxValue, role: str) -> CarrierJaxValue:
        if not self.storage.matches_template(value):
            raise ValueError(f"JAX carrier {role} shape does not match template.")

        return value
