from __future__ import annotations

from typing import Any, cast

import jax  # type: ignore[import-not-found]
import jax.numpy as jnp  # type: ignore[import-not-found]

from stark.engines.carrier_jax.allocation import CarrierAllocationJax
from stark.engines.carrier_jax.basis import CarrierBasisJax
from stark.engines.carrier_jax.arithmetic import CarrierArithmeticJax
from stark.engines.carrier_jax.norm import CarrierNormJaxRMS
from stark.engines.carrier_jax.storage import CarrierJaxValue, CarrierStorageJax
from stark.engines.carrier_jax.validation import CarrierValidationJax
from stark.engines.carriers import CarrierScalarItem


class CarrierJax:
    @classmethod
    def from_shape(
        cls,
        shape: tuple[int, ...],
        dtype: object | None = None,
    ) -> "CarrierJax":
        resolved_dtype = cls.resolve_dtype(dtype)
        return cls(
            cast(
                CarrierJaxValue,
                jnp.zeros(shape, dtype=cast(Any, resolved_dtype)),
            )
        )

    @staticmethod
    def resolve_dtype(dtype: object | None) -> object:
        if dtype is None:
            if CarrierJax.x64_enabled():
                return jnp.dtype(jnp.float64)
            return jnp.dtype(jnp.float32)

        requested = jnp.dtype(cast(Any, dtype))

        if CarrierJax.x64_enabled():
            return requested

        if requested == jnp.dtype(jnp.float64):
            raise ValueError(
                "CarrierJax dtype jnp.float64 requires JAX x64 support. "
                "Enable it with JAX_ENABLE_X64=1 or "
                "jax.config.update('jax_enable_x64', True), "
                "or pass dtype=jnp.float32."
            )

        if requested == jnp.dtype(jnp.complex128):
            raise ValueError(
                "CarrierJax dtype jnp.complex128 requires JAX x64 support. "
                "Enable it with JAX_ENABLE_X64=1 or "
                "jax.config.update('jax_enable_x64', True), "
                "or pass dtype=jnp.complex64."
            )

        return requested

    @staticmethod
    def x64_enabled() -> bool:
        return bool(getattr(jax.config, "jax_enable_x64", False))

    def __init__(self, template: CarrierJaxValue) -> None:
        storage = CarrierStorageJax.from_template(template)

        self.storage = storage
        self.validation = CarrierValidationJax(storage)
        self.allocation = CarrierAllocationJax(storage)
        self.basis = CarrierBasisJax(storage)
        self.arithmetic = CarrierArithmeticJax()
        self.norm = CarrierNormJaxRMS()
        self.scalar = CarrierScalarItem()
