from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp

from stark.routing import RoutingVector, RoutingVectorReturn


@dataclass(frozen=True, slots=True)
class CarrierNormJaxRMS:
    """Root-mean-square norm policy for JAX arrays."""

    def bind(
        self,
        *,
        template: Any,
        carrier: Any | None = None,
    ) -> "CarrierNormJaxRMSBound":
        del template, carrier
        return CarrierNormJaxRMSBound()


@dataclass(frozen=True, slots=True)
class CarrierNormJaxRMSBound:
    """Root-mean-square norm for JAX arrays."""

    def __call__(self, value: Any) -> float:
        if value.size == 0:
            return 0.0

        rms = jnp.sqrt(jnp.mean(jnp.abs(value) ** 2))
        return float(rms)


@dataclass(frozen=True, slots=True)
class CarrierNormJaxMax:
    """Maximum absolute-entry norm policy for JAX arrays."""

    def bind(
        self,
        *,
        template: Any,
        carrier: Any | None = None,
    ) -> "CarrierNormJaxMaxBound":
        del template, carrier
        return CarrierNormJaxMaxBound()


@dataclass(frozen=True, slots=True)
class CarrierNormJaxMaxBound:
    """Maximum absolute-entry norm for JAX arrays."""

    def __call__(self, value: Any) -> float:
        if value.size == 0:
            return 0.0

        maximum = jnp.max(jnp.abs(value))
        return float(maximum)


@dataclass(frozen=True, slots=True)
class CarrierKernelJax:
    """Arithmetic kernel policy for JAX arrays."""

    def bind(
        self,
        *,
        carrier: Any | None = None,
        template: Any | None = None,
        norm: Any,
    ) -> "CarrierKernelJaxBound":
        del carrier, template
        return CarrierKernelJaxBound(norm_policy=norm)


@dataclass(frozen=True, slots=True)
class CarrierKernelJaxBound:
    """Arithmetic worker for JAX arrays."""

    norm_policy: Any

    def translate(self, origin: Any, delta: Any) -> Any:
        return origin + delta

    def add(self, left: Any, right: Any) -> Any:
        return left + right

    def scale(self, scalar: float, value: Any) -> Any:
        return scalar * value

    def combine(self, coefficients: Any, values: Any) -> Any:
        if not values:
            raise ValueError("Cannot combine empty values.")

        result = coefficients[0] * values[0]

        for coefficient, value in zip(coefficients[1:], values[1:]):
            result = result + coefficient * value

        return result

    def norm(self, value: Any) -> float:
        return self.norm_policy(value)


@dataclass(frozen=True, slots=True)
class CarrierJax:
    """Carrier for JAX arrays.

    JAX arrays use return/replacement routing. `copy_inputs` and `copy_outputs`
    are retained for carrier API symmetry, but JAX coercion is based on
    `jax.numpy.asarray` and does not promise a physical copy.
    """

    dtype: Any = None
    copy_inputs: bool = True
    copy_outputs: bool = True
    strict_shape: bool = True
    strict_dtype: bool = False
    norm: Any = field(default_factory=CarrierNormJaxRMS)
    kernel: Any = field(default_factory=CarrierKernelJax)

    def accepts(self, value: Any) -> bool:
        return isinstance(value, jax.Array)

    def coerce_state(self, value: Any, template: Any = None) -> Any:
        return jnp.asarray(value, dtype=self.dtype)

    def bind(self, template: Any) -> "CarrierJaxBound":
        template = self.coerce_state(template)

        bound_norm = self.norm.bind(
            template=template,
            carrier=self,
        )

        kernel = self.kernel.bind(
            template=template,
            carrier=self,
            norm=bound_norm,
        )

        return CarrierJaxBound(
            carrier=self,
            template=template,
            kernel=kernel,
        )

    def recommend_vector_routing(self) -> RoutingVector:
        return RoutingVectorReturn()


@dataclass(frozen=True, slots=True)
class CarrierJaxBound:
    """JAX carrier bound to an array template."""

    carrier: CarrierJax
    template: Any
    kernel: CarrierKernelJaxBound

    def zero_state(self) -> Any:
        return jnp.zeros_like(self.template)

    def zero_translation(self) -> Any:
        return jnp.zeros_like(self.template)

    def copy_state(self, value: Any) -> Any:
        return jnp.asarray(value)

    def copy_translation(self, value: Any) -> Any:
        return jnp.asarray(value)

    def coerce_translation(self, value: Any) -> Any:
        value = jnp.asarray(value)

        self.validate_translation(value)

        return value

    def validate_state(self, value: Any) -> None:
        if not isinstance(value, jax.Array):
            raise TypeError("CarrierJax state must be a jax.Array.")

        self._validate_array(value)

    def validate_translation(self, value: Any) -> None:
        if not isinstance(value, jax.Array):
            raise TypeError("CarrierJax translation must be a jax.Array.")

        self._validate_array(value)

    def _validate_array(self, value: Any) -> None:
        if self.carrier.strict_shape and value.shape != self.template.shape:
            raise ValueError(
                f"JAX value has shape {value.shape}, expected "
                f"{self.template.shape}."
            )

        if self.carrier.strict_dtype and value.dtype != self.template.dtype:
            raise TypeError(
                f"JAX value has dtype {value.dtype}, expected "
                f"{self.template.dtype}."
            )
