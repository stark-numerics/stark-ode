from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cupy as cp

from stark.routing import RoutingVector, RoutingVectorPreferInPlace


@dataclass(frozen=True, slots=True)
class CarrierNormCuPyRMS:
    """Root-mean-square norm policy for CuPy arrays."""

    def bind(
        self,
        *,
        template: Any,
        carrier: Any | None = None,
    ) -> "CarrierNormCuPyRMSBound":
        del template, carrier
        return CarrierNormCuPyRMSBound()


@dataclass(frozen=True, slots=True)
class CarrierNormCuPyRMSBound:
    """Root-mean-square norm for CuPy arrays."""

    def __call__(self, value: Any) -> float:
        if value.size == 0:
            return 0.0

        rms = cp.sqrt(cp.mean(cp.abs(value) ** 2))
        return float(rms.get())


@dataclass(frozen=True, slots=True)
class CarrierNormCuPyMax:
    """Maximum absolute-entry norm policy for CuPy arrays."""

    def bind(
        self,
        *,
        template: Any,
        carrier: Any | None = None,
    ) -> "CarrierNormCuPyMaxBound":
        del template, carrier
        return CarrierNormCuPyMaxBound()


@dataclass(frozen=True, slots=True)
class CarrierNormCuPyMaxBound:
    """Maximum absolute-entry norm for CuPy arrays."""

    def __call__(self, value: Any) -> float:
        if value.size == 0:
            return 0.0

        maximum = cp.max(cp.abs(value))
        return float(maximum.get())


@dataclass(frozen=True, slots=True)
class CarrierKernelCuPy:
    """Arithmetic kernel policy for CuPy arrays."""

    def bind(
        self,
        *,
        carrier: Any | None = None,
        template: Any | None = None,
        norm: Any,
    ) -> "CarrierKernelCuPyBound":
        del carrier, template
        return CarrierKernelCuPyBound(norm_policy=norm)


@dataclass(frozen=True, slots=True)
class CarrierKernelCuPyBound:
    """Arithmetic worker for CuPy arrays."""

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

    def translate_into(self, result: Any, origin: Any, delta: Any) -> None:
        result[...] = origin + delta

    def add_into(self, result: Any, left: Any, right: Any) -> None:
        result[...] = left + right

    def scale_into(self, result: Any, scalar: float, value: Any) -> None:
        result[...] = scalar * value

    def combine_into(self, result: Any, coefficients: Any, values: Any) -> None:
        if not values:
            raise ValueError("Cannot combine empty values.")

        result[...] = coefficients[0] * values[0]

        for coefficient, value in zip(coefficients[1:], values[1:]):
            result[...] = result + coefficient * value


@dataclass(frozen=True, slots=True)
class CarrierCuPy:
    """Carrier for CuPy arrays."""

    dtype: Any = None
    copy_inputs: bool = True
    copy_outputs: bool = True
    strict_shape: bool = True
    strict_dtype: bool = False
    norm: Any = field(default_factory=CarrierNormCuPyRMS)
    kernel: Any = field(default_factory=CarrierKernelCuPy)

    def accepts(self, value: Any) -> bool:
        return isinstance(value, cp.ndarray)

    def coerce_state(self, value: Any, template: Any = None) -> Any:
        return cp.array(value, dtype=self.dtype, copy=self.copy_inputs)

    def bind(self, template: Any) -> "CarrierCuPyBound":
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

        return CarrierCuPyBound(
            carrier=self,
            template=template,
            kernel=kernel,
        )

    def recommend_vector_routing(self) -> RoutingVector:
        return RoutingVectorPreferInPlace()


@dataclass(frozen=True, slots=True)
class CarrierCuPyBound:
    """CuPy carrier bound to an ndarray template."""

    carrier: CarrierCuPy
    template: Any
    kernel: CarrierKernelCuPyBound

    def zero_state(self) -> Any:
        return cp.zeros_like(self.template)

    def zero_translation(self) -> Any:
        return cp.zeros_like(self.template)

    def copy_state(self, value: Any) -> Any:
        if self.carrier.copy_outputs:
            return cp.array(value, copy=True)

        return value

    def copy_translation(self, value: Any) -> Any:
        if self.carrier.copy_outputs:
            return cp.array(value, copy=True)

        return value

    def coerce_translation(self, value: Any) -> Any:
        value = cp.asarray(value)

        self.validate_translation(value)

        return value

    def validate_state(self, value: Any) -> None:
        if not isinstance(value, cp.ndarray):
            raise TypeError("CarrierCuPy state must be a cupy.ndarray.")

        self._validate_array(value)

    def validate_translation(self, value: Any) -> None:
        if not isinstance(value, cp.ndarray):
            raise TypeError("CarrierCuPy translation must be a cupy.ndarray.")

        self._validate_array(value)

    def _validate_array(self, value: Any) -> None:
        if self.carrier.strict_shape and value.shape != self.template.shape:
            raise ValueError(
                f"CuPy value has shape {value.shape}, expected "
                f"{self.template.shape}."
            )

        if self.carrier.strict_dtype and value.dtype != self.template.dtype:
            raise TypeError(
                f"CuPy value has dtype {value.dtype}, expected "
                f"{self.template.dtype}."
            )
