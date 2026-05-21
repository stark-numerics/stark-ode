from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from .kernels import DeprecatedCarrierKernelNative, DeprecatedCarrierKernelNumpy
from .norms import DeprecatedCarrierNormNativeRMS, DeprecatedCarrierNormNumpyRMS
from stark.routing import RoutingVector, RoutingVectorPreferInPlace, RoutingVectorReturn

StateValue = TypeVar("StateValue")
TranslationValue = TypeVar("TranslationValue")

class DeprecatedCarrierError(ValueError):
    """Raised when carrier configuration or binding is invalid."""

class DeprecatedCarrier(Protocol[StateValue, TranslationValue]):
    """Storage policy for values carried through STARK."""

    def accepts(self, value: object) -> bool:
        ...

    def coerce_state(
        self,
        value: object,
        *,
        template: StateValue | None = None,
    ) -> StateValue:
        ...

    def bind(self, template: StateValue) -> DeprecatedCarrierBound[StateValue, TranslationValue]:
        ...

    def recommend_vector_routing(self) -> RoutingVector:
        ...


class DeprecatedCarrierBound(Protocol[StateValue, TranslationValue]):
    """DeprecatedCarrier bound to a concrete template value."""

    template: StateValue
    kernel: object

    def zero_state(self) -> StateValue:
        ...

    def zero_translation(self) -> TranslationValue:
        ...

    def copy_state(self, value: StateValue) -> StateValue:
        ...

    def copy_translation(self, value: TranslationValue) -> TranslationValue:
        ...

    def coerce_translation(self, value: object) -> TranslationValue:
        ...

    def validate_state(self, value: StateValue) -> None:
        ...

    def validate_translation(self, value: TranslationValue) -> None:
        ...


@dataclass(frozen=True, slots=True)
class DeprecatedCarrierNative:
    """DeprecatedCarrier for Python scalars, lists, and tuples."""

    norm: object = DeprecatedCarrierNormNativeRMS()
    kernel: object = DeprecatedCarrierKernelNative()

    def accepts(self, value: object) -> bool:
        if isinstance(value, (int, float)):
            return True

        if isinstance(value, (list, tuple)):
            return all(isinstance(item, (int, float)) for item in value)

        return False

    def coerce_state(
        self,
        value: object,
        *,
        template: object | None = None,
    ):
        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, list):
            return [float(item) for item in value]

        if isinstance(value, tuple):
            return tuple(float(item) for item in value)

        raise TypeError(f"DeprecatedCarrierNative cannot represent {type(value).__name__}")

    def bind(self, template) -> DeprecatedCarrierNativeBound:
        template = self.coerce_state(template)

        bound_norm = self.norm.bind(
            carrier=self,
            template=template,
        )

        bound_kernel = self.kernel.bind(
            carrier=self,
            template=template,
            norm=bound_norm,
        )

        return DeprecatedCarrierNativeBound(
            carrier=self,
            template=template,
            kernel=bound_kernel,
        )
    
    def recommend_vector_routing(self) -> RoutingVector:
        return RoutingVectorReturn()


@dataclass(frozen=True, slots=True)
class DeprecatedCarrierNativeBound:
    """Native carrier bound to a scalar/list/tuple template."""

    carrier: DeprecatedCarrierNative
    template: object
    kernel: object

    def zero_state(self):
        if isinstance(self.template, list):
            return [0.0 for _ in self.template]

        if isinstance(self.template, tuple):
            return tuple(0.0 for _ in self.template)

        return 0.0

    def zero_translation(self):
        return self.zero_state()

    def copy_state(self, value):
        if isinstance(value, list):
            return list(value)

        if isinstance(value, tuple):
            return tuple(value)

        return value

    def copy_translation(self, value):
        return self.copy_state(value)

    def coerce_translation(self, value):
        coerced = self.carrier.coerce_state(value, template=self.template)
        self.validate_translation(coerced)
        return coerced

    def validate_state(self, value) -> None:
        if isinstance(self.template, list):
            if not isinstance(value, list):
                raise TypeError(f"Expected list, got {type(value).__name__}")

            if len(value) != len(self.template):
                raise ValueError(
                    f"Expected length {len(self.template)}, got {len(value)}"
                )

            return

        if isinstance(self.template, tuple):
            if not isinstance(value, tuple):
                raise TypeError(f"Expected tuple, got {type(value).__name__}")

            if len(value) != len(self.template):
                raise ValueError(
                    f"Expected length {len(self.template)}, got {len(value)}"
                )

            return

        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected scalar, got {type(value).__name__}")

    def validate_translation(self, value) -> None:
        self.validate_state(value)


@dataclass(frozen=True, slots=True)
class DeprecatedCarrierNumpy:
    """DeprecatedCarrier for NumPy arrays."""

    dtype: Any | None = None
    copy_inputs: bool = True
    copy_outputs: bool = False
    strict_shape: bool = True
    strict_dtype: bool = False
    norm: object = DeprecatedCarrierNormNumpyRMS()
    kernel: object = DeprecatedCarrierKernelNumpy()

    def _np(self):
        import numpy as np

        return np

    def accepts(self, value: object) -> bool:
        np = self._np()
        return isinstance(value, np.ndarray)

    def coerce_state(
        self,
        value: object,
        *,
        template: object | None = None,
    ):
        np = self._np()

        dtype = self.dtype
        if dtype is None and template is not None and hasattr(template, "dtype"):
            dtype = template.dtype

        return np.array(
            value,
            dtype=dtype,
            copy=self.copy_inputs,
        )

    def bind(self, template) -> DeprecatedCarrierNumpyBound:
        template = self.coerce_state(template)

        bound_norm = self.norm.bind(
            carrier=self,
            template=template,
        )

        bound_kernel = self.kernel.bind(
            carrier=self,
            template=template,
            norm=bound_norm,
        )

        return DeprecatedCarrierNumpyBound(
            carrier=self,
            template=template,
            kernel=bound_kernel,
        )
    
    def recommend_vector_routing(self) -> RoutingVector:
        return RoutingVectorPreferInPlace()


@dataclass(frozen=True, slots=True)
class DeprecatedCarrierNumpyBound:
    """NumPy carrier bound to an ndarray template."""

    carrier: DeprecatedCarrierNumpy
    template: object
    kernel: object

    def zero_state(self):
        np = self.carrier._np()

        return np.zeros_like(
            self.template,
            dtype=(
                self.carrier.dtype
                if self.carrier.dtype is not None
                else self.template.dtype
            ),
        )

    def zero_translation(self):
        return self.zero_state()

    def copy_state(self, value):
        np = self.carrier._np()
        return np.array(value, copy=True)

    def copy_translation(self, value):
        return self.copy_state(value)

    def coerce_translation(self, value):
        np = self.carrier._np()

        dtype = self.carrier.dtype
        if dtype is None:
            dtype = self.template.dtype

        coerced = np.asarray(value, dtype=dtype)

        if self.carrier.copy_outputs:
            coerced = np.array(coerced, dtype=dtype, copy=True)

        self.validate_translation(coerced)
        return coerced

    def validate_state(self, value) -> None:
        if self.carrier.strict_shape and value.shape != self.template.shape:
            raise ValueError(f"Expected shape {self.template.shape}, got {value.shape}")

        if self.carrier.strict_dtype and value.dtype != self.template.dtype:
            raise TypeError(f"Expected dtype {self.template.dtype}, got {value.dtype}")

    def validate_translation(self, value) -> None:
        self.validate_state(value)