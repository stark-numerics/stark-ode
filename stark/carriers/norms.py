from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, TypeVar


Value = TypeVar("Value")


class CarrierNorm(Protocol[Value]):
    """Policy for measuring carried translation values."""

    def bind(
        self,
        *,
        template: Value,
        carrier: Any | None = None,
    ) -> Callable[[Value], float]:
        """Return a hot-path norm callable specialised to a template."""


@dataclass(frozen=True, slots=True)
class CarrierNormZero:
    """Norm for zero-sized carried values."""

    def __call__(self, value: Any) -> float:
        return 0.0


@dataclass(frozen=True, slots=True)
class CarrierNormNativeScalarAbs:
    """Absolute-value norm for native scalar values."""

    def __call__(self, value: int | float) -> float:
        return abs(float(value))


@dataclass(frozen=True, slots=True)
class CarrierNormNativeSequenceRMS:
    """Root-mean-square norm for native sequence values."""

    def __call__(self, value: list[float] | tuple[float, ...]) -> float:
        import math

        return math.sqrt(sum(float(item) * float(item) for item in value) / len(value))


@dataclass(frozen=True, slots=True)
class CarrierNormNativeRMS:
    """Root-mean-square norm policy for native scalar/list/tuple values."""

    def bind(
        self,
        *,
        template: int | float | list[float] | tuple[float, ...],
        carrier: Any | None = None,
    ) -> Callable[[Any], float]:
        if isinstance(template, (list, tuple)):
            if len(template) == 0:
                return CarrierNormZero()

            return CarrierNormNativeSequenceRMS()

        return CarrierNormNativeScalarAbs()

    def __call__(self, value: int | float | list[float] | tuple[float, ...]) -> float:
        return self.bind(template=value)(value)


@dataclass(frozen=True, slots=True)
class CarrierNormNumpyRMSReal:
    """Root-mean-square norm for real NumPy arrays."""

    def __call__(self, value: Any) -> float:
        import numpy as np

        return float(np.sqrt(np.mean(value * value)))


@dataclass(frozen=True, slots=True)
class CarrierNormNumpyRMSComplex:
    """Root-mean-square norm for complex NumPy arrays."""

    def __call__(self, value: Any) -> float:
        import numpy as np

        return float(np.sqrt(np.mean(np.abs(value) ** 2)))


@dataclass(frozen=True, slots=True)
class CarrierNormNumpyRMS:
    """Root-mean-square norm policy for NumPy arrays."""

    def bind(
        self,
        *,
        template: Any,
        carrier: Any | None = None,
    ) -> Callable[[Any], float]:
        import numpy as np

        if template.size == 0:
            return CarrierNormZero()

        if np.iscomplexobj(template):
            return CarrierNormNumpyRMSComplex()

        return CarrierNormNumpyRMSReal()

    def __call__(self, value: Any) -> float:
        return self.bind(template=value)(value)


@dataclass(frozen=True, slots=True)
class CarrierNormNumpyMaxBound:
    """Maximum absolute-entry norm for non-empty NumPy arrays."""

    def __call__(self, value: Any) -> float:
        import numpy as np

        return float(np.max(np.abs(value)))


@dataclass(frozen=True, slots=True)
class CarrierNormNumpyMax:
    """Maximum absolute-entry norm policy for NumPy arrays."""

    def bind(
        self,
        *,
        template: Any,
        carrier: Any | None = None,
    ) -> Callable[[Any], float]:
        if template.size == 0:
            return CarrierNormZero()

        return CarrierNormNumpyMaxBound()

    def __call__(self, value: Any) -> float:
        return self.bind(template=value)(value)