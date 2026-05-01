from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, TypeVar


Value = TypeVar("Value")


class CarrierKernel(Protocol):
    """Policy for building arithmetic workers for carried values."""

    def bind(
        self,
        *,
        carrier: Any | None = None,
        template: Any | None = None,
        norm: Any,
    ) -> CarrierKernelBound:
        """Return a kernel specialised to a carrier/template/norm."""


class CarrierKernelBound(Protocol[Value]):
    """Arithmetic worker bound to a concrete carrier/template/norm."""

    def translate(self, origin: Value, delta: Value) -> Value:
        ...

    def add(self, left: Value, right: Value) -> Value:
        ...

    def scale(self, scalar: float, value: Value) -> Value:
        ...

    def combine(
        self,
        coefficients: Sequence[float],
        values: Sequence[Value],
    ) -> Value:
        ...

    def norm(self, value: Value) -> float:
        ...


@dataclass(frozen=True, slots=True)
class CarrierKernelNative:
    """Arithmetic kernel policy for native scalars, lists, and tuples."""

    def bind(
        self,
        *,
        carrier: Any | None = None,
        template: Any | None = None,
        norm: Any,
    ) -> CarrierKernelNativeBound:
        return CarrierKernelNativeBound(norm_policy=norm)


@dataclass(frozen=True, slots=True)
class CarrierKernelNativeBound:
    """Returning arithmetic worker for native scalars, lists, and tuples."""

    norm_policy: Any

    def translate(self, origin, delta):
        return self.add(origin, delta)

    def add(self, left, right):
        if isinstance(left, list):
            return [item_left + item_right for item_left, item_right in zip(left, right)]

        if isinstance(left, tuple):
            return tuple(
                item_left + item_right for item_left, item_right in zip(left, right)
            )

        return left + right

    def scale(self, scalar: float, value):
        if isinstance(value, list):
            return [scalar * item for item in value]

        if isinstance(value, tuple):
            return tuple(scalar * item for item in value)

        return scalar * value

    def combine(self, coefficients, values):
        if len(values) == 0:
            raise ValueError("combine requires at least one value")

        first = values[0]

        if isinstance(first, list):
            return [
                sum(
                    coefficient * value[index]
                    for coefficient, value in zip(coefficients, values)
                )
                for index in range(len(first))
            ]

        if isinstance(first, tuple):
            return tuple(
                sum(
                    coefficient * value[index]
                    for coefficient, value in zip(coefficients, values)
                )
                for index in range(len(first))
            )

        return sum(
            coefficient * value for coefficient, value in zip(coefficients, values)
        )

    def norm(self, value) -> float:
        return self.norm_policy(value)


@dataclass(frozen=True, slots=True)
class CarrierKernelNumpy:
    """Arithmetic kernel policy for NumPy arrays."""

    def bind(
        self,
        *,
        carrier: Any | None = None,
        template: Any | None = None,
        norm: Any,
    ) -> CarrierKernelNumpyBound:
        return CarrierKernelNumpyBound(norm_policy=norm)


@dataclass(frozen=True, slots=True)
class CarrierKernelNumpyBound:
    """Arithmetic worker for NumPy arrays.

    Returning methods are the base API. In-place methods are optional
    acceleration hooks for callers that can safely reuse output storage.
    """

    norm_policy: Any

    def translate(self, origin, delta):
        return origin + delta

    def add(self, left, right):
        return left + right

    def scale(self, scalar: float, value):
        return scalar * value

    def combine(self, coefficients, values):
        if len(values) == 0:
            raise ValueError("combine requires at least one value")

        result = coefficients[0] * values[0]

        for coefficient, value in zip(coefficients[1:], values[1:]):
            result = result + coefficient * value

        return result

    def norm(self, value) -> float:
        return self.norm_policy(value)

    def translate_into(self, result, origin, delta) -> None:
        result[...] = origin + delta

    def add_into(self, result, left, right) -> None:
        result[...] = left + right

    def scale_into(self, result, scalar: float, value) -> None:
        result[...] = scalar * value

    def combine_into(self, result, coefficients, values) -> None:
        if len(values) == 0:
            raise ValueError("combine_into requires at least one value")

        result[...] = coefficients[0] * values[0]

        for coefficient, value in zip(coefficients[1:], values[1:]):
            result[...] += coefficient * value