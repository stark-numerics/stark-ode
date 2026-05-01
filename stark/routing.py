from dataclasses import dataclass, field
from typing import Any


class RoutingPolicy:
    pass


class RoutingVector(RoutingPolicy):
    def translate(self, kernel: Any, result: Any, origin: Any, delta: Any) -> Any:
        raise NotImplementedError

    def add(self, kernel: Any, result: Any, left: Any, right: Any) -> Any:
        raise NotImplementedError

    def scale(self, kernel: Any, result: Any, scalar: float, value: Any) -> Any:
        raise NotImplementedError

    def combine(
        self,
        kernel: Any,
        result: Any,
        coefficients: Any,
        values: Any,
    ) -> Any:
        raise NotImplementedError


class RoutingVectorReturn(RoutingVector):
    def translate(self, kernel, result, origin, delta):
        result.value = kernel.translate(origin.value, delta.value)
        return result

    def add(self, kernel, result, left, right):
        result.value = kernel.add(left.value, right.value)
        return result

    def scale(self, kernel, result, scalar, value):
        result.value = kernel.scale(scalar, value.value)
        return result

    def combine(self, kernel, result, coefficients, values):
        result.value = kernel.combine(coefficients, [value.value for value in values])
        return result


class RoutingVectorInPlace(RoutingVector):
    def translate(self, kernel, result, origin, delta):
        kernel.translate_into(result.value, origin.value, delta.value)
        return result

    def add(self, kernel, result, left, right):
        kernel.add_into(result.value, left.value, right.value)
        return result

    def scale(self, kernel, result, scalar, value):
        kernel.scale_into(result.value, scalar, value.value)
        return result

    def combine(self, kernel, result, coefficients, values):
        kernel.combine_into(
            result.value,
            coefficients,
            [value.value for value in values],
        )
        return result


class RoutingVectorPreferInPlace(RoutingVector):
    def translate(self, kernel, result, origin, delta):
        if hasattr(kernel, "translate_into"):
            kernel.translate_into(result.value, origin.value, delta.value)
        else:
            result.value = kernel.translate(origin.value, delta.value)
        return result

    def add(self, kernel, result, left, right):
        if hasattr(kernel, "add_into"):
            kernel.add_into(result.value, left.value, right.value)
        else:
            result.value = kernel.add(left.value, right.value)
        return result

    def scale(self, kernel, result, scalar, value):
        if hasattr(kernel, "scale_into"):
            kernel.scale_into(result.value, scalar, value.value)
        else:
            result.value = kernel.scale(scalar, value.value)
        return result

    def combine(self, kernel, result, coefficients, values):
        raw_values = [value.value for value in values]

        if hasattr(kernel, "combine_into"):
            kernel.combine_into(result.value, coefficients, raw_values)
        else:
            result.value = kernel.combine(coefficients, raw_values)

        return result


@dataclass
class Routing:
    vector: RoutingVector = field(default_factory=RoutingVectorReturn)

    @classmethod
    def default(cls) -> "Routing":
        return cls()

    def policy(self, family: type[RoutingPolicy]) -> RoutingPolicy:
        if family is RoutingVector:
            return self.vector

        raise KeyError(f"No routing policy for family {family!r}")
