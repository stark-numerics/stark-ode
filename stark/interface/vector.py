from dataclasses import dataclass, field
from typing import Any

from stark.carriers.core import CarrierBound
from stark.routing import RoutingVector, RoutingVectorReturn


@dataclass
class StarkVector:
    """State wrapper for ordinary vector-space IVPs.

    In the simple vector-space interface, both the state and the increment live
    in the same mathematical vector space. STARK still represents them with
    separate wrapper classes so the general state/translation contract remains
    explicit.
    """
    value: Any
    carrier: CarrierBound


@dataclass
class StarkVectorTranslation:
    """Increment wrapper for ordinary vector-space IVPs.

    This is a vector-space increment: it supports scaling, linear combination,
    addition, and application to a `StarkVector` state.
    """
    value: Any
    carrier: CarrierBound
    routing: RoutingVector = field(default_factory=RoutingVectorReturn)

    def __call__(self, origin: StarkVector, result: StarkVector) -> StarkVector:
        self.routing.translate(self.carrier.kernel, result, origin, self)
        return result

    def norm(self) -> float:
        return self.carrier.kernel.norm(self.value)

    @property
    def linear_combine(self) -> tuple:
        return (
            self.scale,
            self.combine2,
            self.combine3,
            self.combine4,
            self.combine5,
            self.combine6,
            self.combine7,
            self.combine8,
            self.combine9,
            self.combine10,
            self.combine11,
            self.combine12,
        )

    def scale(
        self,
        out: "StarkVectorTranslation",
        coefficient: float,
        value: "StarkVectorTranslation",
    ) -> "StarkVectorTranslation":
        self.routing.scale(self.carrier.kernel, out, coefficient, value)
        return out

    def combine2(self, out: "StarkVectorTranslation", *terms: object) -> "StarkVectorTranslation":
        return self.combine(out, *terms)

    def combine3(self, out: "StarkVectorTranslation", *terms: object) -> "StarkVectorTranslation":
        return self.combine(out, *terms)

    def combine4(self, out: "StarkVectorTranslation", *terms: object) -> "StarkVectorTranslation":
        return self.combine(out, *terms)

    def combine5(self, out: "StarkVectorTranslation", *terms: object) -> "StarkVectorTranslation":
        return self.combine(out, *terms)

    def combine6(self, out: "StarkVectorTranslation", *terms: object) -> "StarkVectorTranslation":
        return self.combine(out, *terms)

    def combine7(self, out: "StarkVectorTranslation", *terms: object) -> "StarkVectorTranslation":
        return self.combine(out, *terms)

    def combine8(self, out: "StarkVectorTranslation", *terms: object) -> "StarkVectorTranslation":
        return self.combine(out, *terms)

    def combine9(self, out: "StarkVectorTranslation", *terms: object) -> "StarkVectorTranslation":
        return self.combine(out, *terms)

    def combine10(self, out: "StarkVectorTranslation", *terms: object) -> "StarkVectorTranslation":
        return self.combine(out, *terms)

    def combine11(self, out: "StarkVectorTranslation", *terms: object) -> "StarkVectorTranslation":
        return self.combine(out, *terms)

    def combine12(self, out: "StarkVectorTranslation", *terms: object) -> "StarkVectorTranslation":
        return self.combine(out, *terms)

    def combine(self, out: "StarkVectorTranslation", *terms: object) -> "StarkVectorTranslation":
        if len(terms) % 2 != 0:
            raise TypeError("Linear combination terms must be coefficient/translation pairs.")

        coefficients = terms[0::2]
        values = terms[1::2]
        self.routing.combine(self.carrier.kernel, out, coefficients, values)
        return out

    def __add__(self, other: "StarkVectorTranslation") -> "StarkVectorTranslation":
        if self.carrier is not other.carrier:
            raise ValueError("Cannot add translations with different carriers.")

        value = self.carrier.kernel.add(self.value, other.value)
        return StarkVectorTranslation(value, self.carrier, self.routing)

    def __rmul__(self, scalar: float) -> "StarkVectorTranslation":
        value = self.carrier.kernel.scale(scalar, self.value)
        return StarkVectorTranslation(value, self.carrier, self.routing)


@dataclass
class StarkVectorWorkbench:
    carrier: CarrierBound
    routing: RoutingVector = field(default_factory=RoutingVectorReturn)

    def allocate_state(self) -> StarkVector:
        return StarkVector(self.carrier.zero_state(), self.carrier)

    def copy_state(self, result: StarkVector, source: StarkVector) -> StarkVector:
        result.value = self.carrier.copy_state(source.value)
        return result

    def allocate_translation(self) -> StarkVectorTranslation:
        return StarkVectorTranslation(
            self.carrier.zero_translation(),
            self.carrier,
            self.routing,
        )
