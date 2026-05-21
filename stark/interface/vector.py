from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from stark.carriers.deprecated.core import DeprecatedCarrierBound
from stark.routing import RoutingVector, RoutingVectorReturn


@dataclass
class StarkVector:
    value: Any
    carrier: DeprecatedCarrierBound

    def translation(
        self,
        value: Any,
        routing: RoutingVector | None = None,
    ) -> "StarkVectorTranslation":
        """Return a translation over the same vector-space carrier."""

        if routing is None:
            return StarkVectorTranslation(value, self.carrier)

        return StarkVectorTranslation(value, self.carrier, routing)

    def zero_translation(
        self,
        routing: RoutingVector | None = None,
    ) -> "StarkVectorTranslation":
        """Return a zero translation over the same vector-space carrier."""

        return self.translation(self.carrier.zero_translation(), routing)

    def workbench(
        self,
        routing: RoutingVector | None = None,
    ) -> "StarkVectorWorkbench":
        """Return a workbench over the same vector-space carrier."""

        if routing is None:
            return StarkVectorWorkbench(self.carrier)

        return StarkVectorWorkbench(self.carrier, routing)


@dataclass
class StarkVectorTranslation:
    value: Any
    carrier: DeprecatedCarrierBound
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
        coefficient: float,
        value: "StarkVectorTranslation",
        out: "StarkVectorTranslation",
    ) -> "StarkVectorTranslation":
        self.routing.scale(self.carrier.kernel, coefficient, value, out)
        return out

    def combine2(self, *terms: object) -> "StarkVectorTranslation":
        return self.combine(*terms)

    def combine3(self, *terms: object) -> "StarkVectorTranslation":
        return self.combine(*terms)

    def combine4(self, *terms: object) -> "StarkVectorTranslation":
        return self.combine(*terms)

    def combine5(self, *terms: object) -> "StarkVectorTranslation":
        return self.combine(*terms)

    def combine6(self, *terms: object) -> "StarkVectorTranslation":
        return self.combine(*terms)

    def combine7(self, *terms: object) -> "StarkVectorTranslation":
        return self.combine(*terms)

    def combine8(self, *terms: object) -> "StarkVectorTranslation":
        return self.combine(*terms)

    def combine9(self, *terms: object) -> "StarkVectorTranslation":
        return self.combine(*terms)

    def combine10(self, *terms: object) -> "StarkVectorTranslation":
        return self.combine(*terms)

    def combine11(self, *terms: object) -> "StarkVectorTranslation":
        return self.combine(*terms)

    def combine12(self, *terms: object) -> "StarkVectorTranslation":
        return self.combine(*terms)

    def combine(self, *terms: object) -> "StarkVectorTranslation":
        if not terms:
            raise TypeError("Linear combination requires coefficient/translation pairs and an output.")
        out = terms[-1]
        terms = terms[:-1]
        if len(terms) % 2 != 0:
            raise TypeError("Linear combination terms must be coefficient/translation pairs.")
        if not isinstance(out, StarkVectorTranslation):
            raise TypeError("Linear combination output must be a StarkVectorTranslation.")

        coefficients = terms[0::2]
        values = terms[1::2]

        self.routing.combine(self.carrier.kernel, coefficients, values, out)
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
    carrier: DeprecatedCarrierBound
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
