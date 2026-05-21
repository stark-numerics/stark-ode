from __future__ import annotations

from dataclasses import dataclass
from stark.carriers.deprecated.core import DeprecatedCarrierNative, DeprecatedCarrierNumpy

@dataclass(frozen=True, slots=True)
class DeprecatedCarrierLibrary:
    """Ordered collection of carrier policies used to select carriers for values."""

    carriers: tuple[object, ...]

    @classmethod
    def default(cls) -> "DeprecatedCarrierLibrary":
        carriers = []

        try:
            from stark.carriers.deprecated.cupy import DeprecatedCarrierCuPy
        except ImportError:
            pass
        else:
            carriers.append(DeprecatedCarrierCuPy())

        try:
            from stark.carriers.deprecated.jax import DeprecatedCarrierJax
        except ImportError:
            pass
        else:
            carriers.append(DeprecatedCarrierJax())

        carriers.extend(
            [
                DeprecatedCarrierNumpy(),
                DeprecatedCarrierNative(),
            ]
        )

        return cls(carriers)

    def carrier_for(self, value: object):
        for carrier in self.carriers:
            if carrier.accepts(value):
                return carrier

        raise TypeError(
            f"Could not find a STARK carrier for {type(value).__name__}. "
            "Pass a carrier explicitly or provide a DeprecatedCarrierLibrary containing "
            "a compatible carrier."
        )

    def with_carrier(self, carrier: object) -> "DeprecatedCarrierLibrary":
        return type(self)((carrier, *self.carriers))

    def without_carrier_type(self, carrier_type: type) -> "DeprecatedCarrierLibrary":
        return type(self)(
            tuple(
                carrier
                for carrier in self.carriers
                if not isinstance(carrier, carrier_type)
            )
        )