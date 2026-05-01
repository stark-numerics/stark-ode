from __future__ import annotations

from dataclasses import dataclass
from stark.carriers.core import CarrierNative, CarrierNumpy

@dataclass(frozen=True, slots=True)
class CarrierLibrary:
    """Ordered collection of carrier policies used to select carriers for values."""

    carriers: tuple[object, ...]

    @classmethod
    def default(cls) -> "CarrierLibrary":
        carriers = []

        try:
            from stark.carriers.cupy import CarrierCuPy
        except ImportError:
            pass
        else:
            carriers.append(CarrierCuPy())

        try:
            from stark.carriers.jax import CarrierJax
        except ImportError:
            pass
        else:
            carriers.append(CarrierJax())

        carriers.extend(
            [
                CarrierNumpy(),
                CarrierNative(),
            ]
        )

        return cls(carriers)

    def carrier_for(self, value: object):
        for carrier in self.carriers:
            if carrier.accepts(value):
                return carrier

        raise TypeError(
            f"Could not find a STARK carrier for {type(value).__name__}. "
            "Pass a carrier explicitly or provide a CarrierLibrary containing "
            "a compatible carrier."
        )

    def with_carrier(self, carrier: object) -> "CarrierLibrary":
        return type(self)((carrier, *self.carriers))

    def without_carrier_type(self, carrier_type: type) -> "CarrierLibrary":
        return type(self)(
            tuple(
                carrier
                for carrier in self.carriers
                if not isinstance(carrier, carrier_type)
            )
        )