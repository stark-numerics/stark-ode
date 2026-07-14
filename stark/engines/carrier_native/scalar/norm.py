from __future__ import annotations

from dataclasses import dataclass

from stark.engines.carrier_native.scalar.storage import CarrierNativeScalarValue


@dataclass(frozen=True)
class CarrierNormNativeScalarAbs:
    def __call__(self, value: CarrierNativeScalarValue) -> float:
        return float(abs(value))


@dataclass(frozen=True)
class CarrierNormNativeScalarRMS:
    def __call__(self, value: CarrierNativeScalarValue) -> float:
        return float(abs(value))


@dataclass(frozen=True)
class CarrierNormNativeScalarMax:
    def __call__(self, value: CarrierNativeScalarValue) -> float:
        return float(abs(value))
