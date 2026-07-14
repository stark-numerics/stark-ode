from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import TypeAlias

from stark.engines.carrier_native.hints import HintNativeNumber

CarrierNativeScalarValue: TypeAlias = HintNativeNumber


@dataclass(frozen=True)
class CarrierStorageNativeScalar:
    template: CarrierNativeScalarValue

    def __post_init__(self) -> None:
        if not isinstance(self.template, Real):
            raise TypeError("Native scalar carrier template must be numeric.")

    def is_state(self, value: CarrierNativeScalarValue) -> bool:
        return self.matches_template(value)

    def is_translation(self, value: CarrierNativeScalarValue) -> bool:
        return self.matches_template(value)

    def matches_template(self, value: object) -> bool:
        return isinstance(value, Real)
