from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from stark.engines.native.carriers.hints import HintNativeNumber

CarrierNativeListValue: TypeAlias = list[HintNativeNumber]


@dataclass(frozen=True)
class CarrierStorageNativeList:
    length: int

    @classmethod
    def from_template(cls, template: CarrierNativeListValue) -> "CarrierStorageNativeList":
        if not isinstance(template, list):
            raise TypeError("Native list carrier template must be a list.")
        return cls(length=len(template))

    def is_state(self, value: CarrierNativeListValue) -> bool:
        return self.matches_template(value)

    def is_translation(self, value: CarrierNativeListValue) -> bool:
        return self.matches_template(value)

    def matches_template(self, value: object) -> bool:
        return isinstance(value, list) and len(value) == self.length
