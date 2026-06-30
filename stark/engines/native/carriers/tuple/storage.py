from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from stark.engines.native.carriers.hints import HintNativeNumber

CarrierNativeTupleValue: TypeAlias = tuple[HintNativeNumber, ...]


@dataclass(frozen=True)
class CarrierStorageNativeTuple:
    length: int

    @classmethod
    def from_template(cls, template: CarrierNativeTupleValue) -> "CarrierStorageNativeTuple":
        if not isinstance(template, tuple):
            raise TypeError("Native tuple carrier template must be a tuple.")
        return cls(length=len(template))

    def is_state(self, value: CarrierNativeTupleValue) -> bool:
        return self.matches_template(value)

    def is_translation(self, value: CarrierNativeTupleValue) -> bool:
        return self.matches_template(value)

    def matches_template(self, value: object) -> bool:
        return isinstance(value, tuple) and len(value) == self.length
