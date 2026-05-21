from dataclasses import dataclass
from collections.abc import Sequence
from numbers import Number
from typing import TypeAlias

CarrierNativeValue: TypeAlias = Number | Sequence[Number]

@dataclass(frozen=True)
class CarrierStorageNative:
    template: CarrierNativeValue

    def is_state(self, value: CarrierNativeValue) -> bool:
        return self.matches_template(value)

    def is_translation(self, value: CarrierNativeValue) -> bool:
        return self.matches_template(value)
    
    def matches_template(self, value: CarrierNativeValue) -> bool:
        if isinstance(self.template, Number):
            return isinstance(value, Number)

        if isinstance(self.template, list):
            return isinstance(value, list) and len(value) == len(self.template)

        if isinstance(self.template, tuple):
            return isinstance(value, tuple) and len(value) == len(self.template)

        return False