from dataclasses import dataclass
from numbers import Number

from stark.carriers.native.storage import CarrierNativeValue, CarrierStorageNative


@dataclass(frozen=True)
class CarrierAllocationNative:
    storage: CarrierStorageNative

    def zero_state(self) -> CarrierNativeValue:
        return self.zero_like_template()

    def zero_translation(self) -> CarrierNativeValue:
        return self.zero_like_template()

    def allocate_translation(self) -> CarrierNativeValue:
        return self.zero_translation()

    def copy_state(self, value: CarrierNativeValue) -> CarrierNativeValue:
        return self.copy_value(value)

    def copy_translation(self, value: CarrierNativeValue) -> CarrierNativeValue:
        return self.copy_value(value)

    def zero_like_template(self) -> CarrierNativeValue:
        template = self.storage.template

        if isinstance(template, Number):
            return 0

        if isinstance(template, list):
            return [0 for _ in template]

        if isinstance(template, tuple):
            return tuple(0 for _ in template)

        raise TypeError("Native carrier template must be numeric, list, or tuple.")

    def copy_value(self, value: CarrierNativeValue) -> CarrierNativeValue:
        if isinstance(value, Number):
            return value

        if isinstance(value, list):
            return list(value)

        if isinstance(value, tuple):
            return tuple(value)

        raise TypeError("Native carrier value must be numeric, list, or tuple.")