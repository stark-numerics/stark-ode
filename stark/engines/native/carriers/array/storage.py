from __future__ import annotations

from array import array
from dataclasses import dataclass
from typing import TypeAlias

CarrierNativeArrayValue: TypeAlias = array
_FLOAT_TYPECODES = frozenset({"f", "d"})


@dataclass(frozen=True)
class CarrierStorageNativeArray:
    length: int
    typecode: str

    @classmethod
    def from_template(cls, template: CarrierNativeArrayValue) -> "CarrierStorageNativeArray":
        if not isinstance(template, array):
            raise TypeError("Native array carrier template must be an array.array.")
        if template.typecode not in _FLOAT_TYPECODES:
            raise TypeError(
                "Native array carrier only supports floating array.array typecodes "
                f"{sorted(_FLOAT_TYPECODES)!r}; got {template.typecode!r}."
            )
        return cls(length=len(template), typecode=template.typecode)

    def is_state(self, value: CarrierNativeArrayValue) -> bool:
        return self.matches_template(value)

    def is_translation(self, value: CarrierNativeArrayValue) -> bool:
        return self.matches_template(value)

    def matches_template(self, value: object) -> bool:
        return (
            isinstance(value, array)
            and value.typecode == self.typecode
            and len(value) == self.length
        )
