from __future__ import annotations

from typing import TypeAlias

from stark.engines.native.carriers.array import CarrierNativeArrayValue
from stark.engines.native.carriers.list import CarrierNativeListValue
from stark.engines.native.carriers.scalar import CarrierNativeScalarValue
from stark.engines.native.carriers.tuple import CarrierNativeTupleValue

CarrierNativeValue: TypeAlias = (
    CarrierNativeScalarValue
    | CarrierNativeListValue
    | CarrierNativeTupleValue
    | CarrierNativeArrayValue
)
