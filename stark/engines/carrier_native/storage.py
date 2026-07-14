from __future__ import annotations

from typing import TypeAlias

from stark.engines.carrier_native.array import CarrierNativeArrayValue
from stark.engines.carrier_native.list import CarrierNativeListValue
from stark.engines.carrier_native.scalar import CarrierNativeScalarValue
from stark.engines.carrier_native.tuple import CarrierNativeTupleValue

CarrierNativeValue: TypeAlias = (
    CarrierNativeScalarValue
    | CarrierNativeListValue
    | CarrierNativeTupleValue
    | CarrierNativeArrayValue
)
