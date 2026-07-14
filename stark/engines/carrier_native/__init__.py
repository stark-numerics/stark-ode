"""Native Python carrier parts.

Native carriers are split by storage shape: scalar, list, tuple, and
`array.array`. The split classes are the current implementation surface.
`CarrierNative` is a small selector facade used by generic native paths; old
allocation/arithmetic/validation compatibility facades are intentionally not
exported before release.
"""

from stark.engines.carrier_native.basis import CarrierBasisNative
from stark.engines.carrier_native.array import (
    CarrierAllocationNativeArray,
    CarrierBasisNativeArray,
    CarrierArithmeticNativeArray,
    CarrierNativeArray,
    CarrierNativeArrayValue,
    CarrierNormNativeArrayMax,
    CarrierNormNativeArrayRMS,
    CarrierStorageNativeArray,
    CarrierValidationNativeArray,
)
from stark.engines.carrier_native.carrier import CarrierNative
from stark.engines.carrier_native.list import (
    CarrierAllocationNativeList,
    CarrierBasisNativeList,
    CarrierArithmeticNativeList,
    CarrierNativeList,
    CarrierNativeListValue,
    CarrierNormNativeListMax,
    CarrierNormNativeListRMS,
    CarrierStorageNativeList,
    CarrierValidationNativeList,
)
from stark.engines.carrier_native.scalar import (
    CarrierAllocationNativeScalar,
    CarrierBasisNativeScalar,
    CarrierArithmeticNativeScalar,
    CarrierNativeScalar,
    CarrierNativeScalarValue,
    CarrierNormNativeScalarMax,
    CarrierNormNativeScalarRMS,
    CarrierStorageNativeScalar,
    CarrierValidationNativeScalar,
)
from stark.engines.carrier_native.storage import CarrierNativeValue
from stark.engines.carrier_native.tuple import (
    CarrierAllocationNativeTuple,
    CarrierBasisNativeTuple,
    CarrierArithmeticNativeTuple,
    CarrierNativeTuple,
    CarrierNativeTupleValue,
    CarrierNormNativeTupleMax,
    CarrierNormNativeTupleRMS,
    CarrierStorageNativeTuple,
    CarrierValidationNativeTuple,
)

__all__ = [
    "CarrierBasisNative",
    "CarrierNative",
    "CarrierNativeValue",
    "CarrierAllocationNativeArray",
    "CarrierBasisNativeArray",
    "CarrierArithmeticNativeArray",
    "CarrierNativeArray",
    "CarrierNativeArrayValue",
    "CarrierNormNativeArrayMax",
    "CarrierNormNativeArrayRMS",
    "CarrierStorageNativeArray",
    "CarrierValidationNativeArray",
    "CarrierAllocationNativeList",
    "CarrierBasisNativeList",
    "CarrierArithmeticNativeList",
    "CarrierNativeList",
    "CarrierNativeListValue",
    "CarrierNormNativeListMax",
    "CarrierNormNativeListRMS",
    "CarrierStorageNativeList",
    "CarrierValidationNativeList",
    "CarrierAllocationNativeScalar",
    "CarrierBasisNativeScalar",
    "CarrierArithmeticNativeScalar",
    "CarrierNativeScalar",
    "CarrierNativeScalarValue",
    "CarrierNormNativeScalarMax",
    "CarrierNormNativeScalarRMS",
    "CarrierStorageNativeScalar",
    "CarrierValidationNativeScalar",
    "CarrierAllocationNativeTuple",
    "CarrierBasisNativeTuple",
    "CarrierArithmeticNativeTuple",
    "CarrierNativeTuple",
    "CarrierNativeTupleValue",
    "CarrierNormNativeTupleMax",
    "CarrierNormNativeTupleRMS",
    "CarrierStorageNativeTuple",
    "CarrierValidationNativeTuple",
]
