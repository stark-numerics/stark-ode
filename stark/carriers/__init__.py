"""Carrier policies for STARK interface values."""

__all__: list[str] = []

# Core carrier composition model

from stark.contracts.carriers import (
    Carrier,
    CarrierAllocation,
    CarrierArithmetic,
    CarrierNorm,
    CarrierStorage,
    CarrierValidation,
)

__all__ += [
    "Carrier",
    "CarrierAllocation",
    "CarrierArithmetic",
    "CarrierNorm",
    "CarrierStorage",
    "CarrierValidation",
]

from stark.carriers.native import (
    CarrierNative,
    CarrierAllocationNative,
    CarrierArithmeticNative,
    CarrierNativeValue,
    CarrierNormNativeMax,
    CarrierNormNativeRMS,
    CarrierNormNativeScalarAbs,
    CarrierStorageNative,
    CarrierValidationNative,
)

__all__ += [
    "CarrierAllocationNative",
    "CarrierArithmeticNative",
    "CarrierNative",
    "CarrierNativeValue",
    "CarrierNormNativeMax",
    "CarrierNormNativeRMS",
    "CarrierNormNativeScalarAbs",
    "CarrierStorageNative",
    "CarrierValidationNative",
]

# NumPy carrier parts

from stark.carriers.numpy import (
    CarrierNumpy,
    CarrierAllocationNumpy,
    CarrierArithmeticNumpy,
    CarrierNormNumpyMax,
    CarrierNormNumpyRMS,
    CarrierNumpyValue,
    CarrierStorageNumpy,
    CarrierValidationNumpy,
)

__all__ += [
    "CarrierAllocationNumpy",
    "CarrierArithmeticNumpy",
    "CarrierNormNumpyMax",
    "CarrierNormNumpyRMS",
    "CarrierNumpy",
    "CarrierNumpyValue",
    "CarrierStorageNumpy",
    "CarrierValidationNumpy",
]

try:
    from stark.carriers.cupy import (
        CarrierAllocationCupy,
        CarrierArithmeticCupy,
        CarrierCupy,
        CarrierCupyValue,
        CarrierNormCupyMax,
        CarrierNormCupyRMS,
        CarrierStorageCupy,
        CarrierValidationCupy,
    )
except ImportError:
    pass
else:
    __all__ += [
        "CarrierAllocationCupy",
        "CarrierArithmeticCupy",
        "CarrierCupy",
        "CarrierCupyValue",
        "CarrierNormCupyMax",
        "CarrierNormCupyRMS",
        "CarrierStorageCupy",
        "CarrierValidationCupy",
    ]

# Optional JAX carrier parts

try:
    from stark.carriers.jax import (
        CarrierAllocationJax,
        CarrierArithmeticJax,
        CarrierJax,
        CarrierJaxValue,
        CarrierNormJaxMax,
        CarrierNormJaxRMS,
        CarrierStorageJax,
        CarrierValidationJax,
    )
except ImportError:
    pass
else:
    __all__ += [
        "CarrierAllocationJax",
        "CarrierArithmeticJax",
        "CarrierJax",
        "CarrierJaxValue",
        "CarrierNormJaxMax",
        "CarrierNormJaxRMS",
        "CarrierStorageJax",
        "CarrierValidationJax",
    ]
