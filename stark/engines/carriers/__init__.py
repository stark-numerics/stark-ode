"""Carrier policies for STARK interface values."""

__all__: list[str] = []

# Core carrier composition model

from stark.core.contracts.carrier import (
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

from stark.engines.carriers.native import (
    CarrierNative,
    CarrierBasisNative,
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
    "CarrierBasisNative",
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

from stark.engines.carriers.numpy import (
    CarrierNumpy,
    CarrierBasisNumpy,
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
    "CarrierBasisNumpy",
    "CarrierArithmeticNumpy",
    "CarrierNormNumpyMax",
    "CarrierNormNumpyRMS",
    "CarrierNumpy",
    "CarrierNumpyValue",
    "CarrierStorageNumpy",
    "CarrierValidationNumpy",
]

try:
    from stark.engines.carriers.cupy import (
        CarrierAllocationCupy,
        CarrierArithmeticCupy,
        CarrierCupy,
        CarrierBasisCupy,
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
        "CarrierBasisCupy",
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
    from stark.engines.carriers.jax import (
        CarrierAllocationJax,
        CarrierArithmeticJax,
        CarrierJax,
        CarrierBasisJax,
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
        "CarrierBasisJax",
        "CarrierArithmeticJax",
        "CarrierJax",
        "CarrierJaxValue",
        "CarrierNormJaxMax",
        "CarrierNormJaxRMS",
        "CarrierStorageJax",
        "CarrierValidationJax",
    ]
