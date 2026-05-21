"""Carrier policies for STARK interface values."""

__all__: list[str] = []

# Core carrier composition model

from stark.carriers.carrier import (
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

# New native carrier parts

from stark.carriers.native import (
    CarrierAllocationNative,
    CarrierArithmeticNative,
    CarrierStorageNative,
    CarrierValidationNative,
)

__all__ += [
    "CarrierAllocationNative",
    "CarrierArithmeticNative",
    "CarrierStorageNative",
    "CarrierValidationNative",
]

# New NumPy carrier parts

from stark.carriers.numpy import (
    CarrierAllocationNumpy,
    CarrierArithmeticNumpy,
    CarrierStorageNumpy,
    CarrierValidationNumpy,
)

__all__ += [
    "CarrierAllocationNumpy",
    "CarrierArithmeticNumpy",
    "CarrierStorageNumpy",
    "CarrierValidationNumpy",
]

# New optional Cupy carrier parts

try:
    from stark.carriers.cupy.allocation import CarrierAllocationCupy
    from stark.carriers.cupy.arithmetic import CarrierArithmeticCupy
    from stark.carriers.cupy.storage import CarrierStorageCupy
    from stark.carriers.cupy.validation import CarrierValidationCupy
except ImportError:
    pass
else:
    __all__ += [
        "CarrierAllocationCupy",
        "CarrierArithmeticCupy",
        "CarrierStorageCupy",
        "CarrierValidationCupy",
    ]

# New optional JAX carrier parts

try:
    from stark.carriers.jax.allocation import CarrierAllocationJax
    from stark.carriers.jax.arithmetic import CarrierArithmeticJax
    from stark.carriers.jax.storage import CarrierStorageJax
    from stark.carriers.jax.validation import CarrierValidationJax
except ImportError:
    pass
else:
    __all__ += [
        "CarrierAllocationJax",
        "CarrierArithmeticJax",
        "CarrierStorageJax",
        "CarrierValidationJax",
    ]

# Deprecated bind-based carrier model

from stark.carriers.deprecated.core import (
    DeprecatedCarrier,
    DeprecatedCarrierBound,
    DeprecatedCarrierError,
    DeprecatedCarrierNative,
    DeprecatedCarrierNativeBound,
    DeprecatedCarrierNumpy,
    DeprecatedCarrierNumpyBound,
)
from stark.carriers.deprecated.kernels import (
    DeprecatedCarrierKernel,
    DeprecatedCarrierKernelBound,
    DeprecatedCarrierKernelNative,
    DeprecatedCarrierKernelNativeBound,
    DeprecatedCarrierKernelNumpy,
    DeprecatedCarrierKernelNumpyBound,
)
from stark.carriers.deprecated.library import DeprecatedCarrierLibrary
from stark.carriers.deprecated.norms import (
    DeprecatedCarrierNorm,
    DeprecatedCarrierNormNativeRMS,
    DeprecatedCarrierNormNumpyMax,
    DeprecatedCarrierNormNumpyRMS,
)
from stark.carriers.deprecated.algebraist import (
    DeprecatedCarrierKernelAlgebraist,
    DeprecatedCarrierKernelAlgebraistBound,
)

__all__ += [
    "DeprecatedCarrier",
    "DeprecatedCarrierBound",
    "DeprecatedCarrierError",
    "DeprecatedCarrierLibrary",
    "DeprecatedCarrierNative",
    "DeprecatedCarrierNativeBound",
    "DeprecatedCarrierNumpy",
    "DeprecatedCarrierNumpyBound",
    "DeprecatedCarrierKernel",
    "DeprecatedCarrierKernelBound",
    "DeprecatedCarrierKernelNative",
    "DeprecatedCarrierKernelNativeBound",
    "DeprecatedCarrierKernelNumpy",
    "DeprecatedCarrierKernelNumpyBound",
    "DeprecatedCarrierNorm",
    "DeprecatedCarrierNormNativeRMS",
    "DeprecatedCarrierNormNumpyMax",
    "DeprecatedCarrierNormNumpyRMS",
    "DeprecatedCarrierKernelAlgebraist",
    "DeprecatedCarrierKernelAlgebraistBound",
]

# Deprecated optional Cupy model

try:
    from stark.carriers.deprecated.cupy import (
        DeprecatedCarrierCuPy,
        DeprecatedCarrierCuPyBound,
        DeprecatedCarrierKernelCuPy,
        DeprecatedCarrierKernelCuPyBound,
        DeprecatedCarrierNormCuPyMax,
        DeprecatedCarrierNormCuPyRMS,
    )
except ImportError:
    pass
else:
    __all__ += [
        "DeprecatedCarrierCuPy",
        "DeprecatedCarrierCuPyBound",
        "DeprecatedCarrierKernelCuPy",
        "DeprecatedCarrierKernelCuPyBound",
        "DeprecatedCarrierNormCuPyMax",
        "DeprecatedCarrierNormCuPyRMS",
    ]

# Deprecated optional JAX model

try:
    from stark.carriers.deprecated.jax import (
        DeprecatedCarrierJax,
        DeprecatedCarrierJaxBound,
        DeprecatedCarrierKernelJax,
        DeprecatedCarrierKernelJaxBound,
        DeprecatedCarrierNormJaxMax,
        DeprecatedCarrierNormJaxRMS,
    )
except ImportError:
    pass
else:
    __all__ += [
        "DeprecatedCarrierJax",
        "DeprecatedCarrierJaxBound",
        "DeprecatedCarrierKernelJax",
        "DeprecatedCarrierKernelJaxBound",
        "DeprecatedCarrierNormJaxMax",
        "DeprecatedCarrierNormJaxRMS",
    ]