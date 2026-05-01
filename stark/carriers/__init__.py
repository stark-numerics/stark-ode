"""Carrier policies for STARK interface values."""

from .core import (
    CarrierError,
    Carrier,
    CarrierBound,
    CarrierNative,
    CarrierNativeBound,
    CarrierNumpy,
    CarrierNumpyBound,
)
from .kernels import (
    CarrierKernel,
    CarrierKernelBound,
    CarrierKernelNative,
    CarrierKernelNativeBound,
    CarrierKernelNumpy,
    CarrierKernelNumpyBound,
)
from .library import CarrierLibrary
from .norms import (
    CarrierNorm,
    CarrierNormNativeRMS,
    CarrierNormNumpyRMS,
    CarrierNormNumpyMax,
)
from .algebraist import CarrierKernelAlgebraist, CarrierKernelAlgebraistBound

__all__ = [
    "CarrierError",
    "Carrier",
    "CarrierBound",
    "CarrierLibrary",
    "CarrierNative",
    "CarrierNativeBound",
    "CarrierNumpy",
    "CarrierNumpyBound",
    "CarrierKernel",
    "CarrierKernelBound",
    "CarrierKernelNative",
    "CarrierKernelNativeBound",
    "CarrierKernelNumpy",
    "CarrierKernelNumpyBound",
    "CarrierNorm",
    "CarrierNormNativeRMS",
    "CarrierNormNumpyRMS",
    "CarrierNormNumpyMax",
    "CarrierKernelAlgebraist",
    "CarrierKernelAlgebraistBound",
]

try:
    from .cupy import (
        CarrierCuPy,
        CarrierCuPyBound,
        CarrierKernelCuPy,
        CarrierKernelCuPyBound,
        CarrierNormCuPyMax,
        CarrierNormCuPyRMS,
    )
except ImportError:
    pass
else:
    __all__ += [
        "CarrierCuPy",
        "CarrierCuPyBound",
        "CarrierKernelCuPy",
        "CarrierKernelCuPyBound",
        "CarrierNormCuPyRMS",
        "CarrierNormCuPyMax",
    ]

try:
    from .jax import (
        CarrierJax,
        CarrierJaxBound,
        CarrierKernelJax,
        CarrierKernelJaxBound,
        CarrierNormJaxMax,
        CarrierNormJaxRMS,
    )
except ImportError:
    pass
else:
    __all__ += [
        "CarrierJax",
        "CarrierJaxBound",
        "CarrierKernelJax",
        "CarrierKernelJaxBound",
        "CarrierNormJaxRMS",
        "CarrierNormJaxMax",
    ]