"""Norm policies for prepared STARK carriers."""

from stark.carriers.native.norm import (
    CarrierNormNativeMax,
    CarrierNormNativeRMS,
    CarrierNormNativeScalarAbs,
)
from stark.carriers.numpy.norm import (
    CarrierNormNumpyMax,
    CarrierNormNumpyRMS,
)

__all__ = [
    "CarrierNormNativeMax",
    "CarrierNormNativeRMS",
    "CarrierNormNativeScalarAbs",
    "CarrierNormNumpyMax",
    "CarrierNormNumpyRMS",
]

try:
    from stark.carriers.cupy.norm import (
        CarrierNormCupyMax,
        CarrierNormCupyRMS,
    )
except ImportError:
    pass
else:
    __all__ += [
        "CarrierNormCupyMax",
        "CarrierNormCupyRMS",
    ]

try:
    from stark.carriers.jax.norm import (
        CarrierNormJaxMax,
        CarrierNormJaxRMS,
    )
except ImportError:
    pass
else:
    __all__ += [
        "CarrierNormJaxMax",
        "CarrierNormJaxRMS",
    ]