"""Storage policies for prepared STARK carriers."""

from stark.carriers.native.storage import CarrierNativeValue, CarrierStorageNative
from stark.carriers.numpy.storage import CarrierNumpyValue, CarrierStorageNumpy

__all__ = [
    "CarrierNativeValue",
    "CarrierStorageNative",
    "CarrierNumpyValue",
    "CarrierStorageNumpy",
]

try:
    from stark.carriers.cupy.storage import CarrierStorageCupy
except ImportError:
    pass
else:
    __all__ += ["CarrierStorageCupy"]

try:
    from stark.carriers.jax.storage import CarrierJaxValue, CarrierStorageJax
except ImportError:
    pass
else:
    __all__ += [
        "CarrierJaxValue",
        "CarrierStorageJax",
    ]