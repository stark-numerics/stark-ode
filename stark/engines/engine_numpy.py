from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from stark.engines.accelerators import AcceleratorNone, AcceleratorNumba
from stark.engines.algebraist import Algebraist
from stark.engines.carrier_numpy import CarrierNumpy
from stark.engines.translation_factory_numpy import TranslationFactoryNumpy
from stark.core.contracts.accelerator import Accelerator
from stark.core.contracts.frame import FrameLike
from stark.engines.allocator import Allocator
from stark.engines.translation_basis import TranslationBasis


def _default_accelerator() -> Accelerator:
    try:
        return AcceleratorNumba()
    except ModuleNotFoundError:
        return AcceleratorNone()


@dataclass(frozen=True, slots=True)
class EngineNumpy:
    """
    NumPy backend bundle for a shaped `Frame`.

    An engine supplies the backend objects used when a `System` prepares an
    IVP: carrier templates for each frame field, an allocator for owned state
    and translation objects, generated algebra kernels, and an accelerator. By
    default this engine uses Numba when it is installed and otherwise falls
    back to unaccelerated NumPy-compatible callables.
    """

    frame: FrameLike
    dtype: Any = np.float64
    accelerator: Accelerator = field(default_factory=_default_accelerator)
    carriers: tuple[CarrierNumpy, ...] = field(init=False, repr=False)
    allocator: Allocator = field(init=False, repr=False)
    algebraist: Algebraist[Any, Any] = field(init=False, repr=False)

    def __repr__(self) -> str:
        accelerator_name = getattr(self.accelerator, "name", str(self.accelerator))
        acceleration = f"accelerator={accelerator_name!r}"
        if accelerator_name == "none":
            acceleration += (
                ", WARNING='unaccelerated CPU engine; install numba or pass an "
                "accelerator to compile generated kernels'"
            )
        return (
            f"{type(self).__name__}(frame={self.frame!r}, "
            f"dtype={np.dtype(self.dtype)!r}, {acceleration})"
        )

    def translation_basis(self) -> TranslationBasis:
        """Return the coordinate basis for engine-owned translations.

        This is an inspection and dense-materialisation helper. Ordinary user
        solves should not need it, but dense inverters and diagnostic examples
        can use it instead of hand-writing translation-basis classes.
        """

        return TranslationBasis(self.frame, self.carriers)

    def __post_init__(self) -> None:
        carriers: list[CarrierNumpy] = []
        dtype = np.dtype(self.dtype)

        for field in self.frame.fields:
            policy = field.policy
            shape = getattr(policy, "shape", None)
            if shape is None:
                shape = getattr(field, "shape", None)
            if getattr(policy, "kind", None) != "looped" or shape is None:
                raise ValueError(
                    "EngineNumpy requires every frame field to declare shape."
                )
            carriers.append(CarrierNumpy(np.zeros(shape, dtype=dtype)))

        carrier_tuple = tuple(carriers)
        allocator = Allocator(
            frame=self.frame,
            carriers=carrier_tuple,
            translation_type=TranslationFactoryNumpy,
        )

        object.__setattr__(self, "carriers", carrier_tuple)
        object.__setattr__(self, "allocator", allocator)
        algebraist = Algebraist.generator(
            frame=self.frame,
            allocator=allocator,
            accelerator=self.accelerator,
        )
        object.__setattr__(self, "algebraist", algebraist)


__all__ = ["EngineNumpy"]
