from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from typing import Any

from stark.engines.accelerators import AcceleratorNone, AcceleratorNumba
from stark.engines.algebraist import Algebraist
from stark.engines.carrier_native import CarrierNativeArray
from stark.engines.translation_factory_native import TranslationFactoryNative
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
class EngineNative:
    """
    Native Python backend bundle for one-dimensional shaped layouts.

    An engine supplies the backend objects used when a `System` prepares an
    IVP: carrier templates for each frame field, an allocator for owned state
    and translation objects, generated algebra kernels, and an accelerator.
    This engine stores fields in Python `array.array` values and uses Numba by
    default when it is installed.
    """

    frame: FrameLike
    typecode: str = "d"
    accelerator: Accelerator = field(default_factory=_default_accelerator)
    carriers: tuple[CarrierNativeArray, ...] = field(init=False, repr=False)
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
            f"typecode={self.typecode!r}, {acceleration})"
        )

    def translation_basis(self) -> TranslationBasis:
        """Return the coordinate basis for engine-owned translations.

        This is an inspection and dense-materialisation helper. Ordinary user
        solves should not need it, but dense inverters and diagnostic examples
        can use it instead of hand-writing translation-basis classes.
        """

        return TranslationBasis(self.frame, self.carriers)

    def __post_init__(self) -> None:
        carriers: list[CarrierNativeArray] = []

        for field in self.frame.fields:
            policy = field.policy
            shape = getattr(policy, "shape", None)
            if shape is None:
                shape = getattr(field, "shape", None)
            if getattr(policy, "kind", None) != "looped" or shape is None:
                raise ValueError(
                    "EngineNative requires every frame field to declare shape."
                )
            if len(shape) != 1:
                raise ValueError(
                    "EngineNative currently supports one-dimensional field shapes only."
                )
            carriers.append(CarrierNativeArray(array(self.typecode, (0.0 for _ in range(shape[0])))))

        carrier_tuple = tuple(carriers)
        allocator = Allocator(
            frame=self.frame,
            carriers=carrier_tuple,
            translation_type=TranslationFactoryNative,
        )

        object.__setattr__(self, "carriers", carrier_tuple)
        object.__setattr__(self, "allocator", allocator)
        algebraist = Algebraist.generator(
            frame=self.frame,
            allocator=allocator,
            accelerator=self.accelerator,
        )
        object.__setattr__(self, "algebraist", algebraist)


__all__ = ["EngineNative"]
