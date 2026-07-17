from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from stark.engines.accelerators import AcceleratorNone, AcceleratorNumba
from stark.engines.carrier_numpy import CarrierNumpy
from stark.engines.generator import (
    Generator,
    GeneratorPolicy,
    GeneratorRequestApplyTranslation,
    GeneratorRequestInnerProduct,
    GeneratorRequestLinearCombineTable,
    GeneratorRequestNorm,
)
from stark.engines.translation_factory_numpy import TranslationFactoryNumpy
from stark.core.contracts.accelerator import Accelerator
from stark.core.contracts.frame import FrameLike
from stark.engines.allocator import AllocatorCarried
from stark.engines.translation_basis import TranslationBasis


def _default_accelerator() -> Accelerator:
    try:
        return AcceleratorNumba()
    except ModuleNotFoundError:
        return AcceleratorNone()


@dataclass(slots=True)
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
    allocator: AllocatorCarried[Any] = field(init=False, repr=False)
    generator: Generator[Any, Any] = field(init=False, repr=False)

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
        allocator: AllocatorCarried[Any] = AllocatorCarried(
            frame=self.frame,
            carriers=carrier_tuple,
            translation_type=TranslationFactoryNumpy,
        )

        self.carriers = carrier_tuple
        self.allocator = allocator
        generator = Generator(
            frame=self.frame,
            accelerator=self.accelerator,
            policy=GeneratorPolicy(),
            allocator=allocator,
        )
        allocator.apply_translation = cast(
            Callable[[Any, Any, Any], Any],
            generator(GeneratorRequestApplyTranslation()),
        )
        allocator.linear_combine = cast(
            tuple[Callable[..., Any], ...],
            generator(GeneratorRequestLinearCombineTable(max_arity=12)),
        )
        allocator.norm = cast(
            Callable[[Any], float],
            generator(GeneratorRequestNorm()),
        )
        allocator.inner_product = cast(
            Callable[[Any, Any], float],
            generator(GeneratorRequestInnerProduct()),
        )
        self.generator = generator


__all__ = ["EngineNumpy"]
