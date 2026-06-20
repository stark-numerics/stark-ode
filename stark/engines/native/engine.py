from __future__ import annotations

from array import array
from dataclasses import dataclass, field

from stark.engines.shared.accelerators import AcceleratorNone, AcceleratorNumba
from stark.engines.shared.algebraist.generator import (
    AlgebraistGeneratorInnerProduct,
    AlgebraistGeneratorLinearCombine,
    AlgebraistGeneratorNorm,
    AlgebraistGeneratorSpecialist,
)
from stark.engines.shared.algebraist.frame import AlgebraistFrame, AlgebraistFrameLooped
from stark.engines.carriers.native import CarrierNativeArray
from stark.core.contracts.accelerator import Accelerator
from stark.engines.native.allocator import EngineAllocatorNative
from stark.problem.frame.frame import Frame


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
    and translation objects, the derived algebraist frame, generated algebra
    kernels, and an accelerator. This engine stores fields in Python
    `array.array` values and uses Numba by default when it is installed.
    """

    frame: Frame
    typecode: str = "d"
    accelerator: Accelerator = field(default_factory=_default_accelerator)
    algebraist_frame: AlgebraistFrame = field(init=False)
    carriers: tuple[CarrierNativeArray, ...] = field(init=False, repr=False)
    allocator: EngineAllocatorNative = field(init=False, repr=False)
    algebraist_inner_product: AlgebraistGeneratorInnerProduct = field(init=False, repr=False)
    algebraist_linear_combine: AlgebraistGeneratorLinearCombine = field(init=False, repr=False)
    algebraist_norm: AlgebraistGeneratorNorm = field(init=False, repr=False)
    algebraist_specialist: AlgebraistGeneratorSpecialist = field(init=False, repr=False)

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

    def __post_init__(self) -> None:
        algebraist_frame = self.frame.to_algebraist_frame()
        carriers: list[CarrierNativeArray] = []

        for field in algebraist_frame.fields:
            policy = field.policy
            if not isinstance(policy, AlgebraistFrameLooped) or policy.shape is None:
                raise ValueError(
                    "EngineNative requires every frame field to declare shape."
                )
            if len(policy.shape) != 1:
                raise ValueError(
                    "EngineNative currently supports one-dimensional field shapes only."
                )
            carriers.append(CarrierNativeArray(array(self.typecode, (0.0 for _ in range(policy.shape[0])))))

        carrier_tuple = tuple(carriers)
        allocator = EngineAllocatorNative(
            algebraist_frame=algebraist_frame,
            carriers=carrier_tuple,
        )

        object.__setattr__(self, "algebraist_frame", algebraist_frame)
        object.__setattr__(self, "carriers", carrier_tuple)
        object.__setattr__(self, "allocator", allocator)
        object.__setattr__(
            self,
            "algebraist_linear_combine",
            AlgebraistGeneratorLinearCombine(
                translation=allocator.allocate_translation(),
                allocator=allocator,
                frame=algebraist_frame,
                accelerator=self.accelerator,
            ),
        )
        algebraist_specialist = AlgebraistGeneratorSpecialist(
            translation=allocator.allocate_translation(),
            allocator=allocator,
            frame=algebraist_frame,
            accelerator=self.accelerator,
        )
        object.__setattr__(
            allocator,
            "apply_translation",
            algebraist_specialist.provide_unit_apply(),
        )
        algebraist_norm = AlgebraistGeneratorNorm(
            translation=allocator.allocate_translation(),
            frame=algebraist_frame,
            accelerator=self.accelerator,
        )
        object.__setattr__(allocator, "norm", algebraist_norm.provide())
        algebraist_inner_product = AlgebraistGeneratorInnerProduct(
            translation=allocator.allocate_translation(),
            frame=algebraist_frame,
            accelerator=self.accelerator,
        )
        object.__setattr__(allocator, "inner_product", algebraist_inner_product.provide())

        object.__setattr__(self, "algebraist_inner_product", algebraist_inner_product)
        object.__setattr__(self, "algebraist_norm", algebraist_norm)
        object.__setattr__(self, "algebraist_specialist", algebraist_specialist)


__all__ = ["EngineNative"]
