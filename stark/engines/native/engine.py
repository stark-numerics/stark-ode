from __future__ import annotations

from array import array
from dataclasses import dataclass, field

from stark.accelerators import AcceleratorNone, AcceleratorNumba
from stark.algebraist.generator import (
    AlgebraistGeneratorInnerProduct,
    AlgebraistGeneratorLinearCombine,
    AlgebraistGeneratorNorm,
    AlgebraistGeneratorSpecialist,
)
from stark.algebraist.layout import AlgebraistLayout, AlgebraistLayoutLooped
from stark.carriers.native import CarrierNativeArray
from stark.contracts.accelerator import Accelerator
from stark.engines.native.allocator import EngineAllocatorNative
from stark.interface.layout import Layout


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
    IVP: carrier templates for each layout field, an allocator for owned state
    and translation objects, the derived algebraist layout, generated algebra
    kernels, and an accelerator. This engine stores fields in Python
    `array.array` values and uses Numba by default when it is installed.
    """

    layout: Layout
    typecode: str = "d"
    accelerator: Accelerator = field(default_factory=_default_accelerator)
    algebraist_layout: AlgebraistLayout = field(init=False)
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
            f"{type(self).__name__}(layout={self.layout!r}, "
            f"typecode={self.typecode!r}, {acceleration})"
        )

    def __post_init__(self) -> None:
        algebraist_layout = self.layout.to_algebraist_layout()
        carriers: list[CarrierNativeArray] = []

        for field in algebraist_layout.fields:
            policy = field.policy
            if not isinstance(policy, AlgebraistLayoutLooped) or policy.shape is None:
                raise ValueError(
                    "EngineNative requires every layout field to declare shape."
                )
            if len(policy.shape) != 1:
                raise ValueError(
                    "EngineNative currently supports one-dimensional field shapes only."
                )
            carriers.append(CarrierNativeArray(array(self.typecode, (0.0 for _ in range(policy.shape[0])))))

        carrier_tuple = tuple(carriers)
        allocator = EngineAllocatorNative(
            algebraist_layout=algebraist_layout,
            carriers=carrier_tuple,
        )

        object.__setattr__(self, "algebraist_layout", algebraist_layout)
        object.__setattr__(self, "carriers", carrier_tuple)
        object.__setattr__(self, "allocator", allocator)
        object.__setattr__(
            self,
            "algebraist_linear_combine",
            AlgebraistGeneratorLinearCombine(
                translation=allocator.allocate_translation(),
                allocator=allocator,
                layout=algebraist_layout,
                accelerator=self.accelerator,
            ),
        )
        algebraist_specialist = AlgebraistGeneratorSpecialist(
            translation=allocator.allocate_translation(),
            allocator=allocator,
            layout=algebraist_layout,
            accelerator=self.accelerator,
        )
        object.__setattr__(
            allocator,
            "apply_translation",
            algebraist_specialist.provide_unit_apply(),
        )
        algebraist_norm = AlgebraistGeneratorNorm(
            translation=allocator.allocate_translation(),
            layout=algebraist_layout,
            accelerator=self.accelerator,
        )
        object.__setattr__(allocator, "norm", algebraist_norm.provide())
        algebraist_inner_product = AlgebraistGeneratorInnerProduct(
            translation=allocator.allocate_translation(),
            layout=algebraist_layout,
            accelerator=self.accelerator,
        )
        object.__setattr__(allocator, "inner_product", algebraist_inner_product.provide())

        object.__setattr__(self, "algebraist_inner_product", algebraist_inner_product)
        object.__setattr__(self, "algebraist_norm", algebraist_norm)
        object.__setattr__(self, "algebraist_specialist", algebraist_specialist)


__all__ = ["EngineNative"]
