from __future__ import annotations

from collections.abc import Callable
from dataclasses import InitVar, dataclass, field as dataclass_field
from typing import Any, Protocol, cast

import numpy as np

from stark.core.contracts.engines.accelerator import Accelerator
from stark.core.contracts.engines.carrier import CarrierLike
from stark.core.contracts.problem.field import FieldPolicyLike
from stark.core.contracts.problem.frame import FrameLike
from stark.engines.accelerators.none import AcceleratorNone
from stark.engines.accelerators.numba import AcceleratorNumba
from stark.engines.carrier_native import CarrierNativeArray
from stark.engines.carrier_numpy import CarrierNumpy
from stark.engines.engine_allocator import EngineAllocator
from stark.engines.generator import (
    Generator,
    GeneratorPolicy,
    GeneratorPolicyLike,
    GeneratorRequestApplyTranslation,
    GeneratorRequestInnerProduct,
    GeneratorRequestLinearCombineTable,
    GeneratorRequestNorm,
)
from stark.engines.translation_basis import TranslationBasis


class EngineCarrierTypeLike(Protocol):
    """Carrier class surface required by the shared engine constructor."""

    @classmethod
    def from_shape(
        cls,
        shape: tuple[int, ...],
        dtype: object,
    ) -> CarrierLike[Any, Any]:
        ...

    @staticmethod
    def resolve_dtype(dtype: object) -> object:
        ...


@dataclass
class Engine:
    """Concrete backend bundle for one frame and one carrier family.

    Engines are the prepared runtime objects used by systems and methods. They
    bind a frame to backend storage carriers, an allocator, a generator, and an
    accelerator. The backend-specific names such as `EngineNumpy` are factories
    that fill in the carrier class, default dtype, default accelerator, and
    generator policy before returning this single engine object.
    """

    frame: FrameLike
    carrier_type: EngineCarrierTypeLike
    dtype: object
    accelerator: Accelerator
    policy: InitVar[GeneratorPolicyLike] = GeneratorPolicy()
    name: str = "Engine"
    carriers: tuple[CarrierLike[Any, Any], ...] = dataclass_field(
        init=False,
        repr=False,
    )
    allocator: EngineAllocator = dataclass_field(init=False, repr=False)
    generator: Generator[Any, Any] = dataclass_field(init=False, repr=False)

    def __repr__(self) -> str:
        acceleration = f"accelerator={self.accelerator.name!r}"
        if self.accelerator.name == "none":
            acceleration += (
                ", WARNING='unaccelerated engine; pass a compiling accelerator "
                "for generated kernels when this backend supports one'"
            )
        return (
            f"{self.name}(frame={self.frame!r}, "
            f"dtype={self.dtype!r}, {acceleration})"
        )

    def translation_basis(self) -> TranslationBasis:
        """Return the coordinate basis for engine-owned translations.

        This is an inspection and dense-materialisation helper. Ordinary user
        solves should not need it, but dense inverters and diagnostic examples
        can use it instead of hand-writing translation-basis classes.
        """

        return TranslationBasis(self.frame, self.carriers)

    def __post_init__(self, policy: GeneratorPolicyLike) -> None:
        dtype = self.carrier_type.resolve_dtype(self.dtype)
        self.dtype = dtype

        carriers: list[CarrierLike[Any, Any]] = []
        for frame_field in self.frame.fields:
            field_policy = cast(FieldPolicyLike, frame_field.policy)
            shape = frame_field.shape
            if field_policy.kind != "looped" or shape is None:
                raise ValueError(
                    f"{self.name} requires every frame field to declare shape."
                )
            carriers.append(self.carrier_type.from_shape(tuple(shape), dtype))

        carrier_tuple = tuple(carriers)
        allocator = EngineAllocator(
            frame=self.frame,
            carriers=carrier_tuple,
        )

        self.carriers = carrier_tuple
        self.allocator = allocator
        generator = Generator(
            frame=self.frame,
            accelerator=self.accelerator,
            policy=policy,
            allocator=allocator,
        )
        self.generator = generator

        if not generator.policy.active:
            return

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


@dataclass
class EngineFactory:
    """Callable backend preset that returns a prepared `Engine`."""

    name: str
    carrier_type: EngineCarrierTypeLike
    default_dtype: object
    accelerator_type: type[Accelerator]
    policy: GeneratorPolicyLike = dataclass_field(
        default_factory=GeneratorPolicy,
    )

    def __call__(
        self,
        frame: FrameLike,
        dtype: object | None = None,
        *,
        accelerator: Accelerator | None = None,
    ) -> Engine:
        return Engine(
            frame=frame,
            carrier_type=self.carrier_type,
            dtype=self.default_dtype if dtype is None else dtype,
            accelerator=self.accelerator_type() if accelerator is None else accelerator,
            policy=self.policy,
            name=self.name,
        )


EngineNumpy = EngineFactory(
    name="EngineNumpy",
    carrier_type=CarrierNumpy,
    default_dtype=np.float64,
    accelerator_type=AcceleratorNumba,
    policy=GeneratorPolicy(active=True),
)

EngineNative = EngineFactory(
    name="EngineNative",
    carrier_type=CarrierNativeArray,
    default_dtype="d",
    accelerator_type=AcceleratorNumba,
    policy=GeneratorPolicy(active=True),
)

try:
    import cupy as cp

    from stark.engines.carrier_cupy import CarrierCupy
except ImportError:
    pass
else:
    EngineCupy = EngineFactory(
        name="EngineCupy",
        carrier_type=CarrierCupy,
        default_dtype=cp.float64,
        accelerator_type=AcceleratorNone,
        policy=GeneratorPolicy(
            active=True,
            traversal="elementwise",
            expression="elementwise",
            scalar="item",
        ),
    )

try:
    from stark.engines.accelerators.jax import AcceleratorJax
    from stark.engines.carrier_jax import CarrierJax
except ImportError:
    pass
else:
    EngineJax = EngineFactory(
        name="EngineJax",
        carrier_type=CarrierJax,
        default_dtype=None,
        accelerator_type=AcceleratorJax,
        policy=GeneratorPolicy(
            active=True,
            mutation="functional",
            traversal="vectorized",
            expression="array_expression",
        ),
    )


__all__ = [
    "Engine",
    "EngineCarrierTypeLike",
    "EngineFactory",
    "EngineNative",
    "EngineNumpy",
]

if "EngineCupy" in globals():
    __all__.append("EngineCupy")

if "EngineJax" in globals():
    __all__.append("EngineJax")
