from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import jax  # type: ignore[import-not-found]
import jax.numpy as jnp  # type: ignore[import-not-found]

from stark.core.contracts.accelerator import Accelerator
from stark.engines.shared.accelerators import AcceleratorJax
from stark.engines.shared.algebraist.arity import AlgebraistArity
from stark.engines.shared.algebraist.frame import (
    AlgebraistFrame,
    AlgebraistFrameLooped,
)
from stark.engines.shared.algebraist.generator import (
    AlgebraistGeneratorInnerProduct,
    AlgebraistGeneratorLinearCombine,
    AlgebraistGeneratorNorm,
    AlgebraistGeneratorSpecialist,
    AlgebraistGeneratorTargetFunctional,
)
from stark.engines.jax.allocator import EngineAllocatorJax
from stark.engines.jax.carriers import CarrierJax
from stark.engines.jax.carriers.storage import CarrierJaxValue
from stark.engines.shared.basis import EngineTranslationBasis
from stark.problem.frame.frame import Frame


def _jax_x64_enabled() -> bool:
    return bool(getattr(jax.config, "jax_enable_x64", False))


def _resolve_jax_dtype(dtype: Any | None) -> Any:
    if dtype is None:
        if _jax_x64_enabled():
            return jnp.dtype(jnp.float64)
        return jnp.dtype(jnp.float32)

    requested = jnp.dtype(dtype)

    if _jax_x64_enabled():
        return requested

    if requested == jnp.dtype(jnp.float64):
        raise ValueError(
            "EngineJax(dtype=jnp.float64) requires JAX x64 support. "
            "Enable it with JAX_ENABLE_X64=1 or "
            "jax.config.update('jax_enable_x64', True), "
            "or pass dtype=jnp.float32."
        )

    if requested == jnp.dtype(jnp.complex128):
        raise ValueError(
            "EngineJax(dtype=jnp.complex128) requires JAX x64 support. "
            "Enable it with JAX_ENABLE_X64=1 or "
            "jax.config.update('jax_enable_x64', True), "
            "or pass dtype=jnp.complex64."
        )

    return requested


@dataclass(frozen=True, slots=True)
class EngineJax:
    """
    JAX backend bundle for a shaped `Frame`.

    An engine supplies the backend objects used when a `System` prepares an
    IVP: carrier templates for each frame field, an allocator for owned state
    and translation objects, the derived algebraist frame, generated algebra
    providers, and an accelerator. This engine keeps field storage in JAX arrays
    and uses functional generated kernels for update-style algebra.

    If `dtype` is omitted, the engine uses JAX's active default real precision:
    `float32` when x64 is disabled and `float64` when x64 is enabled. Complex
    problems should pass an explicit complex dtype, usually `jnp.complex64` or
    `jnp.complex128` with x64 enabled.
    """

    frame: Frame
    dtype: Any | None = None
    accelerator: Accelerator = field(default_factory=AcceleratorJax)
    algebraist_frame: AlgebraistFrame = field(init=False)
    carriers: tuple[CarrierJax, ...] = field(init=False, repr=False)
    allocator: EngineAllocatorJax = field(init=False, repr=False)
    algebraist_inner_product: AlgebraistGeneratorInnerProduct = field(init=False, repr=False)
    algebraist_linear_combine: AlgebraistGeneratorLinearCombine = field(init=False, repr=False)
    algebraist_norm: AlgebraistGeneratorNorm = field(init=False, repr=False)
    algebraist_specialist: AlgebraistGeneratorSpecialist = field(init=False, repr=False)

    def __repr__(self) -> str:
        accelerator_name = getattr(self.accelerator, "name", str(self.accelerator))
        return (
            f"{type(self).__name__}(frame={self.frame!r}, "
            f"dtype={self.dtype!r}, accelerator={accelerator_name!r})"
        )

    def translation_basis(self) -> EngineTranslationBasis:
        """Return the coordinate basis for engine-owned translations.

        This is an inspection and dense-materialisation helper. Ordinary user
        solves should not need it, but dense inverters and diagnostic examples
        can use it instead of hand-writing translation-basis classes.
        """

        return EngineTranslationBasis(self.algebraist_frame, self.carriers)

    def __post_init__(self) -> None:
        dtype = _resolve_jax_dtype(self.dtype)
        object.__setattr__(self, "dtype", dtype)

        algebraist_frame = self.frame.to_algebraist_frame()
        carriers: list[CarrierJax] = []

        for field in algebraist_frame.fields:
            policy = field.policy
            if not isinstance(policy, AlgebraistFrameLooped) or policy.shape is None:
                raise ValueError(
                    "EngineJax requires every frame field to declare shape."
                )
            template = cast(CarrierJaxValue, jnp.zeros(policy.shape, dtype=dtype))
            carriers.append(CarrierJax(template))

        carrier_tuple = tuple(carriers)
        allocator = EngineAllocatorJax(
            algebraist_frame=algebraist_frame,
            carriers=carrier_tuple,
        )

        object.__setattr__(self, "algebraist_frame", algebraist_frame)
        object.__setattr__(self, "carriers", carrier_tuple)
        object.__setattr__(self, "allocator", allocator)

        functional_target = AlgebraistGeneratorTargetFunctional()
        algebraist_linear_combine = AlgebraistGeneratorLinearCombine(
            translation=allocator.allocate_translation(),
            allocator=allocator,
            frame=algebraist_frame,
            accelerator=self.accelerator,
            target=functional_target,
        )
        object.__setattr__(
            allocator,
            "linear_combine",
            tuple(
                algebraist_linear_combine.provide(AlgebraistArity(arity))
                for arity in range(1, 13)
            ),
        )

        algebraist_specialist = AlgebraistGeneratorSpecialist(
            translation=allocator.allocate_translation(),
            allocator=allocator,
            frame=algebraist_frame,
            accelerator=self.accelerator,
            target=functional_target,
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
            target=functional_target,
        )
        object.__setattr__(allocator, "norm", algebraist_norm.provide())

        algebraist_inner_product = AlgebraistGeneratorInnerProduct(
            translation=allocator.allocate_translation(),
            frame=algebraist_frame,
            accelerator=self.accelerator,
            target=functional_target,
        )
        object.__setattr__(allocator, "inner_product", algebraist_inner_product.provide())

        object.__setattr__(self, "algebraist_linear_combine", algebraist_linear_combine)
        object.__setattr__(self, "algebraist_inner_product", algebraist_inner_product)
        object.__setattr__(self, "algebraist_norm", algebraist_norm)
        object.__setattr__(self, "algebraist_specialist", algebraist_specialist)


__all__ = ["EngineJax"]
