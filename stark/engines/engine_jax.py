from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import jax  # type: ignore[import-not-found]
import jax.numpy as jnp  # type: ignore[import-not-found]

from stark.core.contracts.accelerator import Accelerator
from stark.core.contracts.frame import FrameLike
from stark.engines.accelerators import AcceleratorJax
from stark.engines.allocator import AllocatorCarried
from stark.engines.carrier_jax import CarrierJax
from stark.engines.carrier_jax.storage import CarrierJaxValue
from stark.engines.generator import (
    Generator,
    GeneratorPolicy,
    GeneratorRequestApplyTranslation,
    GeneratorRequestInnerProduct,
    GeneratorRequestLinearCombineTable,
    GeneratorRequestNorm,
)
from stark.engines.translation_factory_jax import TranslationFactoryJax
from stark.engines.translation_basis import TranslationBasis


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


@dataclass(slots=True)
class EngineJax:
    """
    JAX backend bundle for a shaped `Frame`.

    An engine supplies the backend objects used when a `System` prepares an
    IVP: carrier templates for each frame field, an allocator for owned state
    and translation objects, generated algebra providers, and an accelerator.
    This engine keeps field storage in JAX arrays and uses functional generated
    kernels for update-style algebra.

    If `dtype` is omitted, the engine uses JAX's active default real precision:
    `float32` when x64 is disabled and `float64` when x64 is enabled. Complex
    problems should pass an explicit complex dtype, usually `jnp.complex64` or
    `jnp.complex128` with x64 enabled.
    """

    frame: FrameLike
    dtype: Any | None = None
    accelerator: Accelerator = field(default_factory=AcceleratorJax)
    carriers: tuple[CarrierJax, ...] = field(init=False, repr=False)
    allocator: AllocatorCarried[Any] = field(init=False, repr=False)
    generator: Generator[Any, Any] = field(init=False, repr=False)

    def __repr__(self) -> str:
        accelerator_name = getattr(self.accelerator, "name", str(self.accelerator))
        return (
            f"{type(self).__name__}(frame={self.frame!r}, "
            f"dtype={self.dtype!r}, accelerator={accelerator_name!r})"
        )

    def translation_basis(self) -> TranslationBasis:
        """Return the coordinate basis for engine-owned translations.

        This is an inspection and dense-materialisation helper. Ordinary user
        solves should not need it, but dense inverters and diagnostic examples
        can use it instead of hand-writing translation-basis classes.
        """

        return TranslationBasis(self.frame, self.carriers)

    def __post_init__(self) -> None:
        dtype = _resolve_jax_dtype(self.dtype)
        self.dtype = dtype

        carriers: list[CarrierJax] = []

        for field in self.frame.fields:
            policy = field.policy
            shape = getattr(policy, "shape", None)
            if shape is None:
                shape = getattr(field, "shape", None)
            if getattr(policy, "kind", None) != "looped" or shape is None:
                raise ValueError(
                    "EngineJax requires every frame field to declare shape."
                )
            template = cast(CarrierJaxValue, jnp.zeros(shape, dtype=dtype))
            carriers.append(CarrierJax(template))

        carrier_tuple = tuple(carriers)
        allocator: AllocatorCarried[Any] = AllocatorCarried(
            frame=self.frame,
            carriers=carrier_tuple,
            translation_type=TranslationFactoryJax,
        )

        self.carriers = carrier_tuple
        self.allocator = allocator

        generator = Generator(
            frame=self.frame,
            accelerator=self.accelerator,
            policy=GeneratorPolicy(
                mutation="functional",
                traversal="vectorized",
                expression="array_expression",
            ),
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


__all__ = ["EngineJax"]
