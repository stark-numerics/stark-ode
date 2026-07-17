"""Generate specialised algebra kernels for frame-backed engines.

STARK's public problem model is deliberately structural: a problem describes
named state fields, translation fields, norms, inner products, and traversal
policies. That structure is pleasant for users and useful for backend
portability, but it is too much bookkeeping to repeat inside every numerical
step. The hottest solver operations are often simple algebra operations:

```text
translation linear combinations
fixed-coefficient scheme stage updates
translation application
norms
inner products
```

A completely generic implementation can perform those operations by inspecting
the frame on every call, branching on field policy, looping over fields one at
a time, and delegating to backend arrays dynamically. That is flexible, but it
puts Python control flow in exactly the place where adaptive schemes,
resolvents, and inverters are most sensitive to overhead.

``Generator`` is the engine-side answer to that problem. It performs the
structural decisions once, when an engine has already chosen a frame,
allocator, backend policy, and accelerator. A request such as
``linear_combine`` or ``norm`` is lowered into ordinary Python source code
specialised to that frame layout. The resulting callable can then be compiled
by the engine accelerator, or used directly when no compiler is active.

This separation is intentional. Generator decides what code should exist:
argument order, field traversal, in-place versus functional updates, and
backend expression shape. The accelerator decides what to do with that code:
leave it as Python, JIT-compile it, cache it, or hand it to a backend fuser.
Keeping those roles separate lets the same request surface work with
``AcceleratorNone`` during development, Numba for CPU kernels, and backend
accelerators such as JAX or CuPy where the generated expression shape matters.

The generated functions are not opaque magic. They are normal Python callables
with source that can be inspected in tests or during development. The point is
not that generated Python is inherently faster than handwritten Python; the
point is that the generated function can remove repeated structural
bookkeeping from hot loops. For example, a generated linear-combination kernel
can know up front which fields are scalars, which arrays share loop bounds,
which backend prefers in-place writes, and which scalar conversions are needed
at the Python boundary.

The request surface stays intentionally small. Each request exposes an
``operation`` string, and operation-specific request protocols provide the
extra data needed by that operation. This keeps duck typing available:
``SchemeStencil`` can also be a valid ``linear_fixed`` request because it
already carries the coefficients, fixed scale, apply flag, and operation name.
Downstream packages can add their own request objects without inheriting from
STARK internals.

Most users should not need to instantiate ``Generator`` by hand. Concrete
engines create one and install the prepared kernels onto their allocators and
translations. Advanced users may call a generator directly when they are
building a custom allocator, comparing generated source, or preparing kernels
for a custom state/translation family.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, overload, cast

from stark.core.contracts.accelerator import Accelerator
from stark.core.contracts.allocator import AllocatorLike
from stark.core.contracts.frame import FrameLike
from stark.core.contracts.state import StateType
from stark.core.contracts.translation import TranslationType
from stark.engines.accelerators.none import AcceleratorNone
from stark.engines.generator.inner_product import GeneratorInnerProduct
from stark.engines.generator.linear_combine import GeneratorLinearCombine
from stark.engines.generator.linear_fixed import GeneratorLinearFixed
from stark.engines.generator.norm import GeneratorNorm
from stark.engines.generator.policy import GeneratorPolicy, GeneratorPolicyLike
from stark.engines.generator.request import (
    GeneratorRequestApplyTranslationLike,
    GeneratorRequestInnerProductLike,
    GeneratorRequestLike,
    GeneratorRequestLinearCombine,
    GeneratorRequestLinearCombineLike,
    GeneratorRequestLinearCombineTableLike,
    GeneratorRequestLinearFixedLike,
    GeneratorRequestNormLike,
)


@dataclass(slots=True)
class Generator(Generic[StateType, TranslationType]):
    """Build specialized algebra kernels from duck-typed operation requests.

    A ``Generator`` is the engine-facing dispatcher for prepared algebra. It
    owns no state arrays itself. Instead, it receives the pieces needed to
    write compatible kernels:

    ``frame``
        The ``FrameLike`` layout describing which state and translation fields
        exist, how to access them, their shapes, and their traversal policies.

    ``accelerator``
        The compiler/fuser used after source emission. ``AcceleratorNone``
        leaves the generated Python callable uncompiled.

    ``policy``
        Source-shape choices such as in-place versus functional updates,
        looped versus vectorized traversal, and backend scalar handling.

    ``allocator``
        Optional allocator context for operation families that need to allocate
        or match the engine's translation objects.

    Calls use a deliberately small request protocol. Every request exposes an
    ``operation`` string; operation-specific request protocols add the data the
    selected generator needs, such as ``arity`` for ``linear_combine`` or
    fixed coefficients for ``linear_fixed``. This keeps the surface friendly to
    duck typing: a scheme stencil can be a valid generator request without
    inheriting from a generator base class.

    The returned value is usually a callable kernel. ``linear_combine_table``
    returns a tuple of kernels for arities ``1..max_arity`` so allocators can
    install a complete fast linear-combination table in one step.
    """

    frame: FrameLike
    accelerator: Accelerator = field(default_factory=AcceleratorNone)
    policy: GeneratorPolicyLike = field(default_factory=GeneratorPolicy)
    allocator: AllocatorLike[StateType, TranslationType] | None = None
    linear_combine: GeneratorLinearCombine[StateType, TranslationType] = field(init=False)
    linear_fixed: GeneratorLinearFixed[StateType, TranslationType] = field(init=False)
    norm: GeneratorNorm[TranslationType] = field(init=False)
    inner_product: GeneratorInnerProduct[TranslationType] = field(init=False)

    def __post_init__(self) -> None:
        self.linear_combine = GeneratorLinearCombine(
            frame=self.frame,
            accelerator=self.accelerator,
            policy=self.policy,
            allocator=self.allocator,
        )
        self.linear_fixed = GeneratorLinearFixed(
            frame=self.frame,
            accelerator=self.accelerator,
            policy=self.policy,
            allocator=self.allocator,
        )
        self.norm = GeneratorNorm(
            frame=self.frame,
            accelerator=self.accelerator,
            policy=self.policy,
        )
        self.inner_product = GeneratorInnerProduct(
            frame=self.frame,
            accelerator=self.accelerator,
            policy=self.policy,
        )

    @overload
    def __call__(
        self,
        request: GeneratorRequestApplyTranslationLike,
    ) -> Callable[..., object]:
        ...

    @overload
    def __call__(
        self,
        request: GeneratorRequestLinearCombineTableLike,
    ) -> tuple[Callable[..., object], ...]:
        ...

    @overload
    def __call__(
        self,
        request: GeneratorRequestLinearCombineLike,
    ) -> Callable[..., object]:
        ...

    @overload
    def __call__(
        self,
        request: GeneratorRequestLinearFixedLike,
    ) -> Callable[..., object]:
        ...

    @overload
    def __call__(
        self,
        request: GeneratorRequestNormLike,
    ) -> Callable[..., object]:
        ...

    @overload
    def __call__(
        self,
        request: GeneratorRequestInnerProductLike,
    ) -> Callable[..., object]:
        ...

    @overload
    def __call__(
        self,
        request: GeneratorRequestLike,
    ) -> Callable[..., object] | tuple[Callable[..., object], ...]:
        ...

    def __call__(
        self,
        request: GeneratorRequestLike,
    ) -> Callable[..., object] | tuple[Callable[..., object], ...]:
        match request.operation:
            case "apply_translation":
                return self.linear_fixed.apply_translation(
                    cast(GeneratorRequestApplyTranslationLike, request)
                )
            case "linear_combine":
                return self.linear_combine(cast(GeneratorRequestLinearCombineLike, request))
            case "linear_combine_table":
                return tuple(
                    self.linear_combine(GeneratorRequestLinearCombine(arity=arity))
                    for arity in range(
                        1,
                        cast(GeneratorRequestLinearCombineTableLike, request).max_arity + 1,
                    )
                )
            case "linear_fixed":
                return self.linear_fixed(cast(GeneratorRequestLinearFixedLike, request))
            case "norm":
                return self.norm(cast(GeneratorRequestNormLike, request))
            case "inner_product":
                return self.inner_product(cast(GeneratorRequestInnerProductLike, request))
            case _:
                raise NotImplementedError(
                    f"Unknown generator operation: {request.operation!r}"
                )


__all__ = ["Generator"]
