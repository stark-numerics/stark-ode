"""Decorator utilities for making custom allocators STARK-ready.

User-defined states are one of STARK's escape hatches: a model can keep its
own state and translation objects instead of being forced into an engine-owned
array layout. The small contract for that path is `AllocatorLike`: allocate a
state, copy a state, and allocate a translation.

The rest of the method machinery needs a little more structure. Schemes need a
table of linear-combination kernels, adaptive methods need a norm, and some
solvers or diagnostics may need an inner product. This module adds those
capabilities with decorators instead of inheritance:

```python
@Allocator.runtime
class MyAllocator:
    ...
```

`Allocator.runtime` keeps ordinary custom allocators light. After each
allocator instance is constructed it prepares a complete `linear_combine`
table, using any optional low-arity seeds or falling back to translation
arithmetic. Users can add fast handwritten kernels without taking over the
whole setup:

```python
@Allocator.runtime
@Allocator.linear_combine(scale, combine2)
@Allocator.inner_product(inner_product)
class MyAllocator:
    ...
```

The seed decorators are optional. A basic allocator can omit all of them if its
translation supports `__rmul__`, `__add__`, and `norm()`. More specialised
models can provide selected linear-combine or inner-product hooks where they
have a faster or more meaningful implementation.

`Allocator.generated(generator)` is the companion path for engines that already
have a frame-aware generator. It installs generated hook factories rather than
runtime fallbacks, while preserving the same allocator shape.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, TypeAlias, TypeVar, cast, overload

from stark.core.contracts.linear_combine import LinearCombine
from stark.core.contracts.translation import Translation
from stark.engines.generator.request import GeneratorLike
from stark.engines.generator.request import (
    GeneratorRequestInnerProduct,
    GeneratorRequestLinearCombineTable,
    GeneratorRequestNorm,
)
from stark.engines.allocator.linear_combine import (
    AllocatorRuntimeLinearCombine,
    AllocatorRuntimeLinearCombineAllocator,
)

AllocatorLinearCombineTableFactory: TypeAlias = Callable[
    [object, int],
    LinearCombine,
]
AllocatorGeneratedLinearCombineTableFactory: TypeAlias = Callable[
    [object, GeneratorLike, int],
    LinearCombine,
]
AllocatorKernelFactory: TypeAlias = Callable[[object], Callable[..., object]]
AllocatorGeneratedKernelFactory: TypeAlias = Callable[
    [object, GeneratorLike],
    Callable[..., object],
]
AllocatorClassType = TypeVar("AllocatorClassType", bound=type[object])


class AllocatorGeneratedHooks(Protocol):
    """Allocator instance decorated with generated hook helpers."""

    def allocate_linear_combine_table(
        self,
        *,
        max_arity: int = 12,
    ) -> LinearCombine:
        ...

    def allocate_norm(self) -> Callable[..., object]:
        ...

    def allocate_inner_product(self) -> Callable[..., object]:
        ...


class AllocatorRuntimeHooks(Protocol):
    """Allocator instance decorated with runtime hook helpers."""

    def allocate_linear_combine_table(
        self,
        *,
        max_arity: int = 12,
    ) -> LinearCombine:
        ...

    def allocate_inner_product(self) -> Callable[..., object]:
        ...


class Allocator:
    """Namespace for allocator decorators.

    A custom allocator should remain the user's object. These decorators
    attach STARK-recognised setup behavior to that object without changing its
    type, requiring a base class, or forcing engine-owned storage conventions.

    The typical custom-state path is `@Allocator.runtime`. It guarantees that
    a constructed allocator has the linear-combination table expected by
    schemes. Optional `linear_combine` and `inner_product` decorators seed
    runtime setup with user-written kernels.

    The engine path is `@Allocator.generated(generator)`, or equivalently an
    engine assigning generated hooks to its allocator instance. That path uses
    the same hook names but asks a generator to produce kernels from frame and
    backend information.
    """

    @staticmethod
    def generated(
        generator: GeneratorLike,
        *,
        linear_combine_table: AllocatorGeneratedLinearCombineTableFactory | None = None,
        norm: AllocatorGeneratedKernelFactory | None = None,
        inner_product: AllocatorGeneratedKernelFactory | None = None,
    ) -> Callable[[AllocatorClassType], AllocatorClassType]:
        """Decorate an allocator class with generator-backed hook allocators."""

        linear_combine_table_factory = (
            linear_combine_table
            if linear_combine_table is not None
            else Allocator.allocate_linear_combine_table_generated
        )
        norm_factory = norm if norm is not None else Allocator.allocate_norm_generated
        inner_product_factory = (
            inner_product
            if inner_product is not None
            else Allocator.allocate_inner_product_generated
        )

        def decorate(allocator_type: AllocatorClassType) -> AllocatorClassType:
            setattr(
                allocator_type,
                "allocate_linear_combine_table",
                lambda self, *, max_arity=12: linear_combine_table_factory(
                    self,
                    generator,
                    max_arity,
                ),
            )
            setattr(
                allocator_type,
                "allocate_norm",
                lambda self: norm_factory(self, generator),
            )
            setattr(
                allocator_type,
                "allocate_inner_product",
                lambda self: inner_product_factory(self, generator),
            )
            return allocator_type

        return decorate

    @staticmethod
    def linear_combine(*kernels: Callable[..., object]) -> Callable[[AllocatorClassType], AllocatorClassType]:
        """Decorate an allocator class with optional linear-combine seed kernels."""

        def decorate(allocator_type: AllocatorClassType) -> AllocatorClassType:
            setattr(allocator_type, "linear_combine", cast(LinearCombine, kernels))
            return allocator_type

        return decorate

    @staticmethod
    def inner_product(
        kernel: Callable[..., object],
    ) -> Callable[[AllocatorClassType], AllocatorClassType]:
        """Decorate an allocator class with a custom runtime inner-product kernel."""

        def decorate(allocator_type: AllocatorClassType) -> AllocatorClassType:
            setattr(allocator_type, "inner_product", staticmethod(kernel))
            return allocator_type

        return decorate

    @overload
    @staticmethod
    def runtime(allocator_type: AllocatorClassType) -> AllocatorClassType:
        ...

    @overload
    @staticmethod
    def runtime(
        allocator_type: None = None,
        *,
        max_arity: int = 12,
        linear_combine_table: AllocatorLinearCombineTableFactory | None = None,
        inner_product: AllocatorKernelFactory | None = None,
    ) -> Callable[[AllocatorClassType], AllocatorClassType]:
        ...

    @staticmethod
    def runtime(
        allocator_type: AllocatorClassType | None = None,
        *,
        max_arity: int = 12,
        linear_combine_table: AllocatorLinearCombineTableFactory | None = None,
        inner_product: AllocatorKernelFactory | None = None,
    ) -> Callable[[AllocatorClassType], AllocatorClassType] | AllocatorClassType:
        """Decorate an allocator class with runtime hook allocators."""

        linear_combine_table_factory = (
            linear_combine_table
            if linear_combine_table is not None
            else Allocator.allocate_linear_combine_table_runtime
        )
        inner_product_factory = (
            inner_product
            if inner_product is not None
            else Allocator.allocate_inner_product_runtime
        )

        def decorate(selected_type: AllocatorClassType) -> AllocatorClassType:
            original_init = cast(Callable[..., None], selected_type.__init__)

            setattr(
                selected_type,
                "allocate_linear_combine_table",
                lambda self, *, max_arity=12: linear_combine_table_factory(
                    self,
                    max_arity,
                ),
            )
            setattr(
                selected_type,
                "allocate_inner_product",
                lambda self: inner_product_factory(self),
            )

            def __init__(self: object, *args: object, **kwargs: object) -> None:
                original_init(self, *args, **kwargs)
                runtime_hooks = cast(AllocatorRuntimeHooks, self)
                setattr(
                    self,
                    "linear_combine",
                    runtime_hooks.allocate_linear_combine_table(max_arity=max_arity),
                )
                if (
                    callable(getattr(self, "inner_product", None))
                    or inner_product is not None
                ):
                    setattr(self, "inner_product", runtime_hooks.allocate_inner_product())

            setattr(selected_type, "__init__", __init__)
            return selected_type

        if allocator_type is not None:
            return decorate(allocator_type)
        return decorate

    @staticmethod
    def allocate_linear_combine_table_generated(
        allocator: object,
        generator: GeneratorLike,
        max_arity: int,
    ) -> LinearCombine:
        del allocator
        return generator(GeneratorRequestLinearCombineTable(max_arity=max_arity))

    @staticmethod
    def allocate_norm_generated(
        allocator: object,
        generator: GeneratorLike,
    ) -> Callable[..., object]:
        del allocator
        return generator(GeneratorRequestNorm())

    @staticmethod
    def allocate_inner_product_generated(
        allocator: object,
        generator: GeneratorLike,
    ) -> Callable[..., object]:
        del allocator
        return generator(GeneratorRequestInnerProduct())

    @staticmethod
    def allocate_linear_combine_table_runtime(
        allocator: object,
        max_arity: int,
    ) -> LinearCombine:
        seed = getattr(allocator, "linear_combine", ())
        translation_allocator = cast(
            AllocatorRuntimeLinearCombineAllocator[Translation],
            allocator,
        )
        linear_combine = cast(Sequence[Callable[..., Translation]], seed)
        return AllocatorRuntimeLinearCombine(
            allocator=translation_allocator,
            linear_combine=linear_combine,
        ).table(max_arity)

    @staticmethod
    def allocate_inner_product_runtime(
        allocator: object,
    ) -> Callable[..., object]:
        existing = getattr(allocator, "inner_product", None)
        if callable(existing):
            return existing
        raise ValueError(
            "Runtime inner-product allocation requires an explicit "
            "allocator.inner_product hook or a custom "
            "Allocator.runtime(inner_product=...) factory. Use "
            "@Allocator.generated(generator) when a generated frame-aware "
            "inner product is available."
        )


__all__ = [
    "Allocator",
    "AllocatorGeneratedHooks",
    "AllocatorGeneratedKernelFactory",
    "AllocatorGeneratedLinearCombineTableFactory",
    "AllocatorKernelFactory",
    "AllocatorLinearCombineTableFactory",
    "AllocatorRuntimeHooks",
]
