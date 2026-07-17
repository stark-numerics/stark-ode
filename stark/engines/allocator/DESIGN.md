# Allocator Design Notes

Last reviewed: 2026-07-17.

`AllocatorLike` is intentionally small. A user-defined allocator only needs to
allocate state, copy state, and allocate translations. Optional engine helpers
are added with decorators, not base classes.

## Decorator Shape

`Allocator.runtime` is a class decorator. It adds runtime-safe setup helpers
and prepares a complete `linear_combine` table after allocator construction
without changing the instance's identity:

```python
@Allocator.runtime
class MyAllocator:
    ...
```

The default runtime helper for `allocate_linear_combine_table` builds a full
table from existing low-arity hooks or from translation arithmetic without
requiring a `Generator`. `Allocator.runtime` calls it automatically during
instance initialization.

Optional low-arity linear-combine hooks are declared separately:

```python
@Allocator.runtime
@Allocator.linear_combine(scale, combine2)
class MyAllocator:
    ...
```

The seed table is an optimisation, not part of the minimal allocator contract.
Users can provide no seeds, one seed, or as many hand-written arities as are
worth maintaining.

Optional inner-product hooks use the same seed-decorator style:

```python
@Allocator.runtime
@Allocator.inner_product(inner_product)
class MyAllocator:
    ...
```

`Allocator.runtime` installs an inner-product hook on the instance after
construction only when it has been explicitly supplied. It does not invent a
default inner product.

Method implementations are allowed to assume this setup has already happened.
Schemes, resolvents, and inverters should consume `allocator.linear_combine`
directly once constructed; they should not extend arity tables or choose
generator/runtime behavior themselves.

Norm ownership stays with the translation object on the custom-state path. The
translation contract already requires `translation.norm()`, and schemes or
adaptive controllers should use that through the translation they receive. The
allocator `norm` hook exists for engine-carried translations that need an
injected generated kernel, not as ordinary custom allocator surface.

The default runtime helper for `allocate_inner_product` is intentionally stricter.
There is no core translation-level inner-product contract, so the decorator
will reuse an existing `allocator.inner_product` hook if one is present, or
raise a direct configuration error. It does not synthesize an inner product from
a norm.

Generated hook allocation is explicit. Use `Allocator.generated(generator)` when
a specific generator should be bound into the decorated class:

```python
@Allocator.generated(generator)
class MyGeneratedAllocator:
    ...
```

Advanced users can pass custom factories to `Allocator.runtime(...)` or
`Allocator.generated(generator, ...)` when they want to hand-optimise hook
allocation for a custom state or translation family.

## Ownership

Runtime fallback belongs here because it is part of making an allocator usable
with ordinary method machinery. Generator belongs on the optimised path: it can
be asked to produce faster kernels when frame/backend information is available,
but it is not required just to make a custom allocator complete enough to run.

`AllocatorCarried` is the concrete allocator for engine-owned carrier layouts.
It uses the same runtime decorator as user allocators, so carried engines and custom
state models share the same setup vocabulary.
