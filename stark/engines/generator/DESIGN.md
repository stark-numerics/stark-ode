# Generator Design Notes

Last reviewed: 2026-07-17.

This package is a staging area for a clearer successor shape to the older
engine algebra-generation machinery. It must not import from that machinery.
The goal is to make the engine's code-generation decisions visible enough for
stark-ode and later stark-pde generators to share the same conceptual surface.

## Goal

`Generator` receives a `FrameLike`, an accelerator, and optionally a
`GeneratorPolicyLike`. Its call surface receives a duck-typed
`GeneratorRequestLike` and dispatches on `request.operation`.

Unknown operations are programmer errors:

```text
unknown operation -> NotImplementedError
```

Known operations emit and compile optimized kernels:

```text
known operation -> source emission and compilation
```

Runtime-safe allocation helpers live in the allocator package. Generator is
reserved for emitted or otherwise optimized kernels. Concrete engines construct
a generator when they have enough frame/backend information for codegen, while
custom allocators can become complete enough to run without constructing a
generator.

## Request Shape

The base request contract is deliberately tiny:

```text
operation: str
```

Operation-specific protocols describe the additional data required by that
operation. For example, `SchemeStencil` is still a stencil, but it can also
serve as a generator request because it exposes `operation == "linear_fixed"`
alongside its coefficients, scale, and apply flag.

Block-level fixed-linear helpers follow the same shape. `BlockLinearFixed`
lifts any callable item provider that accepts a linear-fixed request; it does
not require the older split provider methods such as `provide_delta` and
`provide_apply`.

This keeps duck typing alive. A PDE request can use the same dispatch shape
without making stark-ode know about PDE-specific concepts.

## Policy

`GeneratorPolicy` is a plain dataclass. It should stay explicit: callers set
the active flag, traversal, mutation, expression, and scalar choices directly
instead of selecting from convenience constructors. `Generator` owns a boring
inactive default policy; concrete STARK engine factories opt in with
`active=True` when they are prepared to use generator-backed specialization.
That default can be reset with `Generator.reset_default_policy(...)`.

`active` is not a runtime/codegen mode. Runtime-safe fallbacks live in the
allocator package. `active` only says whether higher-level construction paths
should automatically pass this generator on to schemes, resolvents, inverters,
or downstream assemblers. A generator can still be called directly when the
flag is false.

The policy does not own:

```text
frame
carriers
allocator
accelerator
engine translation
```

Those remain engine resources.

## Generated Code

The code-writing stage should be laid out as separate, inspectable steps:

```text
request validation / lowering
frame layout pass
operation-owned source building
target wrapper
accelerator compilation
```

Frame traversal should be centralized where possible. Emitting one coherent
loop over frame fields is preferable to composing many small generated loops
that walk the same frame repeatedly.

`linear_fixed`, `linear_combine`, and generated `norm` currently perform small
layout passes before source emission. Scalar and broadcast fields are emitted
directly. Non-vectorized looped fields are grouped by rank and concrete shape,
so fields with the same loop bounds are written inside one shared loop nest.

## Current Draft

The current package is intentionally independent and incomplete:

- `linear_combine`, `linear_fixed`, `apply_translation`, `norm`, and
  `inner_product` can emit generated source without importing the older engine
  algebra machinery.
- `linear_fixed` owns its source builder in `linear_fixed_source`; future
  operation families should prefer their own source builders over a shared
  mega-emitter.
- `linear_fixed_source` groups compatible looped fields before emission for
  both fixed-coefficient and runtime-coefficient linear kernels. Generated
  norm source performs the same grouping for adaptive-method hot paths.
- `elementwise` policy options describe backend elementwise generation without
  making request objects backend-specific. The first concrete implementation
  emits CuPy `ElementwiseKernel` source, but the request remains
  `linear_combine`, `linear_fixed`, or `apply_translation`.

Concrete engines now install allocator `apply_translation`, `norm`, and
`inner_product` hooks through Generator requests. NumPy, Native, JAX, and CuPy
also install `linear_combine` through Generator. Engines no longer expose an
`algebraist` bundle.

## Former Algebraist Capabilities

Migration readiness is judged by capabilities, not imports. The old algebra
generation bundle is gone from the engine surface; these responsibilities now
belong to Generator or allocator hooks:

```text
linear_combine       Generator emits in-place, functional/vectorized, and
                     elementwise linear-combine kernels. CuPy uses the generic
                     elementwise policy shape backed by CuPy ElementwiseKernel
                     source.

linear_fixed         Generator handles request-based delta/apply kernels,
                     including backend-specific elementwise and functional
                     variants, and the apply_translation hook used by
                     engine translations.

apply_translation    Generator owns allocator.apply_translation through a
                     standalone request. This is intentionally separate from
                     linear_fixed: it applies an existing translation to an
                     origin state and has no step argument or coefficient
                     stencil.

norm                 Generator owns engine allocator hooks now. Confirm backend
                     scalar-return behavior remains covered before deleting
                     corresponding `_algebraist` providers.

inner_product        Generator owns engine allocator hooks now. Confirm backend
                     scalar-return behavior remains covered before deleting
                     corresponding `_algebraist` providers.

runtime fallback     Allocator decorators are the compatibility path for
                     foreign state/translation families. Runtime-safe
                     completion of custom allocators lives with the allocator
                     design rather than inside generated-code machinery.
```

## Allocator Linear-Combine Hooks

Allocators may expose custom `linear_combine` kernels. This is an intentional
advanced-user bypass: a user who supplies their own state and translation
family might also supply compiled combine kernels that know how to operate on
that family efficiently.

Generator treats allocator-provided kernels as input semantics, not as the
primary engine installation mechanism. Runtime synthesis of missing arities
belongs to `Allocator.runtime`; generated linear-combine requests emit kernels
from the frame and policy.

Allocator preparation should stay out of the base allocator contract. Optional
runtime setup helpers live on classes decorated with `Allocator.runtime`. This
mirrors the problem-layer dynamics style: the user keeps their own object shape
and opts into STARK-recognised setup behavior with a decorator rather than a
base class.

`Allocator.runtime` prepares a complete table during allocator construction:

```python
@Allocator.runtime
@Allocator.linear_combine(scale, combine2)
class MyAllocator:
    ...

allocator = MyAllocator(...)
```

The `Allocator.linear_combine(...)` decorator is optional. Without a generator,
the allocator runtime path uses declared low-arity seeds or translation
arithmetic. Generated hook allocation is explicit: either the engine asks its
`Generator` directly, or a class is decorated with a specific bound generator:

```python
@Allocator.generated(generator)
class MyGeneratedAllocator:
    ...
```

Concrete engines can therefore install a complete arity table through
`Generator` without depending on the older algebra machinery, while custom
allocators can still define the low-level linear-combination semantics that
runtime allocator setup should respect. Advanced users can pass custom factories to
`Allocator.runtime(...)` or `Allocator.generated(generator, ...)` when they
want hand-optimised hook allocation instead of the default allocator/generator
helpers.
