# Engine Design Notes

Engines choose where STARK stores arrays and performs algebra.

An engine package should be cohesive. A future backend such as Torch should
enter as a recognisable backend package, not as scattered helper files across
unrelated directories.

## Topology

The intended shape is flat enough that backend ownership is visible:

```text
engines/shared
engines/native
engines/numpy
engines/cupy
engines/jax
engines/torch    future
```

Backend-specific carriers, allocators, translations, targets, and policies
should live near the backend they serve. Shared abstractions belong in
`engines/shared` only when more than one backend genuinely uses them.

`engines/shared` should not become a second backend. Backend identity should
remain visible in the concrete backend package.

## Role

An engine owns:

- state and translation carriers,
- allocator behaviour,
- backend array type and dtype policy,
- translation bases,
- Algebraist/generator selection,
- accelerator choice where relevant.

## Generated and Runtime Algebra

Known `Frame`-backed state should prefer generated Algebraist paths. Runtime
algebra is the fallback for unknown or foreign user state shapes.

This matters for performance and for mental model clarity: ordinary high-level
models should give engines enough structure to optimise.

## Hint Types

Names with the `Hint` prefix are intentional type-checking scaffolding for
backend boundaries. They are not public STARK concepts and should not be
renamed into domain nouns.

Use `Hint*` protocols and aliases when a backend depends on a small, practical
surface of an optional external object. For example:

```text
HintJaxArray       the JAX array operations carriers actually use
HintCupyArray      the CuPy array operations carriers actually use
HintNativeNumber   the Python numeric operations native carriers actually use
```

When Pyright complains that an optional backend value lacks an operation, first
ask whether the nearest `Hint*` type should describe that operation. This keeps
casts and `Any` at the backend boundary instead of spreading them through hot
carrier code.

## Design Rule

When adding or changing a backend, ask whether a reader can inspect one package
and understand that backend's storage, arithmetic, and acceleration story. If
the answer is no, the topology is drifting.
