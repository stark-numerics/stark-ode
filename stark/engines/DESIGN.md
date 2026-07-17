# Engine Design Notes

Engines choose where STARK stores arrays and performs algebra.

The engine layer should keep backend ownership visible without forcing every
backend through a deep package tree.

## To Do

- Add a Torch engine/carrier pair when there is a real user or
  downstream-package need for it.

## Topology

The intended shape is flat enough that backend ownership is visible:

```text
engines/allocator.py
engines/engine_*.py
engines/translation_factory_*.py
engines/translation_basis.py
engines/accelerators
engines/_algebraist
engines/generator
engines/carriers
engines/carrier_numpy
engines/carrier_native
engines/carrier_cupy
engines/carrier_jax
engines/carrier_torch    future
```

Backend-specific carriers, targets, and policies should live near the backend
they serve. Cross-backend construction helpers such as the shared allocator and
translation factories can live at `engines/` level when their role is visible
and they are parameterised by backend-specific carriers. Shared abstractions
belong in named packages such as `engines/generator`, `engines/accelerators`,
or `engines/carriers`; avoid a vague extra `shared` layer. `_algebraist` is a
private transition package, not a public engine surface.

Backend identity should remain visible in the engine module and concrete
carrier package.

## Role

An engine owns:

- state and translation carriers,
- allocator configuration,
- backend array type and dtype policy,
- translation bases,
- prepared Generator request surface,
- generator policy describing source-shape decisions,
- accelerator choice where relevant.

## Generated and Runtime Algebra

Known `Frame`-backed state should prefer request-driven Generator paths.
Runtime algebra is the fallback for unknown or foreign user state shapes.

This matters for performance and for mental model clarity: ordinary high-level
models should give engines enough structure to optimise.

## Backend Notes

NumPy is the reference array backend for ordinary high-level use. It should
remain predictable, easy to inspect, and representative of the array backend
shape.

Native stores one-dimensional fields in Python `array.array` objects. It is a
dependency-light CPU path and currently rejects multi-dimensional frame fields.

CuPy stores shaped fields in GPU arrays. CuPy-specific expression choices, such
as elementwise kernel generation and host scalar extraction, belong in the CuPy
target/carrier path rather than in generic translation code.

JAX stores shaped fields in immutable JAX arrays. Generated algebra should
prefer functional return-style kernels; whole-solver JIT is a separate future
boundary.

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
