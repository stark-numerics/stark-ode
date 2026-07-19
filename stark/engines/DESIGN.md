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
engines/allocator
engines/engine.py
engines/engine_allocator.py
engines/engine_translation.py
engines/translation_basis.py
engines/accelerators
engines/generator
engines/carriers
engines/carrier_numpy
engines/carrier_native
engines/carrier_cupy
engines/carrier_jax
engines/carrier_torch    future
```

Backend-specific carriers, targets, and policies should live near the backend
they serve. `engines/engine.py` owns the single concrete `Engine` dataclass and
the `EngineFactory` presets such as `EngineNumpy`, `EngineNative`,
`EngineCupy`, and `EngineJax`. Cross-backend construction helpers such as
`EngineAllocator` and `EngineTranslation` live at `engines/` level because
their ownership is engine wide and they are parameterised by backend-specific
carriers. Shared abstractions belong in named packages such as
`engines/generator`, `engines/accelerators`, or `engines/carriers`; avoid a
vague extra `shared` layer.

Backend identity should remain visible in the engine factory preset and
concrete carrier package, not by duplicating one engine implementation per
backend.

## Role

An engine owns:

- state and translation carriers,
- allocator configuration,
- backend array type and dtype policy,
- carrier first-binding from frame field shape and dtype,
- engine-owned translations,
- translation bases,
- prepared Generator request surface,
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
target/carrier path rather than in generic engine translation code.

JAX stores shaped fields in immutable JAX arrays. Generated algebra should
prefer functional return-style kernels; whole-solver JIT is a separate future
boundary.

`EngineTranslation` chooses return-style or into-style fallback arithmetic from
carrier metadata during construction. Backend-specific translation classes
should not be added unless carrier metadata cannot express a genuine backend
semantic difference.

Carrier classes own initial binding from frame shape and dtype through
`from_shape`. Once a carrier exists, its `allocation` object owns repeated
state/translation allocation. Engines should not know backend-specific zero
template construction details.

Every carrier class used by `EngineFactory` should expose `resolve_dtype` as
well as `from_shape`. The shared engine asks the carrier to normalize its dtype
argument before allocating field carriers; backend-specific dtype rules belong
to the carrier family.

Generator policy belongs to the generator. Backend factory presets may choose
the policy they pass during construction, but a prepared engine should expose
it as `engine.generator.policy`, not as a parallel `engine.generator_policy`
attribute.

`engine.generator.policy.active` is the opt-in signal for automatic generated
specialization. STARK-owned engine factories set it to true. A user-defined
engine can still carry a generator-shaped object for inspection or direct
advanced use while leaving `active` false so system construction stays on
conservative runtime paths.

NumPy and Native default to `AcceleratorNumba`. Missing Numba should fail at
accelerator construction with the accelerator's own error message. Use
`AcceleratorNone` or an explicitly named unaccelerated engine preset when
unaccelerated execution is desired.

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
