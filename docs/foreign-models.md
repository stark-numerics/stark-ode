# Integrate a foreign model

This page is for users whose model already has its own state objects and solver increments.

Use the high-level `System`/`Frame` path when your state can be represented as
named fields. Use this page when that would be artificial because the model
already owns its state and increment objects.

## The idea

STARK separates:

```text
state        model configuration
translation  solver increment / tangent object
allocator    factory for states, translations, and scratch
```

A foreign model integration provides these objects directly.

## Nesting is still a Frame job

Nested data does not require a custom allocator. A `Frame` path can name nested
fields such as `model.particle.position`, and the engine will allocate matching
state and translation objects. Prefer that route when the structure is just
named scalar or array data.

Custom state and custom allocators are for preserving an existing object model:
objects with constructor requirements, invariants, non-array storage,
external resources, or behavior that would be damaged by asking STARK to own
the storage.

## Runnable examples

Start with the preferred high-level route:

```powershell
python -m examples.problem.structured_state_minimal
```

That example shows a nested state represented through named `Frame` paths. It
is still structured, but STARK owns the generated state and translation objects.

Then compare the lower-level custom allocator route:

```powershell
python -m examples.problem.foreign_model_allocator
```

That example shows the extra work needed when a model already owns its state
and increment classes.

Finally, use the plug-in solver example when you want to see STARK replacing an
existing stepper without replacing the model:

```powershell
python -m examples.problem.foreign_model_plug_in_solver
```

## What the translation must do

A translation must be able to:

```text
apply itself to a state
be scaled and combined
report a norm when adaptive schemes need one
```

For simple foreign objects, Python special methods may be enough. For high-performance foreign models, provide explicit operations through an allocator or engine-specific path.

## What the allocator must do

An allocator gives schemes and integrators fresh objects without knowing the model's constructor details.

At minimum:

```text
allocate_state
copy_state
allocate_translation
```

Implicit methods and advanced inverters may require additional operator or block allocation support.

## When to prefer `Frame`

Use `Frame` if your model can be described as named array/scalar fields, even
when those fields are nested. The high-level path gives STARK more structure,
which enables generated Algebraist kernels and backend acceleration.

Use custom state/translation only when preserving the foreign model representation is more important than the generated high-level path.

## Next

- [Concepts and terminology](concepts.md) explains the state/translation model.
- [Contract maths](contract-maths.md) gives the formal low-level view.
- [Extending STARK](extending.md) explains method components such as schemes and inverters.
- `stark/engines/shared/algebraist/DESIGN.md` explains why `Frame`-backed models can use generated kernels while foreign models may need runtime fallback.
