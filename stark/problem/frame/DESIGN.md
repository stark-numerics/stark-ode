# Frame Design Notes

`Frame` is the named schema for high-level state and translation fields.

It is the preferred route for ordinary user models because it gives STARK
enough structure to generate carriers, translations, allocators, norms, and
prepared algebra.

## Role

A frame connects:

```text
state field        y
translation field  dy
shape              scalar/vector/array/nested field shape
norm policy        how adaptive errors are measured
```

Convenience constructors such as `Frame.scalar`, `Frame.vector`,
`Frame.array`, and `Frame.from_fields` are concise spellings of the fuller
mapping syntax. Documentation should make that equivalence visible.

## Why Frame Comes Before Custom Allocators

Nested or structured data does not automatically require custom state classes.
If a model can be described as named scalar or array fields, use `Frame` first.

Custom allocators are for foreign models with constructor requirements,
invariants, external resources, or behaviour that would be damaged by asking
STARK to own the storage.

## Norm Policy

Frame fields own their contribution to translation norms. This lets users carry
diagnostic or auxiliary fields without forcing those fields into adaptive error
control.

## Design Rule

Frame should expose the model shape clearly enough for users and structured
enough for engines. Avoid pushing users toward low-level contracts when a named
frame can express the state.
