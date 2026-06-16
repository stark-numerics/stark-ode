# Foreign models and custom translations

This page is for users whose simulation already owns its state representation.

The high-level `Frame` path is convenient when your state can be represented as named fields. Some simulations already have richer objects: meshes, particles, field bundles, sparse structures, domain-specific arrays, or external model classes. In that case, flattening the model solely to call an ODE solver can make the code worse.

STARK supports this by separating state from translation.

## Core idea

A foreign-model integration provides:

- a **state** object: the nonlinear model value;
- a **translation** object: a linear increment/tangent value;
- an **allocator**: creates and copies states, translations, and scratch objects;
- a **derivative**: writes or returns a translation for `f(t, state)`;
- optionally a **linearizer**: applies `Df(t, state)` to a translation.

The solver uses translations for stage algebra. The model remains free to keep its own state structure.

## State versus translation

A state is not required to support addition or scalar multiplication. It may be a domain object with invariants.

A translation is linear. Schemes need to scale, combine, measure, and apply translations. That is why translations are separate from states.

For example, a model state might own a mesh and boundary metadata. A translation might store only field increments on that mesh.

## Allocators

The allocator is the boundary between generic solver code and model-specific storage. Schemes should not know how to build your state or translation. They ask the allocator instead.

A custom allocator usually provides operations such as:

- allocate a blank state;
- allocate a blank translation;
- copy a state;
- create operator or block scratch where needed.

## Derivatives and linearizers

A derivative maps:

```text
(time, state) -> translation
```

A linearizer maps:

```text
(time, state, source translation) -> target translation
```

Explicit schemes need only the derivative. Newton-style implicit schemes need a linearizer.

## When to use this path

Use foreign-model integration when:

- your model state should remain a domain object;
- flattening and unflattening dominate the code;
- solver increments are naturally different from model states;
- you need matrix-free operators or problem-specific preconditioners;
- you are integrating STARK into an existing simulation package.

Do not use this path just to solve a small NumPy vector problem. Use `System` and `Frame` for that.

## Relationship to the mathematical contracts

This page is the practical integration guide. [Mathematical contracts](contracts_math.md) is the formal reference for the same ideas: affine state spaces, translation vector spaces, derivatives, linearizers, blocks, operators, resolvents, and inverters.
