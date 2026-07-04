# Scheme Contributor Notes

Schemes are the hot step-level objects in STARK. They are called repeatedly by
integrators, so construction should choose the call path once and the repeated
`__call__` should stay tiny.

## Role

A scheme owns the time-stepping formula. It receives a problem state, an
interval cursor, and prepared problem workers. It decides:

```text
which stages are evaluated
whether a proposed adaptive step is accepted
how the state is updated when a step is accepted
what step size should be tried next
```

Schemes should not own user callback adaptation, backend storage policy, or
diagnostic report formatting.

## Call Path Convention

Scheme classes should use this shape:

```python
def __call__(self, interval, state):
    return self.redirect_call(interval, state)
```

The constructor prepares:

- `call_body`: the algorithmic body, usually `call_inline` or
  `call_specialized`.
- `call_step`: optional non-critical wrapper, such as monitoring.
- `redirect_call`: the selected callable used by `__call__`.

Specialized kernels, monitoring, and other optional features should update the
selected path at construction time or preparation time. They should not add
branching to the per-step hot path.

## Optional Features

Optional instrumentation should be easy to delete. Prefer small decorators or
wrappers that add monitoring/reporting around `call_body` instead of mixing
diagnostic logic into the numerical body.

Schemes intentionally receive monitors at initialization time. They should not
be mutated to attach or remove monitors later.

## Specialists

A specialist is usually prepared by the Algebraist/engine stack and gives a
scheme a faster implementation of repeated stage algebra. It may also be
user-overridden. That override path is deliberate: future packages such as
`stark-pde` are expected to need domain-specific stage updates.

The default path should still be understandable without a specialist. The
specialist is an acceleration or domain-extension seam, not the mathematical
definition of the scheme.

## Shared Family Objects

Some method families genuinely share a step algorithm. Prefer a clearly named
prepared object over a scheme base class when the shared object is implementation
machinery rather than public scheme identity.

For example, a concrete scheme can own a `KennedyCarpenterAdaptiveStep` instead
of inheriting from a hidden base class. The concrete scheme keeps the descriptor,
tableau, display support, monitoring wrapper, and public `__call__`; the shared
step object owns the reusable stage algorithm.

Do not hide these objects behind vague underscore names. Python underscores are
only a convention, and overusing them makes the architecture harder to read.
If contributors need to understand the object, give it a role-revealing name.

## Runtime Objects

The `explicit`, `implicit`, and `imex` scheme packages each have a `runtime.py`
module. These modules contain owned runtime objects such as
`SchemeRuntimeExplicit`, not base classes. They collect setup work that every
scheme in a family needs:

- auditing dynamics and allocator compatibility;
- allocating translation probes and `SchemeStepSupport`;
- preparing family-specific helpers such as implicit block allocators;
- checking family-specific construction rules, such as tableau compatibility.

Schemes should instantiate their runtime object in `__init__`, then copy hot
attributes such as `workspace`, `dynamics`, `block_allocator`, or `k1` onto
the scheme instance when direct access keeps the step body clearer. That copying
is deliberate; it keeps the runtime object as construction machinery without
forcing every stage operation through an extra indirection.

Avoid one-line helper functions for things a scheme can say directly. A method
such as `snapshot_state` is clearer as a small real method on the scheme than as
a module-level function assigned onto the class.

## Generic Scheme Classes

The core contracts are generic over state and translation types, but concrete
scheme classes intentionally keep readable public constructor signatures for
the beta release. The internals preserve concrete types through runtime objects,
requests, resolvents, inverters, and shared fixtures; users should not need to
write `SchemeEuler[StateType, TranslationType]` merely to solve an IVP.

A future contributor may decide to make every concrete scheme class explicitly
generic, for example `class SchemeEuler(Generic[StateType, TranslationType])`.
Only do this if it fixes a noticeable user-side typing hole, such as IDE hover
text or checker diagnostics failing to preserve a user's concrete state and
translation types through scheme composition. If that change is made, add an
in-code comment near the class definition explaining why the generic parameters
exist; otherwise they look like type-theory decoration rather than useful API
support.
