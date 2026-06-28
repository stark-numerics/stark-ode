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
