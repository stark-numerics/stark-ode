# Scheme Contributor Notes

Schemes are the hot step-level objects in STARK. They are called repeatedly by
integrators, so construction should choose the call path once and the repeated
`__call__` should stay tiny.

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
