# Native Engine Design Notes

`EngineNative` stores one-dimensional frame fields in Python `array.array`
objects.

This backend exists as a small, dependency-light CPU path and as a useful
comparison point for generated algebra. It is not the universal fallback for
all possible state shapes.

## Current Shape

Native currently supports one-dimensional field shapes only. That limitation is
deliberate in the current implementation:

```text
Frame.vector(...)      natural fit
Frame.array(..., 2D)   not currently supported by EngineNative
```

If multi-dimensional native storage is added later, document whether it uses
nested carriers, flattened arrays with shape metadata, or a new carrier family.

## Accelerator Policy

Like NumPy, Native attempts to use `AcceleratorNumba` by default and falls back
to `AcceleratorNone` when Numba is unavailable. The fallback should remain
usable and visibly slower rather than failing import-time.

## Review Note

Native currently differs from NumPy, CuPy, and JAX in how generated linear
combination support is installed. The engine constructs an
`AlgebraistGeneratorLinearCombine`, but the allocator does not currently expose
the arity-indexed `linear_combine` table that the other frame-backed engines
install.

This may be intentional if Native translations use a different fallback path,
but it should be checked during the engine correctness pass. If Native is meant
to be a first-class generated-Algebraist backend, its allocator-installed
prepared algebra should mirror the other engines unless there is a documented
reason not to.

## Design Rule

Keep Native simple. It should be easy to understand what object owns the Python
storage and what generated kernel is operating on it.
