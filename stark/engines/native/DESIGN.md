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

## Hint Types

Native carrier storage is ordinary Python data, but static typing still needs a
boundary description. Use `HintNativeNumber` for the numeric operations native
carriers require, such as addition, multiplication, absolute value, and
conversion to `float`.

Do not annotate native carrier values with `numbers.Number` merely because the
runtime check uses it. Pyright treats `Number` as an abstract marker and cannot
infer the arithmetic operations that these carriers intentionally rely on.

## Generated Linear Combine

Native mirrors the other frame-backed engines by installing the generated
arity-indexed `linear_combine` table on its allocator.

Translations expose that allocator-provided table through their
`linear_combine` property. This keeps scheme/runtime support on prepared
Algebraist kernels instead of falling back to generic translation arithmetic
when the `Frame` is known.

## Design Rule

Keep Native simple. It should be easy to understand what object owns the Python
storage and what generated kernel is operating on it.
