# Engines and backends

An engine chooses the storage and arithmetic backend used for frame-backed states and translations.

Common engines include:

```text
EngineNative
EngineNumpy
EngineJax
EngineCupy
```

The engine is not merely an array type. It also determines allocation, copying, translation storage, and available acceleration hooks.

## Native

The native engine is useful for plain Python values and small examples. It keeps dependencies low and is good for teaching the object model.

## NumPy

The NumPy engine is the usual default for array-valued problems on CPU. It is the best first choice for most structured numerical examples.

## JAX

The JAX engine is useful for JAX-backed arrays and return-style derivatives. JAX arrays are immutable, so return-style derivative and linearizer patterns are often more natural than in-place mutation.

Not every STARK control path is automatically JIT-fused. Solver control flow, monitors, Python-level Krylov loops, and user callbacks may still run at Python level unless a specific accelerated path exists.

## CuPy

The CuPy engine is useful for GPU-backed arrays when the problem is large enough to justify GPU execution and data movement costs.

Examples that use CuPy should be optional: they should report a missing dependency or missing GPU clearly rather than failing the whole example suite.

## Accelerators

Accelerators compile or specialize selected arithmetic kernels. They do not make arbitrary Python control flow disappear.

This distinction matters for benchmarking. A generated translation-combine kernel may be accelerated while a surrounding Newton or Krylov iteration remains Python-level.

## Backend comparison

Backend comparisons should report:

- preparation time;
- warm repeated solve time;
- total time;
- accuracy or final error.

This avoids making JIT or GPU warmup appear free.
