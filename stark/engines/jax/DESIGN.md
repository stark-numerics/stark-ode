# JAX Engine Design Notes

`EngineJax` stores shaped frame fields in JAX arrays and uses functional
generated algebra.

JAX has a different preferred shape from NumPy and CuPy: it wants pure
expression-style kernels that return arrays rather than into-style mutation.

## Target Policy

JAX uses the functional Algebraist target. Generated kernels should look like:

```text
inputs -> returned field values
```

not:

```text
inputs + mutable output buffer -> mutated output buffer
```

The allocator may still present STARK-style state and translation objects, but
the generated algebra should respect JAX's immutable-array model.

## Dtype Policy

JAX defaults depend on whether x64 is enabled. `EngineJax` resolves the active
dtype at construction and raises clear errors for requested 64-bit dtypes when
JAX x64 support is disabled.

Complex dtype support should remain explicit. Do not quietly demote complex
precision.

## JIT Boundary

JAX array support is not the same thing as whole-solver JIT. STARK's adaptive
control flow and Python object orchestration are still Python-level unless a
future design explicitly moves that boundary.

## Hint Types

Names with the `Hint` prefix are intentionally type-hint scaffolding. Their
sole job is to describe the small external-library surface STARK uses without
turning optional backend objects into public abstractions.

For example, `HintJaxArray` should be beefed up when carrier code uses more of
the JAX array protocol. It should not be renamed to a domain noun such as
`JaxArray`, because that would make it look like a STARK concept rather than a
local typing aid.

## Design Rule

Optimise JAX by improving generated functional algebra first. Do not contort
the public problem API to make the entire solver look like one JAX function.
