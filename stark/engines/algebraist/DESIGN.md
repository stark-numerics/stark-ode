# Algebraist Design Notes

Last reviewed: 2026-07-10.

This local design note describes intended internal wiring. Re-check the source before making engine or Algebraist changes.

## Purpose

The Algebraist layer exists so `Frame`-backed state can use prepared algebra kernels instead of generic runtime translation operations.

The short version:

```text
Frame-backed state  -> generated Algebraist kernels -> preferred accelerated path
foreign state       -> runtime Algebraist fallback   -> flexible low-level path
```

Do not treat `AlgebraistRuntime` as the default high-level path for ordinary `Frame` models.

## Main Path

For a high-level solve, the intended path is:

```text
Frame
  -> FrameLike / FieldLike layout
  -> carrier choices from Engine
  -> allocator for state/translation objects
  -> prepared Algebraist bundle
  -> optional accelerator compilation/fusion
  -> prepared kernels supplied through allocator-created translations
  -> schemes/resolvents/inverters call prepared algebra
```

The engine is responsible for choosing carriers, allocator, backend array
behaviour, accelerator, and generated Algebraist target. It should not manually
orchestrate every generated provider. `Algebraist.generator(...)` owns the
linear-combine, specialist, norm, and inner-product providers for a known
frame-backed layout and prepares the allocator-facing kernels during
construction.

## Generated path

Use the generated path when STARK knows the frame structure.

Prepared kernels usually include:

```text
linear combinations
state application
norms
inner products
specialist stage updates
```

This is where backend acceleration should happen for ordinary high-level use.

Concrete engines should expose the prepared bundle as `engine.algebraist`.
Downstream code that needs specialist stage kernels should use
`engine.algebraist.specialist`; code that needs other prepared algebra should
go through the same bundle.

## To Do

- Investigate loop fusion for frames with multiple array fields of the same
  shape. Coupled systems often carry two or more arrays with identical extents;
  the looped Algebraist path may be able to perform matching field operations
  in one loop rather than walking each field separately. This should be decided
  during Algebraist preparation, where the frame layout is known, by detecting
  compatible shapes and dispatching to a generated Algebraist form with a
  smarter loop arrangement.

## Frame, Generator, and Target

The Algebraist stack has three separable concerns:

```text
frame policy   describes the known Frame-backed state layout
Algebraist     owns prepared algebra providers for that layout
generator      emits prepared kernels for each algebra role
target         chooses the backend expression style
```

Keep these separate. A backend may need a different target style without
inventing a new problem API or bypassing the engine. This is especially
important for JAX, CuPy, and future Torch support, where "array operations work"
does not imply "the best generated expression shape is identical".

## Runtime Path

Use runtime Algebraist support for user-defined or foreign state/translation
families where STARK cannot know enough structure to generate code safely.

Examples:

```text
foreign model state
custom translation class
custom allocator with non-frame layout
```

Runtime support is valuable, but it is not the preferred acceleration route for known `Frame` states.
Concrete engines should not default to runtime algebra. If a concrete engine
cannot prepare generated algebra for its own frame-backed layout, that is a
backend/frame contract error rather than a reason to silently fall back.

## NumPy

The NumPy engine should prefer generated Algebraist kernels. When Numba is available and configured, generated kernels may compile on first use.

This means a row labelled "NumPy" may be misleading in performance reports if Numba is active. Backend case studies should distinguish plain NumPy from NumPy + Numba when that distinction matters.

## JAX

JAX wants pure expression-style kernels with immutable arrays.

Into-style generated code such as:

```python
target[...] = a0 * x0 + a1 * x1
```

is not the ideal JAX shape. JAX prefers generated flat expressions such as:

```python
def kernel(a0, x0, a1, x1):
    return a0 * x0 + a1 * x1
```

The wrapper can then place returned fields into the STARK translation object.

Important distinction:

```text
JAX arrays working correctly != whole-solver JIT acceleration
```

A good JAX path should first compile/fuse generated algebra kernels. Whole-solver JIT is a separate, harder target because adaptive control flow and Python object orchestration are involved.

## CuPy

CuPy arrays are mutable and GPU-backed, so into-style broadcast expressions are closer to the desired shape than they are for JAX.

However, a chain of small CuPy operations can still launch many GPU kernels. A stronger CuPy path may need fused kernels, for example through CuPy fusion or elementwise kernel generation.

The first integration goal is to ensure `Frame`-backed CuPy engines use generated Algebraist kernels where structure is known. Later optimisation can reduce kernel-launch count.

## What not to do

Do not add engine-local operation classes merely to patch one benchmark symptom. The existing design already has layers for this:

```text
carrier arithmetic
Algebraist generator/runtime providers
accelerators
allocator-facing prepared kernels
```

If a new object is needed, it should belong clearly to one of those layers.

## Questions to Ask Before Changing Backend Code

```text
Is this a Frame-backed path or a foreign-model path?
Should this use `Algebraist.generator(...)` or explicit runtime support?
Where is the prepared kernel installed?
Which object owns the backend-specific operation?
Does this path accidentally bypass the accelerator?
Does this change preserve the Engine shape?
```

If the answer is unclear, write the discovery down before patching code.
