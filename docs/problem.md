# Define an ODE problem

This page is for users who want to define the equation STARK should solve.

For the high-level path, define:

```text
Frame       the state fields and translation fields
Derivative  the right-hand side f(t, y)
System      the object tying the problem pieces together
Interval    start time, step guess, and stop time
Tolerance   adaptive error scale, when needed
```

## Use `Frame` to describe state and translations

A `Frame` says which fields live on the state and which fields carry solver increments.

```python
frame = Frame({
    "position": {"translation": "dposition", "shape": (3,)},
    "velocity": {"translation": "dvelocity", "shape": (3,)},
})
```

A derivative for that frame writes:

```python
def rhs(t, state, out) -> None:
    del t
    out.dposition[:] = state.velocity
    out.dvelocity[:] = acceleration(state.position)
```

Use this when your model can be represented as named scalar or array fields.

## Use `DerivativeStyle` to adapt the derivative

### In-place derivative

Use in-place style when mutation is natural.

```python
system = System(
    frame=frame,
    derivative=DerivativeStyle.in_place(rhs),
)
```

This is a good fit for NumPy and native mutable arrays.

### Return-style derivative

Use return style when the backend prefers immutable arrays or expression-oriented code.

```python
@DerivativeStyle.returning
def rhs(t, state):
    return {"dy": -0.5 * state.y}
```

This is the recommended shape for JAX examples.

### Kernel styles

Kernel styles bind named fields and parameters once, so the prepared derivative can run with less field discovery.

```python
@DerivativeStyle.kernel(state=("y",), translation=("dy",), parameters=(0.5,))
def decay_kernel(y, dy, rate: float) -> None:
    dy[:] = -rate * y
```

Use kernel styles when you want a compact derivative focused on array fields.

## Build a `System`

```python
system = System(
    derivative=DerivativeStyle.in_place(rhs),
    frame=frame,
)
```

For implicit methods, add a linearizer:

```python
system = System(
    derivative=derivative,
    linearizer=linearizer,
    frame=frame,
)
```

## Add tolerances

Adaptive schemes use tolerances to decide whether a step is accurate enough.

```python
from stark import Configuration, Tolerance

configuration = Configuration(
    scheme_tolerance=Tolerance(atol=1.0e-8, rtol=1.0e-6),
    check_progress=False,
)
```

Use absolute tolerance for small values and relative tolerance for scaled values.

## Define a linearizer for implicit solves

A linearizer describes the Jacobian action of the derivative. Newton-style resolvents use it to solve implicit stage equations.

For a two-component Van der Pol oscillator, the derivative is:

```text
y0' = y1
y1' = mu * (1 - y0**2) * y1 - y0
```

The Jacobian action is:

```python
def jacobian_apply(y, source_dy, out_dy, mu: float) -> None:
    y0 = y[0]
    y1 = y[1]
    v0 = source_dy[0]
    v1 = source_dy[1]

    out_dy[0] = v1
    out_dy[1] = (-2.0 * mu * y0 * y1 - 1.0) * v0 + mu * (1.0 - y0 * y0) * v1
```

Use `LinearizerStyle.operator` when you can provide Jacobian action and, optionally, a dense fill for small dense inverters.

Run the example:

```powershell
python -m examples.features.linearizer_styles
```

## When the high-level problem path is not enough

Use [foreign models](foreign-models.md) when your model already has its own state objects and solver increments. In that path you provide custom state, translation, and allocator objects instead of using `Frame` to generate them.
