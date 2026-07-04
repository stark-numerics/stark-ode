# Contract maths

This is the formal companion to [Concepts and terminology](concepts.md).
Most users do not need this page on day one. Read it when you are building a
custom state model, writing a method component, or trying to understand why
STARK keeps state, translation, linearizer, resolvent, and inverter as distinct
objects.

The short version:

```text
STARK treats an ODE solver as algebra on model states and solver increments.
```

The high-level `System` and `Frame` path generates the algebra for you. The
contracts below describe what must be true when you provide that algebra
yourself.

## State space

The model state lives in a state space:

```text
x in S
```

`S` might be as simple as a NumPy array or as specific as a simulation object
with nested fields, invariants, and labels. STARK does not require `S` to be a
vector space.

That choice is deliberate. Many useful models have states that should not be
added together directly.

## Translation space

A translation is a solver increment:

```text
v in T
```

For ordinary vector problems, `T` often has the same storage shape as `S`. For
foreign object models, `T` may be a separate delta/increment type.

Translations need vector-space-like operations because schemes form weighted
stage combinations:

```text
a v + b w
||v||
```

In code, that means a translation must support the operations expected by the
scheme family you use: application to a state, scaling, addition, and usually a
norm.

## State application

Translations act on states:

```text
apply(x, v) -> x_new
```

For a vector state this looks like:

```text
x_new = x + v
```

For a foreign object model it may mean copying labels, preserving invariants,
or updating several nested fields. This is why STARK keeps the operation as a
contract rather than assuming raw vector addition.

## Dynamics

ODE dynamics maps time and state to a translation:

```text
f : R x S -> T
```

The solver-facing dynamics contract writes into an existing translation:

```text
dynamics(interval, state, out) -> None
```

That in-place shape lets schemes reuse scratch objects. User-facing dynamics
styles adapt friendlier signatures into this contract.

## Frame-backed models

A `Frame` declares a structured state and translation pair:

```text
state fields       y, position, velocity, u
translation fields dy, dposition, dvelocity, du
```

For a `Frame`-backed model, STARK can generate the state carrier, translation
carrier, allocator, norm policy, and repeated algebra kernels. This is the
preferred public path because it gives the engine enough structure to optimise.

## Foreign models

A foreign model provides its own `S`, `T`, and allocator.

Use this route when the model already has meaningful objects that should not be
flattened only to satisfy a solver. The price is that STARK knows less about
the structure, so it may need runtime algebra rather than generated kernels.

## Blocks

Implicit schemes and multi-stage methods often solve product-space equations.
STARK represents those products with blocks:

```text
Block[T] = T x T x ... x T
```

A block is not a user model. It is method machinery: a way for schemes,
resolvents, and inverters to talk about several stage translations together
without flattening the original state.

## Operators

An operator maps translations to translations:

```text
A : T -> T
```

In code, STARK usually uses operator action:

```text
operator(source, target) -> target
```

This shape supports matrix-free methods. A dense matrix is only needed when the
chosen inverter asks for materialisation.

## Linearizer and Jacobian action

For:

```text
x' = f(t, x)
```

the Jacobian action is:

```text
J(t, x) v = Df(t, x)[v]
```

A `Linearizer` supplies this action for Newton-style resolvents. It may also
provide a dense fill for small dense inverters:

```text
operator action     J v
dense fill          matrix entries for J
```

Krylov inverters need only the action. Dense inverters need materialisation.

## Residuals and resolvents

Implicit schemes produce nonlinear stage equations. A residual measures how far
a proposed stage correction is from solving that equation.

A resolvent improves a stage correction until the residual is acceptable:

```text
resolvent(request, output) -> None
```

The scheme owns the stage equation. The resolvent owns the nonlinear solve.

## Inverters

Newton-style resolvents reduce nonlinear solves to linear correction problems:

```text
A delta = residual
```

An inverter applies an inverse action to that request:

```text
inverter(request, output) -> None
```

An inverter must declare whether it writes `output` from scratch or improves an
existing guess. That distinction matters for Krylov and relaxation methods.

## Preconditioners

A preconditioner is an approximate inverse used by a Krylov inverter:

```text
M^-1 residual ~= A^-1 residual
```

STARK keeps preconditioners explicit. Hidden preconditioning makes behaviour
hard to explain and harder to benchmark.

## Predictors

A scheme predictor seeds the initial guess for an implicit stage:

```text
known stage data + previous stage data -> initial delta
```

Predictors belong to schemes. They should not know about the linearizer,
resolvent, or inverter.

## Engines and Algebraist

An engine chooses concrete storage and arithmetic for the abstract objects
above:

```text
state carrier
translation carrier
allocator
norm and inner-product policy
generated or runtime Algebraist path
optional accelerator
```

For known `Frame` structures, the generated Algebraist path is preferred. For
unknown foreign model structures, runtime algebra is the flexible fallback.

## Summary table

| Maths role | STARK role |
|---|---|
| state space `S` | state object / `Frame` state fields |
| translation space `T` | translation object / `Frame` translation fields |
| action `S x T -> S` | translation application |
| right-hand side `f(t, x)` | dynamics |
| Jacobian action `Df(t, x)[v]` | linearizer / operator |
| product space `T^n` | block |
| nonlinear stage solve | resolvent |
| linear correction solve | inverter |
| approximate inverse | preconditioner |
| initial stage guess | scheme predictor |
| concrete arithmetic | engine / carrier / Algebraist / accelerator |

## Design consequence

The contracts may look more elaborate than a flat-vector solver, but they buy
STARK its main design goal: a project can begin with a simple `Frame`, grow
into implicit methods and backend acceleration, and still preserve a meaningful
model shape when the state becomes complicated.
