# STARK contracts in mathematical language

This page explains how STARK's code-level contracts correspond to the low-level
mathematical structures behind explicit and implicit ODE solvers.

It is written for readers who think naturally in terms of spaces, operators,
and convergence, but it also links out to brief refreshers for readers coming
from physics or engineering code.

Useful background:

- [Affine space](https://en.wikipedia.org/wiki/Affine_space)
- [Vector space](https://en.wikipedia.org/wiki/Vector_space)
- [Normed vector space](https://en.wikipedia.org/wiki/Normed_vector_space)
- [Banach space](https://en.wikipedia.org/wiki/Banach_space)
- [Linear operator](https://en.wikipedia.org/wiki/Linear_map)
- [Inner product space](https://en.wikipedia.org/wiki/Inner_product_space)
- [Runge-Kutta methods](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)
- [Backward differentiation formula](https://en.wikipedia.org/wiki/Backward_differentiation_formula)
- [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method)
- [Fixed-point iteration](https://en.wikipedia.org/wiki/Fixed-point_iteration)
- [Anderson acceleration](https://en.wikipedia.org/wiki/Anderson_acceleration)
- [Broyden's method](https://en.wikipedia.org/wiki/Broyden%27s_method)
- [Krylov subspace](https://en.wikipedia.org/wiki/Krylov_subspace)

## States and translations

The user begins with a space `S` of states. In code this is the STARK `State`
concept: any mutable Python object representing the evolving configuration of
the system.

STARK then asks the user to define a second space `T` of translations. A
translation `tau in T` is not itself a state. It is a linear update that can be
applied to a state:

```text
tau : S -> S
```

In STARK that is the `Translation.__call__(origin, result)` contract.

So the basic state update picture is:

```text
x in S
tau in T
tau(x) in S
```

The pair `(S, T)` is meant to behave like an
[affine space](https://en.wikipedia.org/wiki/Affine_space): states live in the
nonlinear space `S`, while differences and updates live in the linear space
`T`.

That affine-space viewpoint matters because it is the weakest natural setting
in which it makes sense to differentiate a state trajectory with respect to an
affine parameter such as time. STARK never asks you to pretend your state
object *is* a flat vector if it is not naturally one. The separate question of
linearizing with respect to state only appears later, when implicit methods ask
for Jacobian actions through a `Linearizer`.

## The extra structure needed for explicit methods

Even the simplest explicit [Runge-Kutta method](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)
needs more than the ability to apply an update. It needs to build weighted
combinations of updates such as

```text
a_1 k_1 + a_2 k_2 + ... + a_n k_n
```

for stage derivatives `k_i in T`.

So STARK equips `T` with the structure of a
[vector space](https://en.wikipedia.org/wiki/Vector_space):

- addition `tau_1 + tau_2`
- scalar multiplication `a tau`

That is why the `Translation` contract asks for:

- `__add__(other)`
- `__rmul__(scalar)`

In code, those operations may be provided generically, or replaced with faster
fused kernels through `linear_combine`.

## Error control and norms

Adaptive schemes need a way to measure the size of a trial update or an
embedded error estimate. So STARK also asks for a norm on `T`:

```text
||tau||
```

This makes `T` a [normed vector space](https://en.wikipedia.org/wiki/Normed_vector_space).
In code, this is `Translation.norm()`.

That norm drives:

- adaptive step rejection/acceptance in `Marcher`
- stopping tests for nonlinear resolvents
- stopping tests for linear inverters

There is nothing in the API forcing `T` to be a
[Banach space](https://en.wikipedia.org/wiki/Banach_space), but users should
think seriously about whether their chosen translation representation is
complete enough for the convergence arguments they care about. STARK can run on
an incomplete normed space; mathematics may be less forgiving than software.

## The derivative contract

Once `S` and `T` are in place, the ODE

```text
x'(t) = f(t, x(t))
```

is encoded by a derivative worker

```text
f : R x S -> T
```

which writes its result into a translation buffer together with the current
integration interval:

```python
derivative(interval, state, out_translation)
```

This is the first point where STARK turns the mathematical separation between
states and increments directly into code.

For IMEX methods the right-hand side is split into two pieces,

```text
f(t, x) = f_im(t, x) + f_ex(t, x),
```

with both maps landing in `T`. STARK represents that split with an
`ImExDerivative` carrying two ordinary derivative workers:

```python
ImExDerivative(
    implicit=implicit_derivative,
    explicit=explicit_derivative,
)
```

This keeps the user-facing interface close to the mathematics: one map is the
implicitly treated part, one map is the explicitly treated part.

## The workbench contract

The `Workbench` is not a mathematical object in the same sense. It is the
bridge that tells STARK how to allocate and copy elements of `S` and `T`.

It provides:

- `allocate_state()`
- `copy_state(dst, src)`
- `allocate_translation()`

So if `S` and `T` describe the mathematics, the workbench describes the memory
layout and scratch discipline needed to compute with them efficiently.

## Implicit methods: residuals and blocks

Implicit schemes add another layer. Instead of directly computing a step from
known stage derivatives, they solve a nonlinear equation for unknown stage
translations.

For a one-stage implicit method this can often be written as:

```text
Find z in T such that R(z) = 0
```

For multi-stage methods the unknown is naturally a tuple of translations:

```text
Find Z = (z_1, ..., z_m) in T^m such that R(Z) = 0
```

STARK calls that product-space object a `Block`. So `Block` is code for an
element of `T^m`, not a new mathematical mystery.

The `Residual` contract represents

```text
R : T^m -> T^m
```

by writing into a supplied `Block`.

## Linear operators and inverters

Once linearization enters the picture, STARK needs a notion of linear operator
on `T`, or on `T^m` for coupled stage systems.

The `Operator` contract represents a map

```text
A : T -> T
```

and the internal block operator used by Newton/Krylov workers lifts that idea to

```text
A : T^m -> T^m
```

The built-in Newton resolvent does not form an explicit inverse. Instead it
binds such an operator to an inverter and asks the inverter to approximately
solve

```text
A delta = b
```

This is the role of `InverterLike`.

## Linearization and the Linearizer contract

Newton-style implicit methods need the derivative of the nonlinear residual.
That in turn depends on the derivative of the user-supplied ODE right-hand
side with respect to its state argument.

If the ODE is

```text
x' = f(t, x)
```

then the relevant object is the derivative of `f` at a state `x`. This is the
[Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant), but
in STARK it should be understood as a linear operator rather than as a dense
matrix.

For a small translation `delta in T`, the linear approximation is

```text
f(t, x + delta) = f(t, x) + J(t, x)[delta] + higher-order terms
```

where

```text
J(t, x) = D_x f(t, x)
```

and

```text
J(t, x) : T -> T
```

is a linear operator acting on translations.

So the user supplies a `Linearizer` that gives the action

```text
delta |-> J(t, x)[delta]
```

without ever needing to materialize a full dense matrix.

In code:

```python
linearizer(interval, out_operator, state)
```

where `out_operator(result, translation)` computes the Jacobian image. This is
the matrix-free form used throughout modern iterative methods and sparse
numerics.

The pencil-and-paper task for the user is therefore:

1. derive the derivative of `f` with respect to its state argument
2. express the Jacobian action `J(t, x)[delta]` on the chosen translation representation
3. write that action as an `Operator`

That is enough for STARK to build the linearized operators needed by implicit
schemes, such as `I - h J(t, x)` for backward Euler or the stage operators in
ESDIRK methods.

## Inner products and Krylov methods

Some linear algebra needs more than a norm. In particular,
[Krylov subspace](https://en.wikipedia.org/wiki/Krylov_subspace) methods such
as GMRES, FGMRES, and BiCGStab rely on projections, orthogonalization, or
bi-orthogonalization.

So STARK introduces an `InnerProduct`:

```text
<tau_1, tau_2>
```

making `T` an [inner product space](https://en.wikipedia.org/wiki/Inner_product_space)
when that extra structure is available.

This is not required for explicit schemes. It becomes relevant when the user
wants Newton-like resolvents with matrix-free Krylov inverters, or secant-based
resolvents such as Anderson and Broyden that project histories in `T`.

## Resolver and resolvent layers

At the nonlinear level STARK supports several solver ideas:

- `ResolventPicard`: fixed-point iteration on the implicit residual map
- `ResolventAnderson`: accelerated fixed-point iteration
- `ResolventBroyden`: quasi-Newton secant updates
- `ResolventNewton`: true linearized Newton corrections

These are all strategies for solving

```text
R(Z) = 0
```

on a block `Z in T^m`. They differ only in what additional structure they
require:

- Picard needs residual evaluation
- Anderson and Broyden also benefit from an inner product and secant history
- Newton needs residual linearization and an inverter

Schemes do not talk to those nonlinear workers directly. They talk to
a `Resolvent`, which binds the stage interval and stage state, then solves a
shifted equation of the form

```text
delta - rhs - alpha f(t, x + delta) = 0.
```

That makes the scheme-facing implicit API closer to the mathematics of a stage
solve and less entangled with the lower-level nonlinear machinery used to carry
it out.

## How the code maps to the mathematics

The key STARK contracts can be summarized this way:

| Mathematical object | STARK concept |
| --- | --- |
| state space `S` | user `State` |
| translation space `T` | `Translation` |
| affine action `T x S -> S` | `Translation.__call__` |
| vector-space operations on `T` | `__add__`, `__rmul__`, `linear_combine` |
| norm `||.||` on `T` | `Translation.norm()` |
| right-hand side `f : R x S -> T` | `Derivative` |
| Jacobian action `D_x f(t, x)` | `Linearizer` + `Operator` |
| product space `T^m` | `Block` |
| residual `R : T^m -> T^m` | `Residual` |
| nonlinear implicit solver | `Resolvent` |
| bound stage solve | `Resolvent` |
| linear solve worker | `InverterLike` |
| inner product `<.,.>` | `InnerProduct` |

This is the core idea of the package: keep the mathematics honest, keep the
state representation natural, and only ask for extra structure when a method
really needs it.






