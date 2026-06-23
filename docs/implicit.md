# Solve stiff and implicit problems

This page is for users whose explicit solves are too slow, unstable, or require tiny steps.

Implicit solving adds three pieces:

```text
linearizer   Jacobian action of the derivative
resolvent    nonlinear stage solver
inverter     linear correction solver
```

## First implicit route: small dense Newton

Use this route for small stiff systems such as Robertson, HIRES, or a low-dimensional Van der Pol oscillator.

```text
SchemeKvaerno3 or SchemeKvaerno5
ResolventNewton
InverterDense
LinearizerStyle.operator or LinearizerStyle.dense
```

Run:

```powershell
python -m examples.problem.linearizer_styles
python -m examples.inverters.inverter_dense
```

## What the linearizer does

For an ODE:

```text
y' = f(t, y)
```

Newton needs the derivative of `f` with respect to `y`:

```text
J(t, y) v = Df(t, y)[v]
```

In STARK, that is a linearizer. It can support:

```text
operator apply   matrix-free action Jv
dense fill       materialise J for dense inverters
```

Dense inverters can use a dense fill. Krylov inverters only need operator action.

## Dense vs Krylov vs relaxation

| Route | Good for | Bad for |
|---|---|---|
| Dense | small systems, cheap dense materialisation | large PDE-like systems |
| Krylov | large matrix-free systems | small systems where Python iteration overhead dominates |
| Relaxation | teaching, structured simple iteration | high-performance stiff solves without special structure |

Use dense for small stiff ODEs. Use Krylov when dense matrices are too expensive and you have a good operator action. Add a preconditioner when Krylov iterations are too slow.

## Preconditioners

A preconditioner approximately solves the linear correction system or an easier related system. Krylov methods can use it to reduce iteration count.

For a large periodic tridiagonal problem, a preconditioner might solve an approximate periodic tridiagonal system cheaply.

Run the matrix-free example:

```powershell
python -m examples.inverters.inverter_krylov
```

Run the matrix-free Jacobian example when you want Krylov in a real Newton solve:

```powershell
python -m examples.methods.matrix_free_jacobian
```

## Chord and VeryChord

Newton refreshes the linearization often. Chord and VeryChord reuse it more aggressively. They can be faster when the linearization remains useful across corrections or stages.

Use competition reports to compare these choices:

```powershell
python -m competition.robertson.report
python -m competition.hires.report
```

Interpret both warm and total timing tables. JIT or compilation work can move into preparation.

## Common mistakes

### Using an implicit scheme without a useful linearizer

Newton can only be effective if the linearizer represents the derivative's Jacobian accurately enough.

### Using Krylov for tiny systems

Krylov has iteration and Python control overhead. Use dense for systems with only a few dimensions.

### Treating monitor timings as solver timings

Monitor-enabled implicit solves are for diagnosis. Use unmonitored runs for speed comparisons.
