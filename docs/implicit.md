# Implicit methods

This page is for users solving stiff or nonlinear problems with implicit schemes.

Implicit methods introduce two extra concepts beyond an explicit scheme:

- a `Linearizer`, which supplies Jacobian actions;
- a `Resolvent`, which solves nonlinear stage equations, often using an `Inverter` for linear corrections.

## Stage equations

An implicit scheme stage usually requires solving an equation of the form:

```text
F(delta) = 0
```

where `delta` is a translation-space stage unknown. The scheme owns the stage structure and asks a resolvent to solve this equation.

## Linearizers

For Newton-style methods, the derivative is not enough. The solver also needs the Jacobian action:

```text
J(t, x) v = Df(t, x)[v]
```

`LinearizerStyle` adapters make it possible to provide this action in a style that matches the problem and backend.

Dense inverters may also use a dense fill of the same operator. Krylov inverters can often use only the matrix-free action.

See:

```powershell
python -m examples.features.linearizer_styles
```

## Resolvents

A resolvent is the nonlinear solver used inside an implicit stage. Typical choices include:

- Picard for simple fixed-point iterations;
- Newton for linearized corrections;
- chord and very-chord variants when the Jacobian is reused.

Newton-type resolvents ask an inverter to solve correction equations. The resolvent does not own the matrix algorithm; that is the inverter's job.

## Inverters

Dense inverters are useful when the correction space is small. They materialise the linear operator and solve the dense system.

Krylov inverters are useful when the correction space is large and matrix-free operator application is cheaper than dense materialisation.

Relaxation inverters are simple iterative methods. They are readable and useful for structured examples, but they are not always the fastest choice.

## Preconditioners

Krylov methods often need a preconditioner. A preconditioner approximately solves a related, cheaper problem so the Krylov iteration converges in fewer steps.

In STARK, preconditioning belongs at the inverter boundary. The Krylov inverter remains generic; problem-specific structure can live in an operator or preconditioner.

## Practical guidance

Use dense inverters for:

- small stiff systems;
- problems where dense Jacobian fill is cheap;
- baseline implicit examples.

Use Krylov inverters for:

- large translation spaces;
- matrix-free linearizers;
- structured PDE-like problems;
- cases where a preconditioner is available.

Use comparison reports to distinguish preparation, warm-run, and total timing. JIT and sparse factorization costs can move work between these columns.
