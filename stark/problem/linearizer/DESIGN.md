# Linearizer Design Notes

Linearizers are the user-side language for Jacobian action.

They live in `problem` because a linearizer is part of what a user writes down
when describing an implicit problem. It is not merely Newton plumbing.

## Mathematical Role

For:

```text
y' = f(t, y)
```

a Newton-style implicit method needs the Jacobian action:

```text
J(t, y) v = Df(t, y)[v]
```

The linearizer supplies that action.

## TranslationOperator Action and Dense Fill

A linearizer may support:

- matrix-free operator action, useful for Krylov inverters,
- dense fill, useful for small dense inverters.

Dense fill is optional. A large problem should not be forced into dense
materialisation just because it provides a Jacobian action.

## Relationship to Dynamics

Linearizer mirrors dynamics in style and structure. Both packages adapt
user-written mathematical callables into the worker shapes consumed by methods.

Decorator names should stay explicit about accepted inputs and whether the
callable writes into an output object.

## Design Rule

Keep linearizer declarations near dynamics and systems. Methods should ask
for linearizer capability; they should not own the user's Jacobian language.
