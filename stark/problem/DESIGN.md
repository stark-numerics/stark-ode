# Problem Design Notes

`stark.problem` is the user-facing language for writing down an IVP before
choosing a numerical method or backend.

The package exists so the mathematical model stays visible:

```text
state shape   -> Frame
right side    -> Derivative
Jacobian data -> Linearizer
assembly      -> System
```

## Boundary

Problem objects describe what is being solved. They should not decide how a
step is taken, how arrays are stored, or how a run is monitored.

Those responsibilities live elsewhere:

```text
methods      choose numerical algorithms
engines      choose storage and algebra
core         owns minimal solver contracts and trajectory machinery
diagnostics  observes and reports
```

This separation is deliberate. A user should be able to change method or engine
without rewriting the problem declaration.

## Friendly Surface, Strict Internal Shape

The public problem API should accept natural mathematical callables and state
descriptions. Internally, those inputs are adapted into stable contracts:

```text
DerivativeLike
DerivativeSplitLike
LinearizerLike
Frame fields
System IVP assembly
```

Do not push callback-signature inspection into schemes or resolvents. Adapt
once in `problem`, then give methods a prepared worker.

## Design Rule

If a concept is something a user would write on paper before selecting a solver,
it probably belongs in `problem`. If it is solver machinery, it probably does
not.

`stark.problem.__init__` should not re-export `Method` or `MethodError`. The
problem domain describes the model; method selection belongs in
`stark.methods` or the package-level convenience surface.
