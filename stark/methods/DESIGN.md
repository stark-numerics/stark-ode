# Methods Design Notes

`stark.methods` owns numerical method choices. This is the domain where users
choose how STARK advances, solves, or approximates the problem they declared.

The current method stack is intentionally flat:

```text
schemes      time-stepping formulas
resolvents   nonlinear or implicit stage solvers
inverters    linear inverse actions used by resolvents
```

Inverters could be seen as support for resolvents, and resolvents as support
for schemes. STARK keeps them as sibling method families because flattening the
concepts makes customization clearer: a user can change the inverse action
without treating it as a hidden implementation detail of a Newton resolvent.

## Method Assembly

The high-level `Method` object is the friendly assembly point. It should make
common configurations concise while still allowing advanced users to pass
concrete scheme, resolvent, inverter, predictor, or linear_fixed instances.

The lower-level families should remain independently usable. A contributor
working on a scheme should not need to know every inverter implementation, and
an inverter should not need to know which scheme will eventually call it.

## Common Subpackage Vocabulary

The method families repeat a few implementation subpackage names. They are
intended to mean the same thing wherever they appear:

```text
method          high-level family assembly or user-facing selection helpers
requests        small problem/request objects passed into hot workers
linear_fixed_generation
                scheme stencils and protocols for generated fixed-linear paths
specialization  resolvent prepared algebra or backend-specific fast paths
monitoring      optional observation wrappers, never core algorithm bodies
display         names, labels, and report formatting helpers
execution       scheme execution helpers that are not themselves schemes
```

Keep this vocabulary boring. A contributor should be able to move from schemes
to resolvents or inverters without relearning what each support package is for.

## Hot Path Convention

Method objects are called repeatedly. Constructors and preparation steps should
choose concrete call paths up front:

```text
__call__ -> redirect_call -> selected body
```

Monitoring, safety wrappers, specialization, and optional diagnostics should be
attached by selecting a different callable, not by adding repeated branches to
the numerical body.

## Design Rule

Keep numerical families composable and explicit. If a feature only exists to
make one composition convenient, put it at the assembly layer. If it is a
general numerical idea, keep it in the relevant family.
