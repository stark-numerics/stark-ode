# Core Design Notes

`stark.core` is the minimal solver toolkit. A sufficiently advanced user should
be able to build an ODE solver from core objects plus their own problem,
method, engine, and diagnostic pieces.

That is why other domains are allowed to import from core. Core is the stable
substrate of the package, not a miscellaneous utility drawer.

## What Belongs Here

Core owns concepts that remain meaningful even when every higher-level
convenience layer is replaced:

- intervals and tolerances,
- stable protocols and audits,
- block/product-space machinery,
- integrator and stepper primitives,
- shared configuration shapes,
- low-level algebra contracts.

Problem declarations, concrete numerical methods, backend engines, and monitors
should not move into core merely because they are widely used.

## Dependency Direction

Other domains may depend on core:

```text
problem      -> core
methods      -> core
engines      -> core
diagnostics  -> core
```

Core should avoid depending on concrete workers from those domains. If core
needs to talk to another domain, prefer a local protocol or a contract in
`stark.core.contracts`.

## Design Rule

Core should answer:

```text
What is the smallest stable shape an expert could use directly?
```

If a class only exists to make the public `System` path nicer, it probably
belongs in `problem`. If it only exists for one scheme, resolvent, inverter, or
engine, it belongs with that family.
