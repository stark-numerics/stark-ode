# STARK Contracts

`stark.contracts` is the public documentation layer for the shapes STARK
accepts from user code. Runtime modules may use smaller local protocols when
that reduces coupling, but these files should explain the stable concepts a
user needs in order to supply custom objects.

Prefer focused imports when writing docs or implementation code:

- `stark.contracts.acceleration`: accelerator backends
- `stark.contracts.blocks`: grouped translation containers
- `stark.contracts.carriers`: carrier bundles
- `stark.contracts.derivatives`: right-hand-side workers
- `stark.contracts.derivative_imex`: IMEX derivative split carrier
- `stark.contracts.inner_products`: translation-space inner products
- `stark.contracts.integrators`: trajectory-building workers
- `stark.contracts.intervals`: timeline cursors
- `stark.contracts.linear_combine`: translation algebra fast paths
- `stark.contracts.linearizers`: Jacobian-action workers
- `stark.contracts.marchers`: step-accepting workers
- `stark.contracts.operators`: matrix-free linear operators
- `stark.contracts.residuals`: residual workers for nonlinear solves
- `stark.contracts.resolvents`: nonlinear stage solvers
- `stark.contracts.inverters`: linear solvers and preconditioners
- `stark.contracts.states`: unconstrained mutable state objects
- `stark.contracts.translations`: linear update objects
- `stark.contracts.allocators`: state and translation factories

Use the package-level `stark.contracts` imports for convenience in examples and
interactive exploration.
