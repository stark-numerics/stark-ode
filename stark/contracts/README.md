# STARK Contracts

`stark.contracts` is the public documentation layer for the shapes STARK
accepts from user code. Runtime modules may use smaller local protocols when
that reduces coupling, but these files should explain the stable concepts a
user needs in order to supply custom objects.

Prefer focused imports when writing docs or implementation code:

- `stark.contracts.accelerator`: accelerator backends
- `stark.contracts.block`: grouped translation containers
- `stark.contracts.carrier`: carrier bundles
- `stark.contracts.derivative`: right-hand-side workers
- `stark.contracts.derivative_imex`: IMEX derivative split carrier
- `stark.contracts.inner_product`: translation-space inner products
- `stark.contracts.integrator`: trajectory-building workers
- `stark.contracts.interval`: timeline cursors
- `stark.contracts.linear_combine`: translation algebra fast paths
- `stark.contracts.linearizer`: Jacobian-action workers
- `stark.contracts.stepper`: step-accepting workers
- `stark.contracts.operator`: matrix-free linear operators
- `stark.contracts.residual`: residual workers for nonlinear solves
- `stark.contracts.resolvent`: nonlinear stage solvers
- `stark.contracts.inverter`: linear solvers and preconditioners
- `stark.contracts.state`: unconstrained mutable state objects
- `stark.contracts.translation`: linear update objects
- `stark.contracts.allocator`: state and translation factories

Use the package-level `stark.contracts` imports for convenience in examples and
interactive exploration.
