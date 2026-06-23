# Inverter Contributor Notes

Inverters provide inverse actions for linear correction problems. They are used
by resolvents, but they are kept as their own method family because the same
inverse-action ideas can support several resolvent styles.

## Current Families

- `dense`: materialises small operators and solves them locally with
  `InverterNucleus`.
- `krylov`: matrix-free Arnoldi/GMRES-style inverse actions.
- `relaxation`: simple stationary iterations such as Richardson and Jacobi.

## Current Design Rules

- Inverters own inverter-specific configuration through
  `InverterConfiguration`.
- Inverters should follow the same hot-path convention as schemes:
  `__call__` delegates to `redirect_call`, and construction chooses the concrete
  path. Optional monitoring should not sit inside unmonitored solve loops.
- Inverters may receive an accelerator, but they should not require callers to
  know which internal kernel will consume it.
- Hot solve paths should trust already-constructed inputs. Expensive safety
  checks belong at construction time, in private probes, or in tests.
- Preconditioners are explicit Krylov collaborators. A missing preconditioner
  means unpreconditioned Arnoldi, not hidden diagonal magic.
- Avoid external dense-solve provider dispatch in the hot path. Experiments
  showed that dispatching to external providers was slower than the local
  nucleus path for the small systems this package currently materialises.

## Known Gaps

- Krylov support is still early. It works on small systems, but poor restart or
  missing preconditioning can make it fail badly even on diagonal multi-block
  probes.
- Preconditioner coverage is minimal. The current package only provides an
  identity preconditioner and a diagonal inverse preconditioner for operators
  whose entries expose `inverse(source, target)`.
- Dense support is intentionally local-first. Any future external-provider
  attempt needs a benchmark that beats `InverterNucleus` before it enters the
  public design.
- Sparse, factorised, and GPU-specialised inverse actions are future work.

Run a private local probe before promoting a Krylov example or changing Arnoldi
defaults.
