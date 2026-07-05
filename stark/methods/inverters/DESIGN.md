# Inverter Contributor Notes

Inverters provide inverse actions for linear correction problems. They are used
by resolvents, but they are kept as their own method family because the same
inverse-action ideas can support several resolvent styles.

## To Do

- Review inverters for hot-path clutter and public shape before a stable
  release.
- Reconcile Krylov implementation details with the intended request-based
  design, especially restart and preconditioner behaviour.
- Confirm that `InverterDense.instance(operator)` remains the intended public
  spelling for operator-bound dense reuse.

## Current Families

- `dense`: materialises small operators and solves them locally with
  `InverterNucleus`.
- `krylov`: matrix-free Arnoldi/GMRES-style inverse actions.
- `relaxation`: simple stationary iterations such as Richardson and Jacobi.

## Current Design Rules

- Inverters own inverter-specific configuration through
  `InverterConfiguration`.
- The public inverter call shape is request-based:

  ```python
  inverter(request, output)
  ```

  where `request.operator(output, image)` applies the linear block operator and
  `request.residual` is the right-hand side block. This is the shape described
  by `stark.core.contracts.Inverter`. Older bind-then-solve or
  residual-then-output test doubles are stale and should not be copied.
- `Inverter.output_mode` is part of the contract. It tells the caller whether
  the supplied `output` block is overwritten with a fresh solution or treated as
  an initial guess to improve in place. This is not monitoring metadata: it is
  solve semantics, and resolvents need it to prepare the right starting block.
- `InverterRequest` is deliberately a handshake object rather than a positional
  argument list. Keeping `operator` and `residual` together leaves room for
  future request metadata without coupling every resolvent to every inverter
  constructor.
- `InverterInstancing.instance(operator)` is an optional optimisation, not the
  base inverter API. Use it when an inverter can prepare operator-specific data
  once and then solve several right-hand sides through the returned
  `InverterInstance(residual, output)`. Dense materialisation is the obvious
  example; generic inverters can ignore this capability.
- The request-level `operator` is a block operator. Dense and Jacobi-style
  implementations may require the more specific diagonal/entry-inspectable
  shape internally, but that is an implementation requirement and should be
  checked or documented at that implementation boundary.
- Inverters should follow the same hot-path convention as schemes:
  `__call__` delegates to `redirect_call`, and construction chooses the concrete
  path. Optional monitoring should not sit inside unmonitored solve loops.
- Inverters may receive an accelerator, but they should not require callers to
  know which internal kernel will consume it.
- Hot solve paths should trust already-constructed inputs. Expensive safety
  checks belong at construction time, in benchmarks, or in tests.
- Preconditioners are explicit Krylov collaborators. A missing preconditioner
  means unpreconditioned Arnoldi, not hidden diagonal magic.
- Public preconditioners keep the inverter prefix. For example,
  `InverterPreconditionerNone` and `InverterPreconditionerDiagonalInverse`
  are inverter collaborators, not package-wide solver concepts.
- Avoid external dense-solve provider dispatch in the hot path. Local
  measurements showed that dispatching to external providers was slower than
  the local nucleus path for the small systems this package currently
  materialises.

## Minimal Test Double Shape

Tests that need a tiny inverter should implement the current contract directly:

```python
class DummyInverter:
    output_mode: ClassVar[InverterOutputMode] = InverterOutputMode.overwrite

    def __call__(
        self,
        request: InverterRequest[DummyTranslation],
        output: BlockLike[DummyTranslation],
    ) -> None:
        ...
```

If the test is about operator-bound reuse, provide `instance(operator)` and
return an `InverterInstance`. Do not add `bind(...)`: it describes an abandoned
shape and makes tests misleading.

## Known Gaps

- Krylov support is still early. It works on small systems, but poor restart or
  missing preconditioning can make it fail badly even on diagonal multi-block
  probes.
- Preconditioner coverage is minimal. The current package only provides an
  `InverterPreconditionerNone` identity preconditioner and an
  `InverterPreconditionerDiagonalInverse` preconditioner for operators whose
  entries expose `inverse(source, target)`.
- Dense support is intentionally local-first. Any future external-provider
  attempt needs a benchmark that beats `InverterNucleus` before it enters the
  public design.
- Sparse, factorised, and GPU-specialised inverse actions are future work.

Use a focused benchmark before promoting a Krylov example or changing Arnoldi
defaults.
