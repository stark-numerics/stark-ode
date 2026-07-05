# Resolvent Contributor Notes

Resolvents solve the nonlinear or implicit correction problems created by
schemes. They are not usually called as often as low-level algebra kernels, but
they still sit on important hot paths for stiff methods.

## To Do

- Review resolvents for hot-path clutter and public shape before a stable
  release.
- Decide whether `ResolventChord` and `ResolventVeryChord` remain public once
  the implicit-method examples and benchmarks make their actual value clear.

## Call Path Convention

Resolvent classes should mirror schemes:

```python
def __call__(self, problem, delta):
    return self.redirect_call(problem, delta)
```

Construction should choose `redirect_call` from the available bodies:

- `call_inline` for direct Python/block operations.
- `call_specialized` for prepared specialist kernels.
- monitored wrappers only when a monitor is supplied.

This keeps optional diagnostics and backend-specific acceleration out of the
core solve loop.

## Inverters

Linearized resolvents should delegate inverse actions to inverters. They should
not grow solver-specific linear algebra internally unless that is the actual
resolvent idea being tested.

## Differential Language

Implicit equation workers expose `differential(...)` for the linearized action.
Do not keep old compatibility aliases such as `linearize(...)` before the first
public release unless they are intentionally part of the API.

Coupled implicit equation workers own their coupled differential operator.
Resolvents should ask the equation worker to refresh that operator rather than
carrying a diagonal fallback buffer. This keeps the difference between
diagonal one-stage correction systems and genuinely coupled stage systems
visible at the call site.

## Prove New Ideas Before Promotion

New resolvent ideas should usually be tested against a problem that exposes
their failure modes before being promoted. If a method can stall, the public
construction should include a finite step/rejection limit.

## Secant-Style Resolvents

The secant-style family (`ResolventAnderson`, `ResolventBroyden`, and related
prototypes) is intentionally not represented by a cheerful first-contact
example yet. These methods can be excellent when the problem and safeguards
line up, but they are also the easiest family to make look frozen or mysterious
on a poor toy problem.

Before promoting a secant-style feature example, make sure the example:

- has a finite and visible iteration/rejection limit,
- shows a problem where the method is genuinely competitive,
- explains what information is reused between iterations, and
- makes failure modes as legible as the Newton and Picard examples.
