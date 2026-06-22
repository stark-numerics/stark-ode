# Resolvent Contributor Notes

Resolvents solve the nonlinear or implicit correction problems created by
schemes. They are not usually called as often as low-level algebra kernels, but
they still sit on important hot paths for stiff methods.

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

## Experiments First

New resolvent ideas should usually start in `benchmarks/experiments` or a
competition example before being promoted. If a method can stall, the experiment
should expose the failure mode and the public construction should include a
finite step/rejection limit.
