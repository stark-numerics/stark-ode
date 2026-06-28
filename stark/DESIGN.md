# Package Surface Design Notes

The top-level `stark` package should stay deliberately small.

Its job is to make the simplest examples readable, not to re-export every
advanced contract, backend, scheme, resolvent, or inverter.

## Intended Surface

The top-level surface should favour names a new user needs to declare and run a
basic problem:

```text
Frame
Derivative / DerivativeStyle
Linearizer / LinearizerStyle
System
Method
Interval
Tolerance
Configuration
Monitor
```

Advanced users can import deeper domains directly:

```text
stark.methods
stark.engines
stark.core
stark.diagnostics
stark.problem
```

## Review Notes

`Auditor`, `AuditError`, `Integrator`, `IntegratorConfiguration`, and
`IntegratorStepper` are intentionally hidden under `stark.core`. They are
available for advanced users, but they are not first-contact problem-declaration
objects.

## Design Rule

If an example aimed at a new user does not need the name, think carefully before
adding it to `stark.__init__`.
