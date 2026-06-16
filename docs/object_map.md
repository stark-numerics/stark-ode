# STARK object map

STARK is organized around a small set of object families. Ordinary users mostly
work with the problem layer; advanced users can supply lower-level contracts
when they need custom state, custom algebra, implicit solvers, or acceleration.

## Problem objects

- **System**
  A reusable problem declaration. It combines a derivative, a `Frame`, and
  optional problem-level ingredients such as a linearizer or inner product.

- **Frame**
  Declares named state fields, their corresponding translation fields, shapes,
  and norm policy. Engines use the frame to allocate state and translation
  objects.

- **Derivative**
  The right-hand side of the ODE. Users normally provide this through
  `DerivativeStyle`: return-style, in-place, field-level kernel, or returning
  field-level kernel.

- **Linearizer**
  A problem-level worker for implicit methods. It prepares the scheme-facing
  `LinearizerLike` callable that configures a local Jacobian operator.

- **SystemIVP**
  A prepared initial-value problem created by `System.ivp(...)`. It owns the
  engine, prepared initial state, interval template, scheme, stepper, and
  integrator used for repeated solves.

## Core integration objects

- **State**
  The nonlinear value being advanced.

- **Translation**
  The linear increment used by Runge-Kutta stages. A translation can be scaled,
  combined, measured, and applied to a state.

- **Allocator**
  Allocates blank states/translations and copies states.

- **Interval**
  Current time, proposed step size, and stop time.

- **Tolerance** and **Configuration**
  Runtime policy for tolerances, progress checks, adaptive behaviour, and method
  options such as scheme predictors.

- **IntegratorStepper** and **Integrator**
  The stepper performs one accepted step. The integrator repeatedly calls a
  stepper over an interval and yields snapshots or mutable working objects.

## Method objects

- **Method**
  A user-facing numerical recipe that selects a scheme and optional method
  components.

- **Scheme**
  The time-stepping formula: explicit, adaptive, implicit, or IMEX.

- **Resolvent**
  Solves nonlinear stage equations for implicit schemes.

- **Inverter**
  Provides inverse actions for linear systems used by Newton-like resolvents.
  Dense and relaxation inverters use the newer request-shaped protocol. The
  legacy Krylov family is still available while the refreshed Krylov API is
  being completed.

- **SchemePredictor**
  A scheme-owned strategy for seeding implicit stage solves.

## Engine and algebra objects

- **Engine**
  Owns backend allocation and execution details for Native, NumPy, JAX, or CuPy
  state fields.

- **Carrier**
  Engine-owned storage for a field. Most users interact with engines rather than
  carriers directly.

- **Accelerator**
  A compiler/backend worker such as no acceleration, Numba, or JAX. Accelerators
  are passed to the workers that can use them.

- **Algebraist**
  Generates or supplies translation-algebra kernels from frame metadata.
  Algebraist is useful when repeated stage combinations dominate runtime.

## Diagnostics objects

- **Monitor**
  Records detailed solver activity. Monitoring is useful for diagnostics but is
  deliberately separate from timing-oriented runs.

- **Comparison** and **competition reports**
  Helpers for comparing solver setups and reporting accuracy, preparation time,
  warm-run time, and total time.

## Extension rule of thumb

Use the high-level `System` layer while the problem can be described as named
fields. Drop to core contracts when the simulation already owns a richer state
model or when a custom method, resolvent, inverter, engine, or algebra path is
part of the problem.
