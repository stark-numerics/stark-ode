# STARK object map

STARK has a small set of object families that appear repeatedly across the
interface layer, core schemes, implicit solvers, and performance hooks. This
page describes what each family is for and whether users are expected to
supply their own objects.

## Everyday user objects

Most users start with `stark.interface.StarkIVP`. In that path, STARK builds
the core objects for you from an initial value and a derivative callable.

- **Initial value**
  The scalar, sequence, NumPy array, CuPy array, or JAX array supplied to
  `StarkIVP`. Users always provide this.

- **Derivative**
  The right-hand side of the ODE. Users usually provide this. It can be a
  return-style callable such as `f(t, y) -> dy`, or an in-place callable marked
  with `Derivative.in_place`.

- **Interval**
  The current time, proposed step, and stop time. Users provide this directly
  with `stark.Interval`.

## Core integration objects

The core API is useful when the problem already has its own state model or
when advanced users need control over schemes, workspaces, accelerators, or
fast paths.

- **State**
  The nonlinear problem value being advanced. In the interface layer this is a
  wrapper around the initial value. In the core API it is often a user-defined
  object such as a particle system, field bundle, or nested dataclass.

- **Translation**
  The linear increment used by Runge-Kutta stages. A translation can be scaled,
  combined, measured with a norm, and applied to a state. Advanced users may
  provide their own translation class when a dense array is not the natural
  representation.

- **Allocator**
  The allocator and copier for states and translations. Schemes ask a
  allocator for blank buffers rather than knowing how a user state is built.
  Core users supply a allocator when they supply custom states and
  translations.

- **Scheme**
  A one-step method such as Euler, RK4, Cash-Karp, backward Euler, or an IMEX
  pair. Users choose schemes. Advanced users can supply custom scheme objects
  if they satisfy the scheme contract.

- **Configuration**
  Runtime policy: tolerances, adaptive regulation checks, and selected
  accelerator. Users may supply one when they need non-default tolerances or
  execution policy.

- **IntegratorStepper**
  Couples a scheme and Configuration into one accepted-step operation. Users touch
  this when using the core API directly.

- **Integrator**
  Repeatedly calls a stepper over an interval. Users touch this when using the
  core API directly or when they need snapshot/live iteration control.

## Algebra and performance objects

These objects exist to keep scheme code independent from the representation of
state increments, while still allowing optimized paths where the representation
is known.

- **Algebraist**
  Provides generated or runtime translation-algebra kernels from explicit
  layout metadata. General providers bind arity-based `linear_combine` kernels
  to a translation type. Specialist providers give schemes fixed-coefficient
  kernels for repeated tableau combinations. Accelerated providers may have
  noticeable compilation cost, so they are best suited to large states, long
  integrations, or repeated solves. Algebraist does not generate resolvent
  iterations, convergence checks, inverter logic, or preconditioner internals.

- **Algebraist layout field**
  Describes how one translation field maps to one state field and which layout
  policy should generate its code. Users define fields when they build an
  Algebraist provider.

- **Algebraist layout policy**
  Controls how generated code treats a field: broadcasted array operations,
  looped compiled kernels, or small fixed unrolled shapes. Advanced users
  choose policies when performance or representation details matter.

- **Accelerator**
  A configured compiler/backend worker such as no acceleration, Numba, or JAX.
  Users may provide one through an Configuration or Algebraist when they want
  accelerated generated kernels.

## Implicit-solver objects

Implicit and IMEX schemes need extra objects because a stage may require
solving a nonlinear or linear problem.

- **Resolvent**
  Solves the nonlinear stage problem for an implicit scheme. Users choose or
  configure resolvents for implicit and IMEX methods, and advanced users can
  supply their own.

- **Inverter**
  Solves linear systems used by Newton-like resolvents. Users usually choose a
  built-in inverter such as GMRES, FGMRES, or BiCGStab, but advanced users can
  supply custom inverters.

- **DerivativeIMEX**
  Splits a derivative into implicit and explicit parts for IMEX schemes. Users
  supply this when choosing an IMEX scheme.

## Checking and comparison objects

- **Auditor**
  Checks that supplied objects satisfy the contracts needed by a run. Users can
  call it before long or expensive integrations.

- **ComparisonRunner**
  Runs several solver setups on the same problem and reports timing,
  diagnostics, and profiling summaries. It is mainly for examples,
  comparisons, and development investigations.

## Rule of thumb

For ordinary scalar or array problems, supply an initial value, derivative, and
interval through `StarkIVP`.

Move to the core API when your state representation matters. Supply custom
states, translations, and a allocator when flattening would obscure the model.
Supply resolvents, inverters, accelerators, or an Algebraist when the numerical
method or performance path needs more control.
