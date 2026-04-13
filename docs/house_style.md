# STARK house style

This is the design and coding style STARK is aiming for. It is not a generic
Python style guide. It is the local language of this package.

At the surface level, STARK should look like modern Python. Under the hood, it
should feel like numerical Fortran written carefully in Python.

## Core idea

STARK is built around configured callable objects.

The package uses callable classes because they give a clean split between:

- construction time, where buffers can be allocated, coefficients resolved,
  policies chosen, and reusable work prepared;
- call time, where the hot path should look like straight numerical code.

In that sense, `__init__` plays the role of a lightweight compile/configure
phase, and `__call__` plays the role of the runtime kernel.

This is why STARK leans so heavily on objects such as:

- `Marcher`
- `Integrator`
- `Auditor`
- schemes
- resolvers
- inverters
- derivatives
- linearizers

These are not classes for their own sake. They are configured numerical
workers.

## Public shape

The public package should read as clear, modern Python:

- top-level concepts are nouns with stable identities;
- `repr` and `str` are readable and useful;
- names should be direct and unsurprising;
- modules should have confident identities.

The user should be able to read the public API as a vocabulary of numerical
objects, not as a pile of loose utilities.

## Deep shape

Inside the library, performance-sensitive code should be written with a
different instinct:

- allocate once, reuse often;
- move repeated setup work out of hot paths;
- avoid branch-heavy inner loops;
- avoid unnecessary temporary allocations;
- keep the numerical path explicit enough that it still reads like the method.

The goal is not to make Python imitate C syntax. The goal is to write Python in
a way that respects the structure of numerical code.

## Callable classes

When a numerical component has setup work, scratch state, or a stable
mathematical role, prefer a callable class over loose helper functions.

Good reasons to use a callable class:

- it owns buffers or scratch storage;
- it resolves kernels or fast paths at construction time;
- it chooses a safe or unsafe execution path once, not on every call;
- it represents a mathematical object with a real identity;
- it will be composed with other workers.

Less convincing reasons:

- the code is merely long;
- the function is only being wrapped to look object-oriented;
- the object would have no state, no setup, and no identity beyond the call.

## Hot paths

In the hot path, STARK prefers straight-line numerical code.

That means:

- resolve branches before `__call__` where possible;
- bind safe and unsafe call paths once rather than checking repeatedly;
- pull frequently used methods, buffers, and coefficients into local names;
- keep the main numerical routine readable as mathematics;
- use jitted kernels where they buy something real.

Safety checks are welcome at the boundary. They should not be smeared across
the core loop unless they are essential to correctness.

## Naming

Naming is a serious part of the style.

- Prefer one-word module names.
- Multi-word concept names should usually belong to classes, not modules.
- Top-level concepts should be nouns.
- Helper modules with weak names are a code smell.

STARK generally prefers:

- `marcher.py`
- `integrate.py`
- `control.py`
- `butcher_tableau.py`

over names that feel temporary or apologetic.

Underscore-prefixed module names are acceptable for genuinely private
implementation details, especially tiny jitted kernels or local technical glue.
They should not become a hiding place for major concepts. If a module contains
real mathematical or architectural machinery, it should usually have a proper
name and a proper place in the package.

## Maths-facing code

When code is implementing a numerical method, it should read like the method.

Good signs:

- coefficients are explicit and visible;
- method structure is recognizable from the literature;
- the main call path is not buried under framework plumbing;
- the reader can connect the code to the underlying equations.

The existing scheme implementations are close to the target here. Library-level
code should tend toward that style.

## Workspaces and support objects

Scratch allocation and non-mathematical support should be owned explicitly.

STARK likes support objects such as:

- `SchemeWorkspace`
- `ResolverWorkspace`
- `InverterWorkspace`

These exist to keep numerical workers focused on the algorithm rather than on
ad hoc buffer management.

Support objects should contain work that is genuinely shared and family-level.
They should not become junk drawers for code that lacks a clear home.

## Composition

STARK methods are built by composing workers.

Examples:

- a `Marcher` couples a scheme to tolerances;
- an `Integrator` repeatedly drives a marcher;
- a resolver can be combined with different schemes;
- an inverter can be swapped beneath a resolver;
- problem-specific derivatives, linearizers, and fast paths plug into the same
  overall machinery.

This composition is not an implementation accident. It is part of the package
style and one of the main reasons the object vocabulary matters.

## Fast paths and JIT

Fast paths are part of the intended design, not an embarrassing escape hatch.

STARK expects problem-specific code to be able to provide:

- fused translation kernels;
- jitted derivative kernels;
- jitted Jacobian actions;
- specialized workbench allocation behavior;
- custom resolvers and inverters where the problem demands it.

The generic path should stay clean and correct. The optimized path should have
an obvious place to live.

## Helper functions

Small helper functions are not banned, but they should justify themselves.

When a helper starts to accumulate any of the following:

- scratch state;
- reusable numerical structure;
- policy choices;
- mathematical identity;
- repeated collaboration with other workers;

it should usually be reconsidered as a real object.

As a rule of thumb, if a helper function feels like part of the architecture,
it probably wants a better home.

## Audit and failure

STARK prefers:

- explicit contracts;
- early validation;
- informative failure;
- readable diagnostics.

The package should fail fast when an object does not satisfy the expected
protocol, rather than continuing with vague errors later.

## In practice

When writing new STARK code, ask:

1. Is this a real numerical worker with setup work or mathematical identity?
2. Can allocation, policy selection, or branch selection move into `__init__`
   or `bind(...)`?
3. Does the hot path read like the method, or like framework plumbing?
4. Is this module name confident enough to deserve its place?
5. Is this support code truly shared, or just homeless?
6. Would a scientific reader recognize the mathematics in the code?

If the answer to those questions is mostly yes, the code is probably moving in
the right direction.
