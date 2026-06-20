# STARK house style

Last reviewed: 2026-06-17.

This document is for contributors and future maintainers. It records the style of the project: not just formatting preferences, but the design habits that keep STARK coherent.

STARK is allowed to be layered. It is not allowed to be vague.

## The shape of the package

STARK separates four concerns:

```text
problem      what is being solved
methods      how time advancement is performed
engines      where storage and arithmetic live
diagnostics  how behaviour is observed
```

Code should normally belong clearly to one of those domains.

A derivative belongs to the problem. A scheme belongs to methods. A carrier belongs to engines. A monitor belongs to diagnostics. Shared low-level protocols belong in core contracts only when they genuinely decouple domains.

Do not move concrete workers into `core` just because more than one place can see them. `core` should describe stable contracts and shared primitives, not become a drawer for implementation.

## Callable classes are intentional

STARK deliberately uses many callable objects.

A callable class gives the package two things at once:

```text
it looks like a function at the call site
it can prepare and retain repeated work in __init__
```

That pattern is central to the project.

Prefer this:

```python
scheme = SchemeKvaerno5(configuration=configuration)
scheme(interval, state, output)
```

over this:

```python
call_scheme_kvaerno5(configuration, interval, state, output)
```

when the implementation has any preparation, scratch allocation, policy selection, dispatch choice, or reusable helper state.

The call should be simple. The preparation should happen before the call.

Good callable objects in STARK often do some of the following in `__init__`:

```text
store configuration
resolve default policies
allocate scratch objects
choose monitored or unmonitored call paths
prepare kernels or specialists
bind stable collaborators
precompute tableau-dependent helpers
```

Then `__call__` should do the work directly.

This is the guiding rule:

```text
Discovery belongs in construction.
Execution belongs in __call__.
```

Do not repeatedly rediscover support, inspect types, allocate avoidable scratch, or choose strategy inside a hot call if that choice can be made once.

## Function-like at the boundary, object-like inside

A STARK worker should usually feel like a function from outside:

```python
derivative(interval, state, out)
linearizer(interval, state, operator)
resolvent(request, output)
inverter(request, output)
scheme(interval, state, output)
```

But inside, it may carry prepared state. This is not a contradiction. It is the house style.

The user should not have to care that a derivative adapter has resolved field access, or that an inverter has selected a fast single-block path, or that a scheme has selected a predictor. The call site should stay readable.

## Owner-first naming

Names should identify ownership before role.

Use:

```text
FrameField
AlgebraistFrame
AlgebraistFrameField
InverterKrylovBasis
InverterKrylovProjection
SchemePredictorKnown
```

Avoid:

```text
Field
PreparedFrame
KrylovBasis
KnownSchemePredictor
```

The project uses compound names left-to-right:

```text
owner concept -> subordinate concept -> specialisation
```

This keeps names searchable and keeps families visually grouped.

If a thing is owned by `Frame`, its name starts with `Frame`. If a thing is owned by `InverterKrylov`, its name starts with `InverterKrylov`.

Do not shorten names by removing the owner when the owner carries architectural meaning.

## Protocols and concrete workers

Protocols describe what one domain needs from another. Concrete classes do the work.

Use protocol names when the object crosses a boundary:

```text
DerivativeLike
LinearizerLike
SchemeLike
InverterLike
SchemePredictorLike
```

Use concrete names inside the owning domain:

```text
SchemeKvaerno5
ResolventNewton
InverterDense
InverterKrylovArnoldi
SchemePredictorKnown
```

A protocol may live in `stark.core.contracts` if it prevents an import cycle or decouples domains.

A concrete implementation should live in its domain.

For example:

```text
core.contracts.SchemePredictorLike
methods.schemes.predictors.SchemePredictorKnown
```

`core` may know the shape. It should not import the concrete method worker.

## No hidden dependencies

Optional acceleration and backend support should be explicit.

Do not quietly add NumPy, SciPy, JAX, CuPy, Numba, or any other dependency to a path that is supposed to be generic.

If an implementation is backend-specific, make that clear in its domain, name, example, or optional dependency handling.

A pure-Python path may be slower. That is acceptable if it is honest, portable, and structurally correct.

A fast path may depend on an accelerator. That dependency should be explicit.

## Keep hot paths lean

The hot path is not where the package teaches users how to call it.

Validate at boundaries. Prepare once. Run lean.

Avoid adding checks inside inner calls just to make impossible internal states produce prettier errors.

This is especially important in:

```text
scheme stage loops
resolvent correction loops
inverter iterations
operator application
translation arithmetic
monitor-free timing paths
```

Invalid public input should be caught at a public boundary. Invalid internal construction can fail plainly.

Do not burden every correct step with checks for a state the package should never have constructed.

## Monitoring is not timing

Monitoring is diagnostic machinery. Timing is performance measurement.

Do not silently wire monitors into benchmark paths and then compare timings.

A monitored run is for understanding behaviour:

```text
step counts
rejected steps
residuals
linear solves
corrections
```

An unmonitored run is for measuring speed.

Competition reports should make this distinction clear. Preparation, warm timing, and total timing are different quantities and should not be collapsed into a single headline.

## Public examples are executable truth

Examples are part of the public API.

They should be:

```text
readable
short enough to understand
runnable as modules
honest about optional dependencies
focused on one idea
```

A feature example should answer one question. A case study may tell a longer story.

Do not use examples as dumping grounds for every option. If an example needs too many concepts, split it.

Docs may include short snippets, but full scripts should live in `examples/`.

## Prefer current families over compatibility clutter

Do not keep old names alive unless compatibility is an explicit project decision.

Compatibility aliases, legacy wrappers, and deprecated paths are not free. They affect docs, tests, examples, search results, and contributor understanding.

If a family is being replaced, move examples and docs to the new family before deleting the old code.

Once the public surface has moved, remove stale public exports.

## Method components should resemble each other

Schemes, resolvents, and inverters should feel like members of the same architecture.

They should prefer:

```text
configuration protocols
prepared workers
descriptor/display support where useful
monitor-free and monitored call paths where relevant
instance/prepared forms when repeated work can be reused
```

Avoid one-off styles unless the algorithm genuinely requires it.

A new inverter should not look like an unrelated subsystem. A new scheme should not invent a private configuration pattern. A new resolvent should not smuggle method policy into an operator or inverter.

## Keep roles separate

A scheme owns stage structure.

A predictor seeds stage guesses.

A resolvent solves nonlinear stage equations.

An inverter solves linear correction equations.

A linearizer supplies Jacobian action.

An engine supplies storage and arithmetic support.

A monitor observes.

Do not move responsibilities across those boundaries just because it is convenient for one benchmark.

If a performance improvement requires crossing a boundary, make that crossing explicit and local.

## Documentation style

Documentation should be layered, but user pages should still solve user problems.

Start with the task:

```text
solve an ODE
choose a backend
solve a stiff problem
monitor a solve
integrate a foreign model
```

Then explain the architecture needed for that task.

Do not flatten all concepts into the first page. Do not write a map when the user needs a route.

## Final rule

When in doubt, ask:

```text
Does this make the public path clearer?
Does this let a user solve a problem?
Does this keep repeated work out of hot calls?
Does this preserve the domain boundary?
Does this name reveal ownership?
Does this avoid future cleanup work?
```

If the answer is no, slow down.
