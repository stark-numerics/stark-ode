# Methods

This page is for users who want to choose or replace numerical method components.

The high-level object is `Method`. It is a recipe for constructing an integration method from smaller pieces.

## Method

A method commonly specifies a scheme:

```python
Method(scheme=SchemeCashKarp)
```

Implicit methods add nonlinear and linear solve components:

```python
Method(
    scheme=SchemeKvaerno3,
    resolvent=ResolventNewton(...),
)
```

## Schemes

A scheme advances the solution by one accepted step or one trial step. Examples include:

- fixed explicit schemes such as Euler and RK4;
- adaptive explicit schemes such as Cash-Karp, Dormand-Prince, and Tsitouras;
- implicit schemes such as backward Euler, SDIRK, and Kvaerno methods;
- IMEX schemes that split explicit and implicit contributions.

Users normally choose a built-in scheme. Advanced users can implement the scheme protocol when they need a new time-stepping method.

See:

```powershell
python -m examples.features.custom_scheme_fixed_explicit
```

## Scheme predictors

Implicit stage solves need an initial guess for the stage unknown. A scheme predictor supplies that guess.

Common policies include:

- known-shift prediction;
- zero prediction;
- previous-stage prediction.

Predictors belong to schemes because schemes know the stage structure. Resolvents and inverters should not need to know where an initial guess came from.

See:

```powershell
python -m examples.features.scheme_predictor
```

## Resolvents

A resolvent solves a nonlinear stage equation. Examples include:

- Picard-style fixed-point iteration;
- Newton iteration;
- chord and very-chord variants;
- Anderson and Broyden-style nonlinear acceleration where available.

A resolvent builds the correction problem and asks an inverter to solve the linear or linearized part when necessary.

## Inverters

An inverter solves a linear correction equation for a resolvent. Current inverter families include:

- dense inverters for small systems;
- relaxation inverters for iterative structured examples;
- Krylov inverters for matrix-free large systems.

Dense inverters are usually right for small stiff systems. Krylov inverters are intended for large translation spaces where dense materialisation is too expensive or impossible.

See:

```powershell
python -m examples.features.inverter_dense
python -m examples.features.inverter_krylov
```

## Monitors

Monitors observe method internals such as accepted steps, residual norms, or inverter defects. Monitoring is useful for diagnostics but changes hot-path cost. Timing reports should normally use unmonitored solves.

See [Diagnostics](diagnostics.md).
