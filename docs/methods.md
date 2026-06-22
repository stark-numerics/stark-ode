# Choose and customise a method

This page is for users who want to choose how STARK advances time.

The high-level object is `Method`:

```python
method = Method(SchemeCashKarp)
```

A method can also carry a resolvent, inverter, predictor, and other prepared pieces when you use implicit or custom solves.

## Which method should I choose?

| Problem type | First choice | Why |
|---|---|---|
| Non-stiff, ordinary ODE | `SchemeCashKarp` | Adaptive explicit scheme, good default for examples. |
| Non-stiff, fixed step | `SchemeRK4` | Simple and predictable. |
| Small stiff system | Kvaerno/SDIRK + Newton + Dense | Dense linear algebra is cheap for small systems. |
| Large matrix-free stiff system | SDIRK/Kvaerno + Newton + Krylov | Avoids dense materialisation. |
| Split explicit/implicit problem | IMEX scheme | Keeps cheap explicit and stiff implicit parts separate. |
| Debugging or teaching | Fixed schemes, relaxation inverters | Easier to inspect than adaptive implicit stacks. |

Run the scheme selection example:

```powershell
python -m examples.getting_started.choose_scheme
```

## Schemes advance time

A scheme owns the stage structure and step update. Examples:

```python
from stark.methods import SchemeCashKarp, SchemeKvaerno3, SchemeRK4
```

Use explicit schemes until stiffness forces an implicit method.

## Resolvents solve nonlinear implicit equations

Implicit schemes create nonlinear stage equations. A resolvent solves them.

Common choices:

```text
Picard       simple fixed-point iteration
Newton       uses a linearizer and inverter
Chord        reuses a linearization for part of the solve
VeryChord    reuses a linearization more aggressively
```

Use Newton when you can provide a linearizer. Use simpler resolvents for teaching or when the equation is mild.

## Inverters solve linear correction equations

Newton-style resolvents need to solve linear correction equations.

| Inverter | Use when |
|---|---|
| `InverterDense` | The system is small or dense materialisation is cheap. |
| `InverterKrylovArnoldi` | The system is large and you can apply the linear operator without building a dense matrix. |
| relaxation inverters | You want a simple iterative method, teaching path, or structured baseline. |

Run:

```powershell
python -m examples.features.inverter_dense
python -m examples.features.inverter_krylov
python -m examples.features.inverter_relaxation_richardson
```

## Predictors seed implicit stage guesses

A scheme predictor sets the initial guess for an implicit stage solve. It belongs to the scheme layer, not the resolvent or inverter layer.

Examples:

```text
SchemePredictorKnown      seed from the known explicit/stage shift
SchemePredictorZero       start from zero
SchemePredictorPrevious   reuse the previous solved increment when available
```

Run:

```powershell
python -m examples.features.scheme_predictor
```

## Custom method pieces

You can keep the high-level `System` and `Frame` path and replace one numerical component at a time:

```text
replace the scheme       write a SchemeLike
replace nonlinear solve  write a ResolventLike
replace linear solve     write an InverterLike
replace stage guess      write a SchemePredictorLike
observe behaviour        write a monitor
```

Start with [Extending STARK](extending.md) when you want to implement one of these pieces.
