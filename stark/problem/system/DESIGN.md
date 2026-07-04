# System Design Notes

`System` is the high-level assembly object for a problem declaration.

It ties together the problem pieces a user would recognise before selecting a
backend or numerical method:

- dynamics,
- frame,
- optional linearizer.

## Role

`System` should make the common path short:

```text
declare problem -> build IVP -> integrate
```

It should not own method algorithms, backend storage, or diagnostics. It should
coordinate those pieces through explicit `ivp(...)` arguments.

## IVP Construction

`System.ivp(...)` is where the declared problem meets:

- initial state,
- interval,
- method,
- engine,
- configuration.

This is a public assembly boundary. It is an appropriate place to adapt friendly
user inputs into core objects.

## Design Rule

Keep `System` focused on user-facing problem assembly. If a behaviour is about
how time stepping works, it belongs in methods. If it is about storage or
arithmetic, it belongs in engines. If it is about trajectory production, it
belongs in core integrator.
