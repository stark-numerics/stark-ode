# Shared Engine Design Notes

`stark.engines.shared` holds engine support that is genuinely shared across
backend packages.

It is not a dumping ground for code that feels vaguely backend-related. A
future backend such as Torch should be able to live mostly inside its own
backend package. Shared support earns its place only when it describes a
cross-backend concept.

## What Belongs Here

Shared engine support currently covers:

```text
accelerators   optional compilation/fusion interfaces
algebraist     prepared algebra for known Frame-backed state
basis          coordinate bases for backend-owned translations
```

Backend-specific carrier arithmetic, dtype policy, array transfer, and target
expression choices should stay with the backend unless two or more backends
really share the same implementation.

## Design Rule

When adding a shared helper, ask:

```text
Would a Torch backend naturally use this exact object?
Would moving it here make backend code easier to inspect?
Does it preserve the Engine as the owner of backend policy?
```

If the answer is no, keep the helper local to the backend.
