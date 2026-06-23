# Block Design Notes

Blocks are method-internal product spaces over translations.

Implicit and multi-stage methods often need to solve equations involving
several stage increments at once. A `Block` lets those methods work with:

```text
T x T x ... x T
```

without flattening the user's original state representation.

## Role

Block machinery is a bridge between method algorithms and translation algebra:

- `Block` groups translations.
- Block operators apply matrix-free actions on grouped translations.
- Block bases expose coordinate views when an algorithm genuinely needs them.
- Materialisers build dense coordinate matrices only for methods that request
  dense linear algebra.
- Specialists provide prepared operations for repeated block updates.

## What Blocks Are Not

Blocks are not a user modelling layer. Users should normally model state with
`Frame`, `System`, or a foreign state/translation pair.

Blocks are also not the default path to dense arrays. Dense materialisation is
an optional view used by dense inverters and related experiments; it should not
become the core representation of a solve.

## Design Rule

Keep block operations generic over translation objects. A block should preserve
the user's state shape by working with translations, not bypass it by assuming a
flat vector unless a basis/materialiser has been explicitly requested.
