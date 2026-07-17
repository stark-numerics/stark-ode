"""Contracts for translation linear-combination fast paths.

`LinearCombine` is a tuple of callables indexed by arity: entry 0 scales one
translation, entry 1 combines two translations, and so on. The arity-specific
protocols below document the intended calling convention for common entries,
but the exported table type stays deliberately broad so generated and
user-provided kernels can duck type cleanly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeAlias

from stark.core.contracts.translation import Translation, TranslationType


class LinearCombineScaleLike(Protocol[TranslationType]):
    """Set `out = a * x` and return `out`.

    The protocol is generic because scale kernels are commonly attached to a
    concrete translation implementation. Preserving that translation type keeps
    predictor and scheme hot-path helpers type-checkable without forcing tests
    or user extensions to erase their richer state shape.
    """

    def __call__(self, a: float, x: TranslationType, out: TranslationType) -> TranslationType:
        ...


class LinearCombineArity2Like(Protocol):
    """Set `out = a0 * x0 + a1 * x1` and return `out`."""

    def __call__(
        self,
        a0: float,
        x0: Translation,
        a1: float,
        x1: Translation,
        out: Translation,
    ) -> Translation:
        ...


class LinearCombineArity3Like(Protocol):
    """Set `out = a0 * x0 + a1 * x1 + a2 * x2` and return `out`."""

    def __call__(
        self,
        a0: float,
        x0: Translation,
        a1: float,
        x1: Translation,
        a2: float,
        x2: Translation,
        out: Translation,
    ) -> Translation:
        ...


class LinearCombineArity4Like(Protocol):
    """Set `out = a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3` and return `out`."""

    def __call__(
        self,
        a0: float,
        x0: Translation,
        a1: float,
        x1: Translation,
        a2: float,
        x2: Translation,
        a3: float,
        x3: Translation,
        out: Translation,
    ) -> Translation:
        ...


class LinearCombineArity5Like(Protocol):
    """Set `out = a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4` and return `out`."""

    def __call__(
        self,
        a0: float,
        x0: Translation,
        a1: float,
        x1: Translation,
        a2: float,
        x2: Translation,
        a3: float,
        x3: Translation,
        a4: float,
        x4: Translation,
        out: Translation,
    ) -> Translation:
        ...


class LinearCombineArity6Like(Protocol):
    """Set `out = a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5` and return `out`."""

    def __call__(
        self,
        a0: float,
        x0: Translation,
        a1: float,
        x1: Translation,
        a2: float,
        x2: Translation,
        a3: float,
        x3: Translation,
        a4: float,
        x4: Translation,
        a5: float,
        x5: Translation,
        out: Translation,
    ) -> Translation:
        ...


class LinearCombineArity7Like(Protocol):
    """Set `out = a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6` and return `out`."""

    def __call__(
        self,
        a0: float,
        x0: Translation,
        a1: float,
        x1: Translation,
        a2: float,
        x2: Translation,
        a3: float,
        x3: Translation,
        a4: float,
        x4: Translation,
        a5: float,
        x5: Translation,
        a6: float,
        x6: Translation,
        out: Translation,
    ) -> Translation:
        ...


class LinearCombineArity8Like(Protocol):
    """Set `out` to an eight-term linear combination and return `out`."""

    def __call__(self, *terms: object) -> Translation:
        ...


class LinearCombineArity9Like(Protocol):
    """Set `out` to a nine-term linear combination and return `out`."""

    def __call__(self, *terms: object) -> Translation:
        ...


class LinearCombineArity10Like(Protocol):
    """Set `out` to a ten-term linear combination and return `out`."""

    def __call__(self, *terms: object) -> Translation:
        ...


class LinearCombineArity11Like(Protocol):
    """Set `out` to an eleven-term linear combination and return `out`."""

    def __call__(self, *terms: object) -> Translation:
        ...


class LinearCombineArity12Like(Protocol):
    """Set `out` to a twelve-term linear combination and return `out`."""

    def __call__(self, *terms: object) -> Translation:
        ...


LinearCombine: TypeAlias = tuple[Callable[..., Any], ...]
"""Linear-combination kernels indexed by translation arity.

A `LinearCombine` table is attached to an allocator or translation-like object
when that object wants to provide its own hot-path linear algebra.

Table layout:

- `linear_combine[0]`: `scale(a, x, out)` writes `out = a * x`
- `linear_combine[1]`: `combine2(a0, x0, a1, x1, out)` writes two terms
- `linear_combine[n - 1]`: combines `n` `(scalar, translation)` pairs into `out`

Every kernel must mutate the supplied `out` translation and return `out`.
These callables sit on scheme hot paths, so custom implementations should avoid
runtime branching, type inspection, allocation, and generic `*terms` parsing
where performance matters.

If a custom allocator wants to provide low-arity handwritten kernels, declare
them as optional optimisation seeds. `Allocator.runtime` prepares a complete
table when the allocator is constructed:

```python
@Allocator.runtime
@Allocator.linear_combine(scale, combine2)
class MyAllocator:
    ...

allocator = MyAllocator(...)
```
"""


class LinearCombineSupporting(Protocol):
    """Translation-like object with generic linear-combination kernels.

    `linear_combine[0]` is `scale`; `linear_combine[1]` is `combine2`;
    higher entries are `combine3`, `combine4`, and so on. Kernels mutate and
    return the output translation. Built-in schemes currently expect entries
    for arities 1..12 after allocator setup. Users can supply optional
    low-arity seeds with `Allocator.linear_combine(...)`; `Allocator.runtime`
    can synthesize the remaining entries.
    """

    linear_combine: LinearCombine


__all__ = [
    "LinearCombine",
    "LinearCombineArity10Like",
    "LinearCombineArity11Like",
    "LinearCombineArity12Like",
    "LinearCombineArity2Like",
    "LinearCombineArity3Like",
    "LinearCombineArity4Like",
    "LinearCombineArity5Like",
    "LinearCombineArity6Like",
    "LinearCombineArity7Like",
    "LinearCombineArity8Like",
    "LinearCombineArity9Like",
    "LinearCombineScaleLike",
    "LinearCombineSupporting",
]









