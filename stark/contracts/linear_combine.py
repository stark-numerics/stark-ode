from __future__ import annotations

from typing import Protocol

from stark.contracts.translations import Translation


class Scale(Protocol):
    """Set `out = a * x` and return `out`."""

    def __call__(self, out: Translation, a: float, x: Translation) -> Translation:
        ...


class Combine2(Protocol):
    """Set `out = a0 * x0 + a1 * x1` and return `out`."""

    def __call__(
        self,
        out: Translation,
        a0: float,
        x0: Translation,
        a1: float,
        x1: Translation,
    ) -> Translation:
        ...


class Combine3(Protocol):
    """Set `out = a0 * x0 + a1 * x1 + a2 * x2` and return `out`."""

    def __call__(
        self,
        out: Translation,
        a0: float,
        x0: Translation,
        a1: float,
        x1: Translation,
        a2: float,
        x2: Translation,
    ) -> Translation:
        ...


class Combine4(Protocol):
    """Set `out = a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3` and return `out`."""

    def __call__(
        self,
        out: Translation,
        a0: float,
        x0: Translation,
        a1: float,
        x1: Translation,
        a2: float,
        x2: Translation,
        a3: float,
        x3: Translation,
    ) -> Translation:
        ...


class Combine5(Protocol):
    """Set `out = a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4` and return `out`."""

    def __call__(
        self,
        out: Translation,
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
    ) -> Translation:
        ...


class Combine6(Protocol):
    """Set `out = a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5` and return `out`."""

    def __call__(
        self,
        out: Translation,
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
    ) -> Translation:
        ...


class Combine7(Protocol):
    """Set `out = a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6` and return `out`."""

    def __call__(
        self,
        out: Translation,
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
    ) -> Translation:
        ...


class Combine8(Protocol):
    """Set `out` to an eight-term linear combination and return `out`."""

    def __call__(self, out: Translation, *terms: object) -> Translation:
        ...


class Combine9(Protocol):
    """Set `out` to a nine-term linear combination and return `out`."""

    def __call__(self, out: Translation, *terms: object) -> Translation:
        ...


class Combine10(Protocol):
    """Set `out` to a ten-term linear combination and return `out`."""

    def __call__(self, out: Translation, *terms: object) -> Translation:
        ...


class Combine11(Protocol):
    """Set `out` to an eleven-term linear combination and return `out`."""

    def __call__(self, out: Translation, *terms: object) -> Translation:
        ...


class Combine12(Protocol):
    """Set `out` to a twelve-term linear combination and return `out`."""

    def __call__(self, out: Translation, *terms: object) -> Translation:
        ...


LinearCombine = tuple[
    Scale
    | Combine2
    | Combine3
    | Combine4
    | Combine5
    | Combine6
    | Combine7
    | Combine8
    | Combine9
    | Combine10
    | Combine11
    | Combine12,
    ...,
]


class SupportsLinearCombine(Protocol):
    """
    Translation-like object with generic vector linear-combination kernels.

    `linear_combine[0]` is `scale`; `linear_combine[1]` is `combine2`;
    higher entries are `combine3`, `combine4`, and so on. This remains the
    generic scheme/workspace fast-path contract. Tableau-specialized kernels
    should use a separate contract rather than overloading this tuple.
    """

    linear_combine: LinearCombine


__all__ = [
    "Combine10",
    "Combine11",
    "Combine12",
    "Combine2",
    "Combine3",
    "Combine4",
    "Combine5",
    "Combine6",
    "Combine7",
    "Combine8",
    "Combine9",
    "LinearCombine",
    "Scale",
    "SupportsLinearCombine",
]









