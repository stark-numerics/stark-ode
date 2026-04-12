from __future__ import annotations

from typing import Protocol

from stark.contract_protocols.linear_algebra import Translation


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


__all__ = [
    "Combine2",
    "Combine3",
    "Combine4",
    "Combine5",
    "Combine6",
    "Combine7",
    "Scale",
]
