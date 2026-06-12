from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AlgebraistArity:
    """Request for a general linear-combination kernel of the given arity.

    Arity convention:
        1 -> scale
        2 -> combine2
        3 -> combine3
        ...
    """

    value: int

    def __post_init__(self) -> None:
        if type(self.value) is not int:
            raise TypeError("AlgebraistArity value must be an integer.")
        if self.value < 1:
            raise ValueError("AlgebraistArity value must be at least 1.")
