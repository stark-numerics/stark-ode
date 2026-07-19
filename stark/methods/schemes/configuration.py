"""Configuration protocol and defaults consumed by scheme implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from stark.core.contracts.methods.scheme_predictor import SchemePredictorLike
from stark.core.tolerance import Tolerance
from stark.methods.schemes.predictor import SchemePredictorKnown


class SchemeConfiguration(Protocol):
    """Read-only scheme configuration shape required by scheme implementations.

    The application-wide `Configuration` object is immutable, while the
    scheme-local default configuration remains a simple mutable dataclass for
    standalone use. Read-only protocol properties let both shapes satisfy the
    same contract without coupling schemes to either concrete class.
    """

    @property
    def scheme_tolerance(self) -> Tolerance:
        ...

    @property
    def adaptive_scheme_safety(self) -> float:
        ...

    @property
    def adaptive_scheme_min_factor(self) -> float:
        ...

    @property
    def adaptive_scheme_max_factor(self) -> float:
        ...

    @property
    def adaptive_scheme_error_exponent(self) -> float:
        ...

    @property
    def adaptive_scheme_maximum_rejections(self) -> int | None:
        ...

    @property
    def scheme_predictor(self) -> SchemePredictorLike | None:
        ...


@dataclass(frozen=False, slots=True)
class SchemeConfigurationDefault:
    """Default scheme configuration used when only scheme settings are needed."""

    scheme_tolerance: Tolerance = field(default_factory=Tolerance)
    adaptive_scheme_safety: float = 0.8
    adaptive_scheme_min_factor: float = 0.1
    adaptive_scheme_max_factor: float = 5.0
    adaptive_scheme_error_exponent: float = 0.2
    adaptive_scheme_maximum_rejections: int | None = 64
    scheme_predictor: SchemePredictorLike | None = field(default_factory=SchemePredictorKnown)


__all__ = ["SchemeConfiguration", "SchemeConfigurationDefault"]
