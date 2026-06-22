from __future__ import annotations

from dataclasses import dataclass, field

from stark.core.contracts.scheme_predictor import SchemePredictorLike
from stark.core.tolerance import Tolerance


@dataclass(frozen=True, slots=True)
class Configuration:
    """
    Immutable configuration shared by schemes, resolvents, and inverters.

    The scheme fields control adaptive step-size tolerance and safety factors.
    Resolvent and inverter fields control nonlinear and linear solve tolerances.
    `check_progress` enables the integrator guard that detects stalled time
    advancement after an accepted step.
    """

    scheme_tolerance: Tolerance = field(default_factory=Tolerance)
    adaptive_scheme_safety: float = 0.8
    adaptive_scheme_min_factor: float = 0.1
    adaptive_scheme_max_factor: float = 5.0
    adaptive_scheme_error_exponent: float = 0.2
    adaptive_scheme_maximum_rejections: int | None = 64
    scheme_predictor: SchemePredictorLike | None = None
    check_progress: bool = False
    resolvent_tolerance: Tolerance = field(
        default_factory=lambda: Tolerance(atol=1.0e-9, rtol=1.0e-9)
    )
    resolvent_maximum_steps: int = 16
    inverter_tolerance: Tolerance = field(default_factory=Tolerance)
    inverter_maximum_steps: int = 32

    def __post_init__(self) -> None:
        if self.adaptive_scheme_min_factor <= 0.0:
            raise ValueError("Configuration.adaptive_scheme_min_factor must be positive.")
        if self.adaptive_scheme_max_factor <= 0.0:
            raise ValueError("Configuration.adaptive_scheme_max_factor must be positive.")
        if self.adaptive_scheme_min_factor > self.adaptive_scheme_max_factor:
            raise ValueError(
                "Configuration.adaptive_scheme_min_factor must not exceed "
                "adaptive_scheme_max_factor."
            )
        if self.adaptive_scheme_safety <= 0.0:
            raise ValueError("Configuration.adaptive_scheme_safety must be positive.")
        if self.adaptive_scheme_error_exponent <= 0.0:
            raise ValueError("Configuration.adaptive_scheme_error_exponent must be positive.")
        if self.adaptive_scheme_maximum_rejections is not None and self.adaptive_scheme_maximum_rejections < 1:
            raise ValueError("Configuration.adaptive_scheme_maximum_rejections must be at least 1 or None.")
        if self.resolvent_maximum_steps < 1:
            raise ValueError("Configuration.resolvent_maximum_steps must be at least 1.")
        if self.inverter_maximum_steps < 1:
            raise ValueError("Configuration.inverter_maximum_steps must be at least 1.")


__all__ = [
    "Configuration",
]
