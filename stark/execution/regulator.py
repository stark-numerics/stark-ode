from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Regulator:
    """
    User-tunable adaptive step-size regulator knobs.

    These values influence how aggressively an adaptive scheme grows or
    shrinks its next trial step after estimating local error.
    """

    safety: float = 0.8
    min_factor: float = 0.1
    max_factor: float = 5.0
    error_exponent: float = 0.2

    def __repr__(self) -> str:
        return (
            "Regulator("
            f"safety={self.safety!r}, "
            f"min_factor={self.min_factor!r}, "
            f"max_factor={self.max_factor!r}, "
            f"error_exponent={self.error_exponent!r})"
        )

    def __str__(self) -> str:
        return (
            f"safety={self.safety:g}, "
            f"min_factor={self.min_factor:g}, "
            f"max_factor={self.max_factor:g}, "
            f"error_exponent={self.error_exponent:g}"
        )


__all__ = ["Regulator"]









