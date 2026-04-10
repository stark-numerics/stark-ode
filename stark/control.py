from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Tolerance:
    """
    User-facing error tolerances for adaptive stepping.

    The intended long-term model is that schemes ask this object for the
    effective tolerance they should apply to their local error estimate,
    rather than reaching straight for raw numbers.
    """

    atol: float = 1.0e-6
    rtol: float = 1.0e-6

    def __repr__(self) -> str:
        return f"Tolerance(atol={self.atol!r}, rtol={self.rtol!r})"

    def __str__(self) -> str:
        return f"atol={self.atol:g}, rtol={self.rtol:g}"

    def bound(self, scale: float) -> float:
        """
        Return the admissible absolute error bound for the supplied scale.
        """
        return self.atol + self.rtol * scale

    def ratio(self, error: float, scale: float) -> float:
        """
        Return the normalized error ratio.

        Ratios below or equal to one indicate that the error is acceptable.
        """
        return error / self.bound(scale)

    def accepts(self, error: float, scale: float) -> bool:
        """
        Return whether the supplied error is acceptable at the supplied scale.
        """
        return self.ratio(error, scale) <= 1.0


@dataclass(slots=True)
class Regulator:
    """
    User-tunable adaptive step-size regulator knobs.

    These values influence how aggressively an adaptive Runge-Kutta scheme
    grows or shrinks its next trial step after estimating local error.
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


__all__ = ["Regulator", "Tolerance"]
