"""Fixed-point resolvents for implicit stage equations.

Picard-style resolvents are the simplest way to solve a shifted implicit stage
equation. They are most useful when the implicit equation is already a strong
contraction or when a problem is small enough that simplicity is more valuable
than robustness. They are not the default recommendation for hard stiff
systems.
"""

from stark.methods.resolvents.fixed_point.coupled_picard import ResolventCoupledPicard
from stark.methods.resolvents.fixed_point.picard import ResolventPicard

__all__ = ["ResolventCoupledPicard", "ResolventPicard"]
