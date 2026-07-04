"""Fixed-step IMEX Runge-Kutta schemes.

This package currently contains the compact IMEX Euler baseline. It is useful
for smoke tests, teaching the split-dynamics contract, and simple operator
splitting experiments; richer fixed-step IMEX methods are future work.
"""

from stark.methods.schemes.imex.fixed.euler import IMEX_EULER_TABLEAU, SchemeIMEXEuler

__all__ = ["IMEX_EULER_TABLEAU", "SchemeIMEXEuler"]









