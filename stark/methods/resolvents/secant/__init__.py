"""Secant-style resolvents for implicit stage equations.

Anderson and Broyden methods reuse iteration history instead of requiring a
fresh linearization at every step. This family is in development: it is public
for experimentation and contribution, but it should not be a first-contact
choice until examples and benchmarks show where it is reliably competitive.
"""

from stark.methods.resolvents.secant.anderson import (
    ResolventAnderson,
    ResolventAndersonHistory,
)
from stark.methods.resolvents.secant.broyden import (
    ResolventBroyden,
    ResolventBroydenHistory,
)

__all__ = [
    "ResolventAnderson",
    "ResolventAndersonHistory",
    "ResolventBroyden",
    "ResolventBroydenHistory",
]
