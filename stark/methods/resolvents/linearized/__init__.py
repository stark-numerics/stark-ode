"""Linearized resolvents for implicit stage equations.

This is the main public family for serious implicit solves. Newton rebuilds
the differential as it iterates and is the robust baseline. Chord reuses a
linearization during a stage solve to reduce repeated linearization cost.
VeryChord pushes reuse further and is more problem-dependent.
"""

from stark.methods.resolvents.linearized.chord import ResolventChord
from stark.methods.resolvents.linearized.coupled_newton import ResolventCoupledNewton
from stark.methods.resolvents.linearized.newton import ResolventNewton
from stark.methods.resolvents.linearized.very_chord import ResolventVeryChord

__all__ = ["ResolventChord", "ResolventCoupledNewton", "ResolventNewton", "ResolventVeryChord"]
