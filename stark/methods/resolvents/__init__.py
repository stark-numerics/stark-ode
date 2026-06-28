"""Built-in implicit resolvent catalogue.

Resolvents solve the nonlinear stage equations created by implicit and IMEX
schemes. Fixed-point resolvents are simple and useful when the stage equation
is strongly contractive. Linearized resolvents are the main serious implicit
path. Secant-style resolvents are in development and should be treated as
advanced tools until stronger examples and safeguards settle their public
shape.
"""

from stark.methods.resolvents.method.descriptor import ResolventDescriptor
from stark.methods.resolvents.method.errors import ResolventError
from stark.methods.resolvents.configuration import ResolventConfiguration
from stark.methods.resolvents.fixed_point.coupled_picard import ResolventCoupledPicard
from stark.methods.resolvents.fixed_point.picard import ResolventPicard
from stark.methods.resolvents.linearized.chord import ResolventChord
from stark.methods.resolvents.linearized.coupled_newton import ResolventCoupledNewton
from stark.methods.resolvents.linearized.newton import ResolventNewton
from stark.methods.resolvents.linearized.very_chord import ResolventVeryChord
from stark.methods.resolvents.secant.anderson import ResolventAnderson
from stark.methods.resolvents.secant.broyden import ResolventBroyden

__all__ = [
    "ResolventAnderson",
    "ResolventBroyden",
    "ResolventChord",
    "ResolventCoupledNewton",
    "ResolventCoupledPicard",
    "ResolventNewton",
    "ResolventPicard",
    "ResolventVeryChord",
    "ResolventError",
    "ResolventDescriptor",
    "ResolventConfiguration",
]
