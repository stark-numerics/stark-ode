"""Adaptive implicit schemes.

Use these for stiff problems where the step size should be selected from local
error evidence. The Kvaerno family is currently the strongest public face of
STARK's built-in implicit stack; `SchemeKvaerno5` is the method to try first
when accuracy and robust adaptive behaviour matter.
"""

from stark.methods.schemes.implicit.adaptive.bdf2 import SchemeBDF2
from stark.methods.schemes.implicit.adaptive.kvaerno3 import KVAERNO3_TABLEAU, SchemeKvaerno3
from stark.methods.schemes.implicit.adaptive.kvaerno4 import KVAERNO4_TABLEAU, SchemeKvaerno4
from stark.methods.schemes.implicit.adaptive.kvaerno5 import KVAERNO5_TABLEAU, SchemeKvaerno5
from stark.methods.schemes.implicit.adaptive.sdirk21 import SDIRK21_TABLEAU, SchemeSDIRK21

__all__ = [
    "KVAERNO3_TABLEAU",
    "KVAERNO4_TABLEAU",
    "KVAERNO5_TABLEAU",
    "SDIRK21_TABLEAU",
    "SchemeBDF2",
    "SchemeKvaerno3",
    "SchemeKvaerno4",
    "SchemeKvaerno5",
    "SchemeSDIRK21",
]








