"""Built-in time-stepping scheme catalogue.

Scheme families answer different modelling questions:

- explicit fixed schemes are simple, predictable-cost steppers for non-stiff
  problems or externally controlled step sizes;
- explicit adaptive schemes are the usual first choice for non-stiff problems
  when automatic step-size control is helpful;
- implicit fixed schemes target stiff or constraint-like problems where the
  caller controls the step size;
- implicit adaptive schemes target stiff problems where automatic step-size
  control matters;
- IMEX schemes target split derivatives where one part is best treated
  explicitly and another part benefits from implicit treatment.
"""

from stark.methods.schemes.explicit.adaptive.bogacki_shampine import SchemeBogackiShampine
from stark.methods.schemes.explicit.adaptive.cash_karp import SchemeCashKarp
from stark.methods.schemes.explicit.adaptive.dormand_prince import SchemeDormandPrince
from stark.methods.schemes.explicit.adaptive.fehlberg45 import SchemeFehlberg45
from stark.methods.schemes.explicit.adaptive.tsitouras5 import SchemeTsitouras5
from stark.methods.schemes.explicit.fixed.euler import SchemeEuler
from stark.methods.schemes.explicit.fixed.heun import SchemeHeun
from stark.methods.schemes.explicit.fixed.kutta3 import SchemeKutta3
from stark.methods.schemes.explicit.fixed.midpoint import SchemeMidpoint
from stark.methods.schemes.explicit.fixed.ralston import SchemeRalston
from stark.methods.schemes.explicit.fixed.rk38 import SchemeRK38
from stark.methods.schemes.explicit.fixed.rk4 import SchemeRK4
from stark.methods.schemes.explicit.fixed.ssprk33 import SchemeSSPRK33
from stark.methods.schemes.imex.adaptive.kennedy_carpenter32 import SchemeKennedyCarpenter32
from stark.methods.schemes.imex.adaptive.kennedy_carpenter43_6 import SchemeKennedyCarpenter43_6
from stark.methods.schemes.imex.adaptive.kennedy_carpenter43_7 import SchemeKennedyCarpenter43_7
from stark.methods.schemes.imex.adaptive.kennedy_carpenter54 import SchemeKennedyCarpenter54
from stark.methods.schemes.imex.adaptive.kennedy_carpenter54b import SchemeKennedyCarpenter54b
from stark.methods.schemes.imex.fixed.euler import SchemeIMEXEuler
from stark.methods.schemes.implicit.adaptive.bdf2 import SchemeBDF2
from stark.methods.schemes.implicit.adaptive.kvaerno3 import SchemeKvaerno3
from stark.methods.schemes.implicit.adaptive.kvaerno4 import SchemeKvaerno4
from stark.methods.schemes.implicit.adaptive.kvaerno5 import SchemeKvaerno5
from stark.methods.schemes.implicit.adaptive.sdirk21 import SchemeSDIRK21
from stark.methods.schemes.implicit.fixed.backward_euler import SchemeBackwardEuler
from stark.methods.schemes.implicit.fixed.crank_nicolson import SchemeCrankNicolson
from stark.methods.schemes.implicit.fixed.crouzeix_dirk3 import SchemeCrouzeixDIRK3
from stark.methods.schemes.implicit.fixed.gauss_legendre4 import SchemeGaussLegendre4
from stark.methods.schemes.implicit.fixed.implicit_midpoint import SchemeImplicitMidpoint
from stark.methods.schemes.implicit.fixed.lobatto_iiic4 import SchemeLobattoIIIC4
from stark.methods.schemes.implicit.fixed.radau_iia5 import SchemeRadauIIA5

__all__ = (
    "SchemeBDF2",
    "SchemeBackwardEuler",
    "SchemeBogackiShampine",
    "SchemeCashKarp",
    "SchemeCrankNicolson",
    "SchemeCrouzeixDIRK3",
    "SchemeDormandPrince",
    "SchemeEuler",
    "SchemeFehlberg45",
    "SchemeGaussLegendre4",
    "SchemeHeun",
    "SchemeIMEXEuler",
    "SchemeImplicitMidpoint",
    "SchemeKennedyCarpenter32",
    "SchemeKennedyCarpenter43_6",
    "SchemeKennedyCarpenter43_7",
    "SchemeKennedyCarpenter54",
    "SchemeKennedyCarpenter54b",
    "SchemeKutta3",
    "SchemeKvaerno3",
    "SchemeKvaerno4",
    "SchemeKvaerno5",
    "SchemeLobattoIIIC4",
    "SchemeMidpoint",
    "SchemeRK4",
    "SchemeRK38",
    "SchemeRadauIIA5",
    "SchemeRalston",
    "SchemeSDIRK21",
    "SchemeSSPRK33",
    "SchemeTsitouras5",
)
