"""Built-in scheme classes."""

from stark.schemes.explicit.adaptive import (
    SchemeBogackiShampine,
    SchemeCashKarp,
    SchemeDormandPrince,
    SchemeFehlberg45,
    SchemeTsitouras5,
)
from stark.schemes.explicit.fixed import (
    SchemeEuler,
    SchemeHeun,
    SchemeKutta3,
    SchemeMidpoint,
    SchemeRK4,
    SchemeRK38,
    SchemeRalston,
    SchemeSSPRK33,
)
from stark.schemes.imex.adaptive import (
    SchemeKennedyCarpenter32,
    SchemeKennedyCarpenter43_6,
    SchemeKennedyCarpenter43_7,
    SchemeKennedyCarpenter54,
    SchemeKennedyCarpenter54b,
)
from stark.schemes.imex.fixed import SchemeIMEXEuler
from stark.schemes.implicit.adaptive import (
    SchemeBDF2,
    SchemeKvaerno3,
    SchemeKvaerno4,
    SchemeSDIRK21,
)
from stark.schemes.implicit.fixed import (
    SchemeBackwardEuler,
    SchemeCrankNicolson,
    SchemeCrouzeixDIRK3,
    SchemeGaussLegendre4,
    SchemeImplicitMidpoint,
    SchemeLobattoIIIC4,
    SchemeRadauIIA5,
)

__all__ = [
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
    "SchemeLobattoIIIC4",
    "SchemeMidpoint",
    "SchemeRK4",
    "SchemeRK38",
    "SchemeRadauIIA5",
    "SchemeRalston",
    "SchemeSDIRK21",
    "SchemeSSPRK33",
    "SchemeTsitouras5",
]
