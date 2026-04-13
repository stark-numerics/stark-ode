"""Built-in Runge-Kutta schemes.

The library is organised by stepping behaviour:

- `adaptive` contains embedded schemes with error estimates and step control.
- `adaptive_implicit` contains adaptive implicit schemes, including ESDIRK and BDF methods.
- `fixed_step` contains classic explicit fixed-step schemes.

Scheme classes and tableaus are re-exported here for the common import style:
`from stark.scheme_library import SchemeCashKarp`.
"""

from stark.scheme_library.adaptive import (
    BS23_TABLEAU,
    RKCK_TABLEAU,
    RKDP_TABLEAU,
    RKF45_TABLEAU,
    TSIT5_TABLEAU,
    SchemeBogackiShampine,
    SchemeCashKarp,
    SchemeDormandPrince,
    SchemeFehlberg45,
    SchemeRKCK,
    SchemeTsitouras5,
)
from stark.scheme_library.adaptive_implicit import (
    KVAERNO3_TABLEAU,
    KVAERNO4_TABLEAU,
    SDIRK21_TABLEAU,
    SchemeBDF2,
    SchemeKvaerno3,
    SchemeKvaerno4,
    SchemeSDIRK21,
)
from stark.scheme_library.fixed_step import (
    EULER_TABLEAU,
    HEUN_TABLEAU,
    KUTTA3_TABLEAU,
    MIDPOINT_TABLEAU,
    RALSTON_TABLEAU,
    RK4_TABLEAU,
    RK38_TABLEAU,
    SSPRK33_TABLEAU,
    SchemeEuler,
    SchemeHeun,
    SchemeKutta3,
    SchemeMidpoint,
    SchemeRK4,
    SchemeRK38,
    SchemeRalston,
    SchemeSSPRK33,
)
from stark.scheme_library.implicit import (
    BE_TABLEAU,
    SchemeBackwardEuler,
)

__all__ = [
    "BE_TABLEAU",
    "BS23_TABLEAU",
    "EULER_TABLEAU",
    "HEUN_TABLEAU",
    "KUTTA3_TABLEAU",
    "KVAERNO3_TABLEAU",
    "KVAERNO4_TABLEAU",
    "MIDPOINT_TABLEAU",
    "RALSTON_TABLEAU",
    "RK4_TABLEAU",
    "RK38_TABLEAU",
    "RKCK_TABLEAU",
    "RKDP_TABLEAU",
    "RKF45_TABLEAU",
    "SDIRK21_TABLEAU",
    "SSPRK33_TABLEAU",
    "TSIT5_TABLEAU",
    "SchemeBogackiShampine",
    "SchemeBDF2",
    "SchemeBackwardEuler",
    "SchemeCashKarp",
    "SchemeDormandPrince",
    "SchemeEuler",
    "SchemeFehlberg45",
    "SchemeHeun",
    "SchemeKutta3",
    "SchemeKvaerno3",
    "SchemeKvaerno4",
    "SchemeMidpoint",
    "SchemeRalston",
    "SchemeRK4",
    "SchemeRK38",
    "SchemeRKCK",
    "SchemeSDIRK21",
    "SchemeSSPRK33",
    "SchemeTsitouras5",
]
