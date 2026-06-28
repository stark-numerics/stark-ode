"""Embedded adaptive Runge-Kutta schemes.

Use these as the default first-contact family for non-stiff IVPs. Embedded
pairs estimate local error and choose the next step size automatically.
`SchemeCashKarp`, `SchemeDormandPrince`, and `SchemeTsitouras5` are the main
practical choices; lower-order pairs are useful when cheap coarse control is
enough.
"""

from stark.methods.schemes.explicit.adaptive.bogacki_shampine import BS23_TABLEAU, SchemeBogackiShampine
from stark.methods.schemes.explicit.adaptive.cash_karp import RKCK_TABLEAU, SchemeCashKarp
from stark.methods.schemes.explicit.adaptive.dormand_prince import RKDP_TABLEAU, SchemeDormandPrince
from stark.methods.schemes.explicit.adaptive.fehlberg45 import RKF45_TABLEAU, SchemeFehlberg45
from stark.methods.schemes.explicit.adaptive.tsitouras5 import TSIT5_TABLEAU, SchemeTsitouras5

__all__ = [
    "BS23_TABLEAU",
    "RKCK_TABLEAU",
    "RKDP_TABLEAU",
    "RKF45_TABLEAU",
    "TSIT5_TABLEAU",
    "SchemeBogackiShampine",
    "SchemeCashKarp",
    "SchemeDormandPrince",
    "SchemeFehlberg45",
    "SchemeTsitouras5",
]









