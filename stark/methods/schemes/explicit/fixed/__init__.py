"""Classic explicit fixed-step Runge-Kutta schemes.

Use these when a model already has a meaningful step size, when repeatable
per-step cost matters more than automatic control, or when implementing and
testing custom scheme behaviour. `SchemeRK4` is the familiar high-quality
baseline; lower-order schemes are useful for teaching, debugging, and cheap
probes.
"""

from stark.methods.schemes.explicit.fixed.euler import EULER_TABLEAU, SchemeEuler
from stark.methods.schemes.explicit.fixed.heun import HEUN_TABLEAU, SchemeHeun
from stark.methods.schemes.explicit.fixed.kutta3 import KUTTA3_TABLEAU, SchemeKutta3
from stark.methods.schemes.explicit.fixed.midpoint import MIDPOINT_TABLEAU, SchemeMidpoint
from stark.methods.schemes.explicit.fixed.ralston import RALSTON_TABLEAU, SchemeRalston
from stark.methods.schemes.explicit.fixed.rk38 import RK38_TABLEAU, SchemeRK38
from stark.methods.schemes.explicit.fixed.rk4 import RK4_TABLEAU, SchemeRK4
from stark.methods.schemes.explicit.fixed.ssprk33 import SSPRK33_TABLEAU, SchemeSSPRK33

__all__ = [
    "EULER_TABLEAU",
    "HEUN_TABLEAU",
    "KUTTA3_TABLEAU",
    "MIDPOINT_TABLEAU",
    "RALSTON_TABLEAU",
    "RK4_TABLEAU",
    "RK38_TABLEAU",
    "SSPRK33_TABLEAU",
    "SchemeEuler",
    "SchemeHeun",
    "SchemeKutta3",
    "SchemeMidpoint",
    "SchemeRalston",
    "SchemeRK4",
    "SchemeRK38",
    "SchemeSSPRK33",
]









