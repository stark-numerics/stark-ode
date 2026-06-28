"""Dense inverse actions for small materialised operators.

Dense inverters materialise compact block operators and solve them locally with
`InverterNucleus`. This is the preferred built-in path for small implicit
systems and a useful baseline when judging more elaborate iterative inverters.
"""

from stark.methods.inverters.dense.dense import (
    InverterDense,
    InverterDenseInstance,
    InverterDenseInstanceSingle,
)

__all__ = [
    "InverterDense",
    "InverterDenseInstance",
    "InverterDenseInstanceSingle",
]
