"""Implicit-explicit scheme families.

IMEX schemes require a split dynamics. They are useful when a model has a
cheap or non-stiff part that should remain explicit and a stiff part that
benefits from implicit treatment. This family is important for optimisation
and PDE-style extensions, so contributions that add examples, benchmarks, or
new split methods are especially welcome.
"""

__all__: list[str] = []
