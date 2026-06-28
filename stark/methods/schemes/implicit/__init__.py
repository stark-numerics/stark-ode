"""Implicit scheme families.

Implicit schemes create stage equations that must be resolved by a resolvent.
They cost more per step than explicit schemes but are the serious path for
stiff problems, dissipative dynamics, and models where explicit stability
limits would force tiny steps.
"""

__all__: list[str] = []
