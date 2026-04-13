from __future__ import annotations

from typing import Any, Protocol

from stark.contract_protocols.intervals import IntervalLike
from stark.contract_protocols.linear_algebra import State, Translation
from stark.tolerance import Tolerance
from stark.scheme_support.descriptor import SchemeDescriptor


class Workbench(Protocol):
    """
    Factory for reusable scratch objects and state-copy operations.

    This is the main integration point for user-defined state types. A custom
    workbench tells STARK how to:

    - allocate mutable state objects
    - copy one state into another
    - allocate translation objects compatible with that state

    Once this contract is satisfied, the built-in schemes, resolvers, and
    inverters can reuse those objects without knowing the concrete state shape.
    """

    def allocate_state(self) -> State:
        ...

    def copy_state(self, dst: State, src: State) -> None:
        ...

    def allocate_translation(self) -> Translation:
        ...


class SchemeLike(Protocol):
    """
    Minimal protocol accepted by STARK for one-step integration schemes.

    A custom scheme does not need to inherit from STARK's internal helper
    classes as long as it satisfies this interface.

    The scheme is responsible for mutating `state` forward by one accepted step
    and returning the step size that was actually taken. Adaptive schemes may
    also update `interval.step` to propose the next step size.
    """

    def __call__(self, interval: IntervalLike, state: State, tolerance: Tolerance) -> float:
        ...

    def snapshot_state(self, state: State) -> State:
        ...

    def set_apply_delta_safety(self, enabled: bool) -> None:
        ...


class Scheme(SchemeLike, Protocol):
    """
    Richer scheme protocol for STARK's built-in, tableau-backed schemes.

    This adds the descriptor and tableau metadata used for readable reporting,
    table display, and package-level exports.
    """

    descriptor: SchemeDescriptor
    tableau: Any

    @classmethod
    def display_tableau(cls) -> str:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def __format__(self, format_spec: str) -> str:
        ...


__all__ = ["Scheme", "SchemeLike", "Workbench"]

