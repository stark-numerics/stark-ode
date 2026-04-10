from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, Self, TypeAlias, TypeVar

from stark.control import Tolerance
from stark.scheme_descriptor import SchemeDescriptor

State = TypeVar("State")


class IntervalLike(Protocol):
    """
    Protocol for a rolling integration interval.

    User-defined intervals are fine as long as they expose the same attributes
    and behavior as STARK's primitive `Interval`.

    Required fields:
    - `present`: the current integration time
    - `step`: the next trial step size
    - `stop`: the terminal time for the current integration run

    Required methods:
    - `increment(dt)`: advance `present` by the accepted step size `dt`
    - `copy()`: return a snapshot copy of the interval

    Notes:
    - adaptive schemes may update `step` in place to propose the next trial step
    - snapshot integration relies on `copy()`
    - live integration only needs the mutable rolling object itself
    """

    present: float
    step: float
    stop: float

    def copy(self) -> Self:
        ...

    def increment(self, dt: float) -> None:
        ...


class Translation(Protocol):
    """
    A state update object carrying the linear structure of the problem.

    STARK separates nonlinear mutable state from linear translation objects.
    Schemes build weighted combinations of translations, and a translation can
    then be applied to a state to produce an updated state.

    Required behavior:
    - `__call__(origin, result)` applies the translation to `origin` and writes
      the translated state into `result`
    - `norm()` returns the size of the translation for adaptive error control
    - `__add__` and `__rmul__` provide the generic linear-combination fallback

    Optional optimized behavior:
    - a translation may define
      `linear_combine = [scale, combine2, combine3, ...]`
    - these callables should satisfy the `Scale`, `Combine2`, `Combine3`, ...
      contracts exported by this module
    - if present, STARK will use them instead of the generic `__add__` /
      `__rmul__` fallback when forming weighted sums of translations
    - if `combine2` is present but higher arities are omitted, STARK builds
      the missing higher arities from the lower-arity kernels and scratch
      translations allocated by the workbench
    - this is especially useful for array-backed translations that can provide
      fused kernels or compiled implementations

    Important aliasing rule:
    - translation application must be correct even when `origin is result`
    - if optimized `linear_combine` kernels are supplied, they should also be
      correct when the output buffer aliases an input buffer

    The aliasing requirement is what lets STARK support a fast in-place path
    while still allowing safe snapshotting through the workbench copy hooks.
    """

    def __call__(self, origin: State, result: State) -> None:
        ...

    def norm(self) -> float:
        ...

    def __add__(self, other: Self) -> Self:
        ...

    def __rmul__(self, scalar: float) -> Self:
        ...


Derivative: TypeAlias = Callable[[State, Translation], None]


class Scale(Protocol):
    """Set `out = a * x` and return `out`."""

    def __call__(self, out: Translation, a: float, x: Translation) -> Translation:
        ...


class Combine2(Protocol):
    """Set `out = a0 * x0 + a1 * x1` and return `out`."""

    def __call__(
        self,
        out: Translation,
        a0: float,
        x0: Translation,
        a1: float,
        x1: Translation,
    ) -> Translation:
        ...


class Combine3(Protocol):
    """Set `out = a0 * x0 + a1 * x1 + a2 * x2` and return `out`."""

    def __call__(
        self,
        out: Translation,
        a0: float,
        x0: Translation,
        a1: float,
        x1: Translation,
        a2: float,
        x2: Translation,
    ) -> Translation:
        ...


class Combine4(Protocol):
    """Set `out = a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3` and return `out`."""

    def __call__(
        self,
        out: Translation,
        a0: float,
        x0: Translation,
        a1: float,
        x1: Translation,
        a2: float,
        x2: Translation,
        a3: float,
        x3: Translation,
    ) -> Translation:
        ...


class Combine5(Protocol):
    """Set `out = a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4` and return `out`."""

    def __call__(
        self,
        out: Translation,
        a0: float,
        x0: Translation,
        a1: float,
        x1: Translation,
        a2: float,
        x2: Translation,
        a3: float,
        x3: Translation,
        a4: float,
        x4: Translation,
    ) -> Translation:
        ...


class Combine6(Protocol):
    """Set `out = a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5` and return `out`."""

    def __call__(
        self,
        out: Translation,
        a0: float,
        x0: Translation,
        a1: float,
        x1: Translation,
        a2: float,
        x2: Translation,
        a3: float,
        x3: Translation,
        a4: float,
        x4: Translation,
        a5: float,
        x5: Translation,
    ) -> Translation:
        ...


class Combine7(Protocol):
    """Set `out = a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6` and return `out`."""

    def __call__(
        self,
        out: Translation,
        a0: float,
        x0: Translation,
        a1: float,
        x1: Translation,
        a2: float,
        x2: Translation,
        a3: float,
        x3: Translation,
        a4: float,
        x4: Translation,
        a5: float,
        x5: Translation,
        a6: float,
        x6: Translation,
    ) -> Translation:
        ...


class Workbench(Protocol):
    """
    Factory for reusable scratch objects and state-copy operations.

    This is the main integration point for user-defined state types.
    Schemes use the workbench to allocate scratch states/translations once and
    then reuse them across many steps.

    Required methods:
    - `allocate_state()`: return a blank mutable state object
    - `copy_state(dst, src)`: overwrite `dst` with the contents of `src`
    - `allocate_translation()`: return a blank translation object

    Notes:
    - `copy_state` is required for snapshot integration
    - `copy_state` is also used internally to support alias-safe state updates
    - for array-heavy problems, this is a good place to centralize fast copy
      behavior and compatible scratch-object layouts
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

    `SchemeLike` is the minimal contract needed by `Marcher`. This richer
    protocol describes the more "lived in" objects that expose metadata and
    readable string representations.
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
__all__ = [
    "Combine2",
    "Combine3",
    "Combine4",
    "Combine5",
    "Combine6",
    "Combine7",
    "Derivative",
    "IntervalLike",
    "Scale",
    "Scheme",
    "SchemeDescriptor",
    "SchemeLike",
    "State",
    "Translation",
    "Workbench",
]
