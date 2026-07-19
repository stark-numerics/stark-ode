"""Contracts for structured fields inside a frame-like declaration."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol, TypeVar, overload

FieldPathLike = str | Sequence[str]
"""Path-like input for locating a field on a structured object."""

PathType = TypeVar("PathType", covariant=True)
"""Path object used to read and write one field on a state or translation."""


class FieldPath(Protocol):
    """Validated path to a field on a structured state or translation object."""

    @property
    def parts(self) -> tuple[str, ...]:
        """Attribute path segments from the root object to this field."""
        ...

    @property
    def name(self) -> str:
        """Identifier-safe flattened name for generated code parameters."""
        ...

    def expression(self, root: str) -> str:
        """Python source expression for reading this path from `root`."""
        ...

    def __call__(self, root) -> object:
        """Runtime value reached by following this path from `root`."""
        ...

    def assign(self, root, value) -> None:
        """Assign `value` at this path on `root`."""
        ...

    def ensure_parent(self, root) -> object:
        """Return or create the parent object containing the final segment."""
        ...

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[str, ...]: ...

    def __getitem__(self, index: int | slice) -> str | tuple[str, ...]: ...

    def __iter__(self) -> Iterator[str]: ...

    def __len__(self) -> int: ...


FieldPolicyType = TypeVar("FieldPolicyType", covariant=True)
"""Policy object describing how a field should be traversed or emitted."""


class FieldPolicyLike(Protocol):
    """Policy describing how a field should be interpreted by a frame."""

    @property
    def kind(self) -> str:
        """Traversal kind, such as `scalar`, `broadcast`, `looped`, or `unravel`."""
        ...


class FieldLike(Protocol[PathType, FieldPolicyType]):
    """Structural contract for one field in a frame-like declaration.

    A field is both the user declaration and the normalized layout consumed by
    allocation and algebra kernels. Domain extensions such as PDE fields should
    satisfy this contract directly while adding their own metadata.
    """

    @property
    def state(self) -> FieldPathLike:
        """State-side field path as supplied or normalized by the field."""
        ...

    @property
    def translation(self) -> FieldPathLike:
        """Translation-side field path; defaults to `state` when omitted."""
        ...

    @property
    def shape(self) -> tuple[int, ...] | None:
        """Field storage shape, or `None` for scalar-like/broadcast fields."""
        ...

    @property
    def policy(self) -> FieldPolicyType:
        """Normalized traversal policy for generated/backend algebra."""
        ...

    @property
    def translation_path(self) -> PathType:
        """Validated translation-side path used for runtime access."""
        ...

    @property
    def state_path(self) -> PathType:
        """Validated state-side path used for runtime access."""
        ...

    @property
    def translation_name(self) -> str:
        """Identifier-safe translation field name for generated code."""
        ...

    @property
    def state_name(self) -> str:
        """Identifier-safe state field name for generated code."""
        ...

    def translation_expression(self, root: str) -> str:
        """Python source expression for this field on a translation object."""
        ...

    def state_expression(self, root: str) -> str:
        """Python source expression for this field on a state object."""
        ...


__all__ = [
    "FieldPath",
    "FieldPathLike",
    "FieldLike",
    "FieldPolicyLike",
    "FieldPolicyType",
    "PathType",
]
