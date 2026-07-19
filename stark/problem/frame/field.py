"""User-facing field declarations for structured frames."""

from __future__ import annotations

from dataclasses import dataclass

from stark.core.contracts.problem.field import FieldPolicyLike
from stark.problem.frame.path import FieldPath, FieldPathLike
from stark.problem.frame.policy import FieldPolicy


@dataclass(frozen=True, slots=True, init=False)
class Field:
    """One user-facing state field in a STARK frame."""

    state: FieldPath
    translation: FieldPath
    shape: tuple[int, ...] | None
    policy: FieldPolicyLike

    def __init__(
        self,
        state: FieldPathLike,
        *,
        translation: FieldPathLike | None = None,
        shape: tuple[int, ...] | list[int] | None = None,
        policy: FieldPolicyLike | None = None,
    ) -> None:
        state_path = FieldPath(state)
        translation_path = state_path if translation is None else FieldPath(translation)
        normalized_shape = self.normalize_shape(shape)
        normalized_policy = self.normalize_policy(
            FieldPolicy() if policy is None else policy,
            shape=normalized_shape,
        )

        object.__setattr__(self, "state", state_path)
        object.__setattr__(self, "translation", translation_path)
        object.__setattr__(self, "shape", normalized_shape)
        object.__setattr__(self, "policy", normalized_policy)

    @staticmethod
    def normalize_shape(
        shape: tuple[int, ...] | list[int] | None,
    ) -> tuple[int, ...] | None:
        if shape is None:
            return None
        normalized = tuple(shape)
        if not normalized:
            raise ValueError("shape must contain at least one dimension.")
        for dimension in normalized:
            if not isinstance(dimension, int):
                raise TypeError("shape dimensions must be integers.")
            if dimension <= 0:
                raise ValueError("shape dimensions must be positive.")
        return normalized

    @staticmethod
    def normalize_policy(
        policy: FieldPolicyLike,
        *,
        shape: tuple[int, ...] | None,
    ) -> FieldPolicyLike:
        kind = policy.kind
        if kind == "auto":
            return FieldPolicy.looped() if shape is not None else FieldPolicy.broadcast()
        if kind in {"looped", "unravel"} and shape is None:
            raise ValueError(f"{kind} field policy requires shape.")
        return policy

    @property
    def translation_path(self) -> FieldPath:
        return FieldPath(self.translation)

    @property
    def state_path(self) -> FieldPath:
        return FieldPath(self.state)

    @property
    def translation_name(self) -> str:
        return self.translation_path.name

    @property
    def state_name(self) -> str:
        return self.state_path.name

    def translation_expression(self, root: str) -> str:
        return self.translation_path.expression(root)

    def state_expression(self, root: str) -> str:
        return self.state_path.expression(root)


__all__ = [
    "Field",
]
