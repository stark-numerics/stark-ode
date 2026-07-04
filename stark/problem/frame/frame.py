"""User-facing frame declarations for structured state.

A `Frame` names the state fields a model owns, the translation fields where
dynamics are written, each field's storage shape, and the norm policy used
by adaptive methods. Engines translate this declaration into backend-specific
allocation and algebra kernels.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from stark.engines.shared.algebraist.frame import (
    AlgebraistFrame,
    AlgebraistFrameBroadcast,
    AlgebraistFrameField,
    AlgebraistFrameLooped,
    AlgebraistFrameNormPolicy,
    AlgebraistFramePolicy,
    AlgebraistFrameScalar,
    AlgebraistFrameUnravel,
)
from stark.engines.shared.algebraist.frame.path import AlgebraistFramePathLike
from stark.problem.frame.norm import FrameNormPolicy, FrameNormRMS

FrameFieldSpec = tuple[
    AlgebraistFramePathLike,
    AlgebraistFramePathLike,
    tuple[int, ...] | list[int],
]


@dataclass(frozen=True, slots=True)
class FrameField:
    """One user-facing state field in a STARK frame."""

    state: AlgebraistFramePathLike
    translation: AlgebraistFramePathLike | None = None
    shape: tuple[int, ...] | list[int] | None = None
    norm: FrameNormPolicy | AlgebraistFrameNormPolicy = field(
        default_factory=FrameNormRMS
    )

    def __post_init__(self) -> None:
        if self.translation is None:
            object.__setattr__(self, "translation", self.state)

    def to_algebraist_field(self) -> AlgebraistFrameField:
        translation = self.translation
        if translation is None:
            translation = self.state
        return AlgebraistFrameField(
            translation,
            self.state,
            policy=self.algebraist_policy(),
            norm=self.algebraist_norm(),
        )

    def algebraist_norm(self) -> AlgebraistFrameNormPolicy:
        norm = self.norm
        if hasattr(norm, "to_algebraist_norm"):
            return cast(FrameNormPolicy, norm).to_algebraist_norm()
        return norm

    def algebraist_policy(self) -> AlgebraistFramePolicy:
        if self.shape is not None:
            return AlgebraistFrameLooped(shape=self.shape)
        return AlgebraistFrameBroadcast()


@dataclass(frozen=True, slots=True)
class Frame:
    """
    User-facing declaration of structured state and translation fields.

    A frame tells an engine which state paths exist, which translation paths
    hold their updates, what shape each field has, and how each field contributes
    to norms. It accepts explicit `FrameField` objects, simple path names,
    or a mapping such as `{"y": {"translation": "dy", "shape": (2,)}}`.
    """

    fields: tuple[FrameField, ...]

    def __init__(
        self,
        fields: FrameField
        | AlgebraistFramePathLike
        | Mapping[AlgebraistFramePathLike, Any]
        | Iterable[FrameField | AlgebraistFramePathLike | Mapping[str, Any]],
    ) -> None:
        if isinstance(fields, Mapping):
            field_mapping = cast(Mapping[AlgebraistFramePathLike, Any], fields)
            normalized = tuple(
                self._field_from_mapping_item(state, spec)
                for state, spec in field_mapping.items()
            )
        elif isinstance(fields, FrameField) or isinstance(fields, str):
            normalized = (self._coerce_field(fields),)
        else:
            normalized = tuple(self._coerce_field(field) for field in fields)
        if not normalized:
            raise ValueError("Frame requires at least one field.")

        object.__setattr__(self, "fields", normalized)
        self.to_algebraist_frame()

    @classmethod
    def scalar(
        cls,
        state: AlgebraistFramePathLike,
        *,
        translation: AlgebraistFramePathLike | None = None,
        norm: FrameNormPolicy | AlgebraistFrameNormPolicy | None = None,
    ) -> "Frame":
        """Build a one-field frame for scalar-like state storage.

        This is a convenience spelling for the full mapping syntax. The
        resulting field still has shape `(1,)`, matching the single-entry
        array style used by the getting-started examples.
        """

        spec: dict[str, Any] = {
            "translation": translation if translation is not None else state,
            "shape": (1,),
        }
        if norm is not None:
            spec["norm"] = norm
        return cls({state: spec})

    @classmethod
    def vector(
        cls,
        state: AlgebraistFramePathLike,
        *,
        translation: AlgebraistFramePathLike | None = None,
        length: int,
        norm: FrameNormPolicy | AlgebraistFrameNormPolicy | None = None,
    ) -> "Frame":
        """Build a one-field frame for vector state storage.

        This is a convenience spelling for `Frame({state: {"translation": ...,
        "shape": (length,)}})`.
        """

        spec: dict[str, Any] = {
            "translation": translation if translation is not None else state,
            "shape": (length,),
        }
        if norm is not None:
            spec["norm"] = norm
        return cls({state: spec})

    @classmethod
    def array(
        cls,
        state: AlgebraistFramePathLike,
        *,
        translation: AlgebraistFramePathLike | None = None,
        shape: tuple[int, ...] | list[int],
        norm: FrameNormPolicy | AlgebraistFrameNormPolicy | None = None,
    ) -> "Frame":
        """Build a one-field frame for array state storage.

        Use `array` when the field is naturally an array and its dimensionality
        matters to the model. This is a convenience spelling for
        `Frame({state: {"translation": ..., "shape": shape}})`.
        """

        spec: dict[str, Any] = {
            "translation": translation if translation is not None else state,
            "shape": tuple(shape),
        }
        if norm is not None:
            spec["norm"] = norm
        return cls({state: spec})

    @classmethod
    def from_fields(cls, *fields: FrameField | FrameFieldSpec) -> "Frame":
        """Build a frame from compact `(state, translation, shape)` entries.

        This is a convenience spelling for small multi-field systems where the
        full mapping syntax is visually heavier than the declaration itself.
        For advanced field options, pass `FrameField(...)` objects or use the
        full `Frame({...})` mapping syntax.
        """

        return cls(tuple(cls._coerce_field_spec(field) for field in fields))

    @staticmethod
    def _coerce_field_spec(field: FrameField | FrameFieldSpec) -> FrameField:
        if isinstance(field, FrameField):
            return field
        state, translation, shape = field
        return FrameField(state, translation=translation, shape=shape)

    @staticmethod
    def _coerce_field(
        field: FrameField | AlgebraistFramePathLike | Mapping[str, Any],
    ) -> FrameField:
        if isinstance(field, FrameField):
            return field
        if isinstance(field, Mapping):
            return Frame._field_from_spec(field)
        return FrameField(field)

    @staticmethod
    def _field_from_spec(spec: Mapping[str, Any]) -> FrameField:
        if "state" not in spec:
            raise ValueError("Frame field mappings require a 'state' entry.")
        kwargs = Frame._field_kwargs(spec)
        return FrameField(spec["state"], **kwargs)

    @staticmethod
    def _field_from_mapping_item(
        state: AlgebraistFramePathLike,
        spec: Any,
    ) -> FrameField:
        if spec is None:
            return FrameField(state)
        if not isinstance(spec, Mapping):
            raise TypeError(
                "Frame mapping values must be field option mappings or None."
            )
        kwargs = Frame._field_kwargs(spec)
        return FrameField(state, **kwargs)

    @staticmethod
    def _field_kwargs(spec: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {"state", "translation", "shape", "norm"}
        unsupported = tuple(name for name in spec if name not in allowed)
        if unsupported:
            names = ", ".join(str(name) for name in unsupported)
            raise ValueError(f"Unsupported Frame field option(s): {names}.")
        return {
            name: spec[name]
            for name in ("translation", "shape", "norm")
            if name in spec
        }

    def __iter__(self):
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def to_algebraist_frame(self) -> AlgebraistFrame:
        return AlgebraistFrame(field.to_algebraist_field() for field in self.fields)


__all__ = [
    "Frame",
    "FrameField",
]
