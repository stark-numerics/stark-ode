"""Assignment helpers for return-style derivatives.

Return-style user callables produce values rather than writing into the scheme
translation object directly. This module owns the policy for copying those
values back into frame-backed translations, mapping-shaped returns, and simple
test doubles.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

from stark.engines.shared.algebraist.frame.path import AlgebraistFramePath


def assign_returned_translation(result: Any, out: Any) -> None:
    """Copy a return-style derivative result into the scheme translation object."""

    if result is None:
        raise TypeError("Return-style derivatives must return translation values.")

    frame = getattr(out, "algebraist_frame", None)
    fields_ = getattr(frame, "fields", None)
    if fields_ is not None:
        field_tuple = tuple(fields_)
        values = returned_values_for_layout(result, field_tuple)
        for field_, value in zip(field_tuple, values, strict=True):
            field_.translation_path.set(out, value)
        return

    assign_returned_without_layout(result, out)


def assign_returned_fields(result: Any, out: Any, paths: tuple[str, ...]) -> None:
    """Copy a pure field-kernel result into explicit translation paths."""

    if result is None:
        raise TypeError("Return-style kernels must return translation values.")

    normalized = tuple(AlgebraistFramePath(path) for path in paths)
    if isinstance(result, Mapping):
        values = [mapping_value(result, path, path) for path in normalized]
    elif len(normalized) == 1:
        values = [result]
    elif isinstance(result, tuple | list) and len(result) == len(normalized):
        values = list(result)
    else:
        values = [object_value(result, path, path) for path in normalized]

    for path, value in zip(normalized, values, strict=True):
        path.set(out, value)


def returned_values_for_layout(result: Any, fields_: tuple[Any, ...]) -> tuple[Any, ...]:
    """Return translation values from a user result, aligned to frame fields."""

    if isinstance(result, Mapping):
        return tuple(
            mapping_value(result, field_.translation_path, field_.state_path)
            for field_ in fields_
        )
    if len(fields_) == 1:
        field_ = fields_[0]
        value = object_value_or_missing(result, field_.translation_path, field_.state_path)
        if value is not _MISSING:
            return (value,)
        return (result,)
    if isinstance(result, tuple | list) and len(result) == len(fields_):
        return tuple(result)
    return tuple(
        object_value(result, field_.translation_path, field_.state_path)
        for field_ in fields_
    )


def mapping_value(
    result: Mapping[Any, Any],
    translation_path: AlgebraistFramePath,
    state_path: AlgebraistFramePath,
) -> Any:
    """Read one frame field from a mapping-style derivative return value."""

    for key in path_keys(translation_path, state_path):
        if key in result:
            return result[key]
    raise KeyError(
        "Return-style derivative did not provide translation field "
        f"{translation_path!s}."
    )


def object_value(
    result: Any,
    translation_path: AlgebraistFramePath,
    state_path: AlgebraistFramePath,
) -> Any:
    """Read one frame field from an object-style derivative return value."""

    value = object_value_or_missing(result, translation_path, state_path)
    if value is _MISSING:
        raise AttributeError(
            "Return-style derivative did not provide translation field "
            f"{translation_path!s}."
        )
    return value


def object_value_or_missing(
    result: Any,
    translation_path: AlgebraistFramePath,
    state_path: AlgebraistFramePath,
) -> Any:
    """Read one frame field from an object-style result if present."""

    for path in (translation_path, state_path):
        try:
            return path.get(result)
        except AttributeError:
            continue
    return _MISSING


def path_keys(
    translation_path: AlgebraistFramePath,
    state_path: AlgebraistFramePath,
) -> tuple[Any, ...]:
    """Mapping keys accepted for one frame field."""

    keys: list[Any] = []
    for path in (translation_path, state_path):
        keys.extend((str(path), path.name, path.parts))
    return tuple(dict.fromkeys(keys))


def assign_returned_without_layout(result: Any, out: Any) -> None:
    """Best-effort return assignment for tests and simple ad-hoc translations."""

    if isinstance(result, Mapping):
        for key, value in result.items():
            if isinstance(key, str) and key.isidentifier():
                setattr(out, key, value)
            else:
                raise TypeError(
                    "Mapping-style derivative returns require string identifier keys."
                )
        return

    names = writable_field_names(out)
    if len(names) == 1:
        setattr(out, names[0], result)
        return

    raise TypeError(
        "Return-style derivative assignment requires a frame-backed translation "
        "object, a mapping return value, or an output object with one writable field."
    )


def writable_field_names(out: Any) -> tuple[str, ...]:
    """Return likely writable public field names for a translation-like object."""

    if is_dataclass(out):
        return tuple(field_.name for field_ in fields(out) if field_.init)

    slots = getattr(type(out), "__slots__", ())
    if isinstance(slots, str):
        slots = (slots,)
    if slots:
        return tuple(name for name in slots if not name.startswith("_"))

    dictionary = getattr(out, "__dict__", None)
    if dictionary is not None:
        return tuple(name for name in dictionary if not name.startswith("_"))

    return ()


_MISSING = object()


__all__ = [
    "assign_returned_fields",
    "assign_returned_translation",
    "assign_returned_without_layout",
    "mapping_value",
    "object_value",
    "object_value_or_missing",
    "path_keys",
    "returned_values_for_layout",
    "writable_field_names",
]
