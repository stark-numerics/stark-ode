"""Assignment helpers for return-style dynamics.

Return-style user callables produce values rather than writing into the scheme
translation object directly. This module owns the policy for copying those
values back into frame-backed translations, mapping-shaped returns, and simple
test doubles.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

from stark.problem.frame.path import FieldPath


def assign_returned_translation(result: Any, out: Any) -> None:
    """Copy a return-style dynamics result into the scheme translation object."""

    if result is None:
        raise TypeError("Return-style dynamics must return translation values.")

    frame = getattr(out, "frame", None)
    fields_ = getattr(frame, "fields", None)
    if fields_ is not None:
        field_tuple = tuple(fields_)
        values = _returned_values_for_frame(result, field_tuple)
        for field_, value in zip(field_tuple, values, strict=True):
            field_.translation_path.assign(out, value)
        return

    _assign_returned_without_frame(result, out)


def assign_returned_fields(result: Any, out: Any, paths: tuple[str, ...]) -> None:
    """Copy a pure field-kernel result into explicit translation paths."""

    if result is None:
        raise TypeError("Return-style kernels must return translation values.")

    normalized = tuple(FieldPath(path) for path in paths)
    if isinstance(result, Mapping):
        values = [_mapping_value(result, path, path) for path in normalized]
    elif len(normalized) == 1:
        values = [result]
    elif isinstance(result, tuple | list) and len(result) == len(normalized):
        values = list(result)
    else:
        values = [_object_value(result, path, path) for path in normalized]

    for path, value in zip(normalized, values, strict=True):
        path.assign(out, value)


def _returned_values_for_frame(result: Any, fields_: tuple[Any, ...]) -> tuple[Any, ...]:
    """Return translation values from a user result, aligned to frame fields."""

    if isinstance(result, Mapping):
        return tuple(
            _mapping_value(result, field_.translation_path, field_.state_path)
            for field_ in fields_
        )
    if len(fields_) == 1:
        field_ = fields_[0]
        value = _object_value_or_missing(result, field_.translation_path, field_.state_path)
        if value is not _MISSING:
            return (value,)
        return (result,)
    if isinstance(result, tuple | list) and len(result) == len(fields_):
        return tuple(result)
    return tuple(
        _object_value(result, field_.translation_path, field_.state_path)
        for field_ in fields_
    )


def _mapping_value(
    result: Mapping[Any, Any],
    translation_path: FieldPath,
    state_path: FieldPath,
) -> Any:
    """Read one frame field from a mapping-style dynamics return value."""

    for key in _path_keys(translation_path, state_path):
        if key in result:
            return result[key]
    raise KeyError(
        "Return-style dynamics did not provide translation field "
        f"{translation_path!s}."
    )


def _object_value(
    result: Any,
    translation_path: FieldPath,
    state_path: FieldPath,
) -> Any:
    """Read one frame field from an object-style dynamics return value."""

    value = _object_value_or_missing(result, translation_path, state_path)
    if value is _MISSING:
        raise AttributeError(
            "Return-style dynamics did not provide translation field "
            f"{translation_path!s}."
        )
    return value


def _object_value_or_missing(
    result: Any,
    translation_path: FieldPath,
    state_path: FieldPath,
) -> Any:
    """Read one frame field from an object-style result if present."""

    for path in (translation_path, state_path):
        try:
            return path(result)
        except AttributeError:
            continue
    return _MISSING


def _path_keys(
    translation_path: FieldPath,
    state_path: FieldPath,
) -> tuple[Any, ...]:
    """Mapping keys accepted for one frame field."""

    keys: list[Any] = []
    for path in (translation_path, state_path):
        keys.extend((str(path), path.name, path.parts))
    return tuple(dict.fromkeys(keys))


def _assign_returned_without_frame(result: Any, out: Any) -> None:
    """Best-effort return assignment for tests and simple ad-hoc translations."""

    if isinstance(result, Mapping):
        for key, value in result.items():
            if isinstance(key, str) and key.isidentifier():
                setattr(out, key, value)
            else:
                raise TypeError(
                    "Mapping-style dynamics returns require string identifier keys."
                )
        return

    names = _writable_field_names(out)
    if len(names) == 1:
        setattr(out, names[0], result)
        return

    raise TypeError(
        "Return-style dynamics assignment requires a frame-backed translation "
        "object, a mapping return value, or an output object with one writable field."
    )


def _writable_field_names(out: Any) -> tuple[str, ...]:
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
]
